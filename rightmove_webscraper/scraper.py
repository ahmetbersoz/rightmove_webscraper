import datetime
import json
from typing import Any, Dict
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from lxml import html
import numpy as np
import pandas as pd
import requests


BASE_URL = "https://www.rightmove.co.uk"
REQUEST_TIMEOUT = 20
PROPERTY_DETAILS_TIMEOUT = 5
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class RightmoveData:
    """The `RightmoveData` webscraper collects structured data on properties
    returned by a search performed on www.rightmove.co.uk

    An instance of the class provides attributes to access data from the search
    results, the most useful being `get_results`, which returns all results as a
    Pandas DataFrame object.

    The query to rightmove can be renewed by calling the `refresh_data` method.
    """

    def __init__(
        self,
        url: str,
        get_floorplans: bool = False,
        include_property_details: bool = False,
    ):
        """Initialize the scraper with a URL from the results of a property
        search performed on www.rightmove.co.uk.

        Args:
            url (str): full HTML link to a page of rightmove search results.
            get_floorplans (bool): optionally scrape links to the individual
                floor plan images for each listing (be warned this drastically
                increases runtime so is False by default).
            include_property_details (bool): when True, request each individual
                property page to enrich the results with additional letting
                details (increases runtime).
        """
        self._session = requests.Session()
        self._status_code, self._first_page = self._request(url)
        self._url = url
        self._validate_url()
        self._include_property_details = include_property_details
        self._property_details_cache: Dict[str, Dict[str, Any]] = {}
        self._first_search_results = self._extract_search_results(self._first_page)
        self._results = self._get_results(get_floorplans=get_floorplans)

    def _request(self, url: str, *, timeout: float | None = None):
        request_timeout = timeout if timeout is not None else REQUEST_TIMEOUT
        response = self._session.get(
            url, headers=REQUEST_HEADERS, timeout=request_timeout
        )
        return response.status_code, response.content

    def refresh_data(
        self,
        url: str = None,
        get_floorplans: bool = False,
        include_property_details: bool | None = None,
    ):
        """Make a fresh GET request for the rightmove data.

        Args:
            url (str): optionally pass a new HTML link to a page of rightmove
                search results (else defaults to the current `url` attribute).
            get_floorplans (bool): optionally scrape links to the individual
                flooplan images for each listing (this drastically increases
                runtime so is False by default).
            include_property_details (bool | None): optionally override the
                existing behaviour for fetching individual property pages.
        """
        if include_property_details is not None:
            self._include_property_details = include_property_details
        url = self.url if not url else url
        self._status_code, self._first_page = self._request(url)
        self._url = url
        self._validate_url()
        self._property_details_cache = {}
        self._first_search_results = self._extract_search_results(self._first_page)
        self._results = self._get_results(get_floorplans=get_floorplans)

    def _validate_url(self):
        """Basic validation that the URL at least starts in the right format and
        returns status code 200."""
        real_url = "{}://www.rightmove.co.uk/{}/find.html?"
        protocols = ["http", "https"]
        types = ["property-to-rent", "property-for-sale", "new-homes-for-sale"]
        urls = [real_url.format(p, t) for p in protocols for t in types]
        conditions = [self.url.startswith(u) for u in urls]
        conditions.append(self._status_code == 200)
        if not any(conditions):
            raise ValueError(f"Invalid rightmove search URL:\n\n\t{self.url}")

    @property
    def url(self):
        return self._url

    @property
    def get_results(self):
        """Pandas DataFrame of all results returned by the search."""
        return self._results

    @property
    def results_count(self):
        """Total number of results returned by `get_results`. Note that the
        rightmove website may state a much higher number of results; this is
        because they artificially restrict the number of results pages that can
        be accessed to 42."""
        return len(self.get_results)

    @property
    def average_price(self):
        """Average price of all results returned by `get_results` (ignoring
        results which don't list a price)."""
        total = self.get_results["price"].dropna().sum()
        return total / self.results_count if self.results_count else np.nan

    def summary(self, by: str = None):
        """DataFrame summarising results by mean price and count. Defaults to
        grouping by `number_bedrooms` (residential) or `type` (commercial), but
        accepts any column name from `get_results` as a grouper.

        Args:
            by (str): valid column name from `get_results` DataFrame attribute.
        """
        if not by:
            by = "type" if "commercial" in self.rent_or_sale else "number_bedrooms"
        assert (
            by in self.get_results.columns
        ), f"Column not found in `get_results`: {by}"
        df = self.get_results.dropna(axis=0, subset=["price"])
        groupers = {"price": ["count", "mean"]}
        df = df.groupby(df[by]).agg(groupers)
        df.columns = df.columns.get_level_values(1)
        df.reset_index(inplace=True)
        if "number_bedrooms" in df.columns:
            df["number_bedrooms"] = df["number_bedrooms"].astype(int)
            df.sort_values(by=["number_bedrooms"], inplace=True)
        else:
            df.sort_values(by=["count"], inplace=True, ascending=False)
        return df.reset_index(drop=True)

    @property
    def rent_or_sale(self):
        """String specifying if the search is for properties for rent or sale.
        Required because Xpaths are different for the target elements."""
        if "/property-for-sale/" in self.url or "/new-homes-for-sale/" in self.url:
            return "sale"
        elif "/property-to-rent/" in self.url:
            return "rent"
        elif "/commercial-property-for-sale/" in self.url:
            return "sale-commercial"
        elif "/commercial-property-to-let/" in self.url:
            return "rent-commercial"
        else:
            raise ValueError(f"Invalid rightmove URL:\n\n\t{self.url}")

    @property
    def results_count_display(self):
        """Returns an integer of the total number of listings as displayed on
        the first page of results. Note that not all listings are available to
        scrape because rightmove limits the number of accessible pages."""
        raw_count = self._first_search_results.get("resultCount", 0)
        if isinstance(raw_count, str):
            raw_count = raw_count.replace(",", "")
        try:
            return int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("Unexpected result count format.") from exc

    @property
    def page_count(self):
        """Returns the number of result pages returned by the search URL. There
        are 24 results per page. Note that the website limits results to a
        maximum of 42 accessible pages."""
        pagination = self._first_search_results.get("pagination") or {}
        total = pagination.get("total")
        if total is None:
            total = self.results_count_display // 24
            if self.results_count_display % 24 > 0:
                total += 1
        return min(int(total), 42)

    def _extract_search_results(self, request_content):
        tree = html.fromstring(request_content)
        script = tree.xpath("//script[@id='__NEXT_DATA__']/text()")
        if not script:
            raise ValueError("Unable to locate embedded search results data.")
        try:
            data = json.loads(script[0])
            return data["props"]["pageProps"]["searchResults"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError("Unexpected rightmove response format.") from exc

    def _page_url_with_index(self, index: int) -> str:
        split_url = urlsplit(self.url)
        query_pairs = [
            (key, value)
            for key, value in parse_qsl(split_url.query, keep_blank_values=True)
            if key != "index"
        ]
        if index > 0:
            query_pairs.append(("index", str(index)))
        new_query = urlencode(query_pairs)
        return urlunsplit(
            (
                split_url.scheme,
                split_url.netloc,
                split_url.path,
                new_query,
                split_url.fragment,
            )
        )

    @staticmethod
    def _extract_json_blob(script_text: str) -> str | None:
        if not script_text:
            return None
        start = script_text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(script_text)):
            char = script_text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return script_text[start : idx + 1]
        return None

    @staticmethod
    def _clean_description(raw_description: str | None):
        if not raw_description:
            return np.nan
        try:
            fragment = html.fromstring(f"<div>{raw_description}</div>")
            text = fragment.text_content()
        except (ValueError, TypeError):
            return raw_description
        cleaned = " ".join(text.split())
        return cleaned if cleaned else np.nan

    @staticmethod
    def _format_key_features(features: list | None):
        if not features:
            return np.nan
        filtered = [f for f in features if f]
        if not filtered:
            return np.nan
        return "; ".join(filtered)

    @staticmethod
    def _format_council_tax(living_costs: Dict[str, Any] | None):
        if not living_costs:
            return np.nan
        band = living_costs.get("councilTaxBand")
        included = living_costs.get("councilTaxIncluded")
        exempt = living_costs.get("councilTaxExempt")
        if not band:
            return np.nan
        parts = [f"Band {band}"]
        if included is True:
            parts.append("included in rent")
        elif included is False:
            parts.append("not included")
        if exempt:
            parts.append("exempt")
        return " - ".join(parts)

    @staticmethod
    def _format_utilities(features: Dict[str, Any] | None):
        if not isinstance(features, dict):
            return np.nan
        prepared: Dict[str, Any] = {}
        for key, value in features.items():
            if isinstance(value, list):
                items = []
                for entry in value:
                    if isinstance(entry, dict):
                        text = entry.get("displayText") or entry.get("alias")
                        if text:
                            items.append(text)
                    elif entry:
                        items.append(str(entry))
                if items:
                    prepared[key] = items
            elif isinstance(value, dict):
                nested = {k: v for k, v in value.items() if v not in (None, "", [])}
                if nested:
                    prepared[key] = nested
            elif value not in (None, "", []):
                prepared[key] = value
        if not prepared:
            return np.nan
        return json.dumps(prepared, ensure_ascii=False)

    @staticmethod
    def _normalise_detail_columns(results: pd.DataFrame):
        if "minimum_tenancy_months" in results.columns:
            results["minimum_tenancy_months"] = pd.to_numeric(
                results["minimum_tenancy_months"], errors="coerce"
            )

        if "deposit" in results.columns:
            cleaned = results["deposit"].astype(str)
            cleaned = cleaned.str.replace(r"[^\d.]", "", regex=True)
            cleaned = cleaned.where(cleaned != "", np.nan)
            results["deposit"] = pd.to_numeric(cleaned, errors="coerce")

        if "latitude" in results.columns:
            results["latitude"] = pd.to_numeric(results["latitude"], errors="coerce")
        if "longitude" in results.columns:
            results["longitude"] = pd.to_numeric(results["longitude"], errors="coerce")

    def _fetch_property_details(self, property_url: str | None, include_floorplans: bool):
        if not property_url:
            return {}
        cached = self._property_details_cache.get(property_url)
        if cached is None:
            cached = self._extract_property_details(property_url)
            self._property_details_cache[property_url] = cached
        details = cached.copy()
        if not include_floorplans:
            details.pop("floorplan_url", None)
        return details

    def _extract_property_details(self, property_url: str) -> Dict[str, Any]:
        details: Dict[str, Any] = {}
        try:
            status_code, content = self._request(
                property_url, timeout=PROPERTY_DETAILS_TIMEOUT
            )
        except requests.RequestException:
            return details
        if status_code != 200:
            return details
        try:
            tree = html.fromstring(content)
        except (TypeError, ValueError):
            return details
        scripts = tree.xpath("//script[contains(text(),'window.PAGE_MODEL')]/text()")
        if not scripts:
            return details
        json_text = self._extract_json_blob(scripts[0])
        if not json_text:
            return details
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            return details
        property_data = payload.get("propertyData") or {}
        lettings = property_data.get("lettings") or {}
        living_costs = property_data.get("livingCosts") or {}
        location = property_data.get("location") or {}
        features = property_data.get("features") or {}
        key_features = property_data.get("keyFeatures") or []
        floorplans = property_data.get("floorplans") or []
        description = property_data.get("text", {}).get("description")

        floorplan_url = np.nan
        if floorplans:
            first_floorplan = floorplans[0]
            if isinstance(first_floorplan, dict):
                fp_url = first_floorplan.get("url")
                if fp_url:
                    floorplan_url = urljoin(BASE_URL, fp_url)

        def optional(value: Any):
            if value in (None, ""):
                return np.nan
            return value

        details.update(
            {
                "let_type": optional(lettings.get("letType")),
                "furnish_type": optional(lettings.get("furnishType")),
                "council_tax": self._format_council_tax(living_costs),
                "minimum_tenancy_months": optional(
                    lettings.get("minimumTermInMonths")
                ),
                "deposit": optional(lettings.get("deposit")),
                "description": self._clean_description(description),
                "property_type": optional(
                    property_data.get("propertySubType")
                    or property_data.get("propertyType")
                ),
                "key_features": self._format_key_features(key_features),
                "utilities_rights_restrictions": self._format_utilities(features),
                "location": json.dumps(location, ensure_ascii=False)
                if location
                else np.nan,
                "latitude": optional(location.get("latitude")),
                "longitude": optional(location.get("longitude")),
                "floorplan_url": floorplan_url,
            }
        )
        return details

    def _results_from_search(self, search_results, get_floorplans: bool = False):
        properties = search_results.get("properties") or []
        rows: list[Dict[str, Any]] = []
        for prop in properties:
            price_data = prop.get("price") or {}
            price = price_data.get("amount")
            property_type = (
                prop.get("heading")
                or prop.get("propertyTypeFullDescription")
                or prop.get("summary")
                or prop.get("propertySubType")
            )
            address = prop.get("displayAddress")
            property_url = prop.get("propertyUrl")
            agent_url = prop.get("contactUrl")
            bedrooms = prop.get("bedrooms")
            let_available_date = prop.get("letAvailableDate")
            full_property_url = (
                urljoin(BASE_URL, property_url) if property_url else None
            )
            row: Dict[str, Any] = {
                "price": price if price is not None else np.nan,
                "type": property_type,
                "address": address,
                "url": full_property_url if full_property_url else np.nan,
                "agent_url": urljoin(BASE_URL, agent_url)
                if agent_url
                else np.nan,
                "number_bedrooms": bedrooms if bedrooms is not None else np.nan,
                "let_available_date": let_available_date,
            }
            detail_defaults = {
                "let_type": np.nan,
                "furnish_type": np.nan,
                "council_tax": np.nan,
                "minimum_tenancy_months": np.nan,
                "deposit": np.nan,
                "description": np.nan,
                "property_type": np.nan,
                "key_features": np.nan,
                "utilities_rights_restrictions": np.nan,
                "location": np.nan,
                "latitude": np.nan,
                "longitude": np.nan,
                "floorplan_url": np.nan,
            }
            row.update(detail_defaults)
            should_fetch_details = (
                (self._include_property_details and "rent" in self.rent_or_sale)
                or get_floorplans
            )
            if should_fetch_details and full_property_url:
                details = self._fetch_property_details(
                    full_property_url, include_floorplans=get_floorplans
                )
                row.update(details)
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df[df["address"].notnull()]
        return df

    def _get_results(self, get_floorplans: bool = False):
        frames = [
            self._results_from_search(
                self._first_search_results, get_floorplans=get_floorplans
            )
        ]
        for offset in range(24, self.page_count * 24, 24):
            page_url = self._page_url_with_index(offset)
            status_code, content = self._request(page_url)
            if status_code != 200:
                break
            search_results = self._extract_search_results(content)
            frames.append(
                self._results_from_search(
                    search_results, get_floorplans=get_floorplans
                )
            )
        if len(frames) == 1:
            results = frames[0]
        else:
            results = pd.concat(frames, ignore_index=True)
        return self._clean_results(results)

    def enrich_property_details(
        self, df: pd.DataFrame, include_floorplans: bool = False
    ) -> pd.DataFrame:
        """Fetch property details for the provided DataFrame rows.

        Args:
            df (pd.DataFrame): subset of rows to enrich. Must include a `url`
                column containing full property URLs.
            include_floorplans (bool): when True, attempt to extract floorplan
                URLs while fetching property details (slower).

        Returns:
            pd.DataFrame: copy of the input DataFrame with additional columns
            populated where available.
        """

        if df.empty or "url" not in df.columns:
            return df.copy()

        detail_columns = [
            "let_type",
            "furnish_type",
            "council_tax",
            "minimum_tenancy_months",
            "deposit",
            "description",
            "property_type",
            "key_features",
            "utilities_rights_restrictions",
            "location",
            "latitude",
            "longitude",
            "floorplan_url",
        ]

        result = df.copy()
        for column in detail_columns:
            if column not in result.columns:
                result[column] = np.nan
            result[column] = result[column].astype(object)

        urls = [
            url
            for url in result["url"].dropna().unique()
            if isinstance(url, str) and url
        ]

        for url in urls:
            details = self._fetch_property_details(url, include_floorplans)
            if not details:
                continue
            mask = result["url"] == url
            for key, value in details.items():
                if key in detail_columns:
                    result.loc[mask, key] = value

        RightmoveData._normalise_detail_columns(result)
        return result

    @staticmethod
    def _clean_results(results: pd.DataFrame):
        results.reset_index(inplace=True, drop=True)
        if "price" in results.columns:
            results["price"] = pd.to_numeric(results["price"], errors="coerce")
        else:
            results["price"] = np.nan

        if "type" in results.columns:
            results["type"] = results["type"].astype(object)
            results["type"] = results["type"].str.strip("\n").str.strip()

        if "number_bedrooms" in results.columns:
            results["number_bedrooms"] = pd.to_numeric(
                results["number_bedrooms"], errors="coerce"
            )
        else:
            results["number_bedrooms"] = np.nan

        if "let_available_date" in results.columns:
            results["let_available_date"] = pd.to_datetime(
                results["let_available_date"], errors="coerce", utc=True
            )
            results["let_available_date"] = results["let_available_date"].dt.tz_convert(
                None
            )

        RightmoveData._normalise_detail_columns(results)

        studio_mask = results["type"].str.contains(
            "studio", case=False, na=False
        )
        results.loc[studio_mask, "number_bedrooms"] = 0

        missing_beds = results["number_bedrooms"].isna()
        if missing_beds.any():
            bed_pattern = r"\b([\d][\d]?)\b"
            extracted = (
                results.loc[missing_beds, "type"]
                .astype(str)
                .str.extract(bed_pattern, expand=True)[0]
            )
            results.loc[missing_beds, "number_bedrooms"] = pd.to_numeric(
                extracted, errors="coerce"
            )

        postcode_pattern = r"\b([A-Za-z][A-Za-z]?[0-9][0-9]?[A-Za-z]?)\b"
        full_postcode_pattern = (
            r"([A-Za-z][A-Za-z]?[0-9][0-9]?[A-Za-z]?[0-9]?\s[0-9]?[A-Za-z][A-Za-z])"
        )
        address_str = results["address"].astype(str)
        results["postcode"] = address_str.str.extract(
            postcode_pattern, expand=True
        )[0]
        results["full_postcode"] = address_str.str.extract(
            full_postcode_pattern, expand=True
        )[0]

        now = datetime.datetime.now()
        results["search_date"] = now

        return results
