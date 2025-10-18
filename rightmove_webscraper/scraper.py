import datetime
import json
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from lxml import html
import numpy as np
import pandas as pd
import requests


BASE_URL = "https://www.rightmove.co.uk"
REQUEST_TIMEOUT = 20
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

    def __init__(self, url: str, get_floorplans: bool = False):
        """Initialize the scraper with a URL from the results of a property
        search performed on www.rightmove.co.uk.

        Args:
            url (str): full HTML link to a page of rightmove search results.
            get_floorplans (bool): optionally scrape links to the individual
                floor plan images for each listing (be warned this drastically
                increases runtime so is False by default).
        """
        self._status_code, self._first_page = self._request(url)
        self._url = url
        self._validate_url()
        self._first_search_results = self._extract_search_results(self._first_page)
        self._results = self._get_results(get_floorplans=get_floorplans)

    @staticmethod
    def _request(url: str):
        response = requests.get(
            url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
        )
        return response.status_code, response.content

    def refresh_data(self, url: str = None, get_floorplans: bool = False):
        """Make a fresh GET request for the rightmove data.

        Args:
            url (str): optionally pass a new HTML link to a page of rightmove
                search results (else defaults to the current `url` attribute).
            get_floorplans (bool): optionally scrape links to the individual
                flooplan images for each listing (this drastically increases
                runtime so is False by default).
        """
        url = self.url if not url else url
        self._status_code, self._first_page = self._request(url)
        self._url = url
        self._validate_url()
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

    def _results_from_search(self, search_results, get_floorplans: bool = False):
        properties = search_results.get("properties") or []
        columns = [
            "price",
            "type",
            "address",
            "url",
            "agent_url",
            "number_bedrooms",
            "let_available_date",
        ]
        rows = []
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
            rows.append(
                [
                    price if price is not None else np.nan,
                    property_type,
                    address,
                    urljoin(BASE_URL, property_url) if property_url else np.nan,
                    urljoin(BASE_URL, agent_url) if agent_url else np.nan,
                    bedrooms if bedrooms is not None else np.nan,
                    let_available_date,
                ]
            )
        df = pd.DataFrame(rows, columns=columns)
        df = df[df["address"].notnull()]
        if get_floorplans and not df.empty:
            df["floorplan_url"] = df["url"].apply(self._fetch_floorplan_url)
        return df

    def _fetch_floorplan_url(self, property_url):
        if not isinstance(property_url, str) or not property_url:
            return np.nan
        status_code, content = self._request(property_url)
        if status_code != 200:
            return np.nan
        tree = html.fromstring(content)
        xp_floorplan = "//img[contains(@src, 'floorplan')]/@src"
        floorplan_urls = tree.xpath(xp_floorplan)
        if not floorplan_urls:
            return np.nan
        floorplan_url = floorplan_urls[0]
        if floorplan_url.startswith("//"):
            floorplan_url = f"https:{floorplan_url}"
        elif floorplan_url.startswith("/"):
            floorplan_url = urljoin(BASE_URL, floorplan_url)
        return floorplan_url

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
