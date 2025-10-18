"""Example script for fetching and filtering Rightmove listings."""
from datetime import datetime
from pathlib import Path

from rightmove_webscraper import RightmoveData

# Paste the Rightmove search results URL here
url = (
    "https://www.rightmove.co.uk/property-to-rent/find.html?searchLocation=London&useLocationIdentifier=true&locationIdentifier=REGION%5E87490&rent=To+rent&radius=5.0&minPrice=900&maxPrice=2000&minBedrooms=1&maxBedrooms=2&dontShow=houseShare%2Cretirement%2Cstudent&sortType=6&channel=RENT&transactionType=LETTING&displayLocationIdentifier=London-87490.html&furnishTypes=furnished&index=0&includeLetAgreed=true"
)

# Desired availability date (DD/MM/YYYY)
target_date_str = "01/12/2025"
target_date = datetime.strptime(target_date_str, "%d/%m/%Y")

# Create an instance of the RightmoveData class
# This will start the scraping process
rm = RightmoveData(url)

print(f"Scraping data from: {url}")
print(f"Found {rm.results_count} results.")

results = rm.get_results

if "let_available_date" not in results.columns:
    raise ValueError(
        "The current search results do not include 'let_available_date' data."
    )

available = results[
    results["let_available_date"].notna()
    & (results["let_available_date"] >= target_date)
].copy()
available.sort_values("let_available_date", inplace=True)

output_filename = (
    f"rightmove_listings_{target_date.strftime('%Y%m%d')}.csv"
)
output_path = Path(output_filename)
available.to_csv(output_path, index=False)

print(
    f"\nWrote {len(available)} properties available from {target_date_str} "
    f"onwards to {output_path.resolve()}"
)
