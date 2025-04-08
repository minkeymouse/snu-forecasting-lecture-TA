import requests
import pandas as pd
import itertools
from tqdm import tqdm
import time
from io import StringIO
from datetime import datetime
from typing import List, Dict, Optional
from config import CONFIG  # Make sure config.py is in your PYTHONPATH

class OECDAPI:
    def __init__(self, 
                 config: Dict = CONFIG, 
                 countries: List[str] = None, 
                 start: str = None, 
                 end: str = None, 
                 freq: str = "Q", 
                 base_url: str = "https://sdmx.oecd.org/public/rest/data/"):
        """
        Initialize with configuration for each indicator, the list of countries, and the time period as strings.
        For example, start="1990-Q1" and end="2024-Q4".
        freq: Frequency of data (e.g., "Q" for quarterly, "Y" for yearly).
        """
        self.start = start
        self.end = end
        self.config = config
        self.freq = freq
        self.base_url = base_url
        # Use the user-provided list of countries or default to an empty list.
        self.countries = countries if countries is not None else []

        if self.start is None or self.end is None:
            raise ValueError("Both start and end periods must be provided.")

    def test_api(self) -> None:
        """
        Performs a simple test of the OECD API by requesting a known query.
        """
        test_url = (
            "https://sdmx.oecd.org/public/rest/data/"
            "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_CAPITA,1.1/"
            "Q............?startPeriod=2024-Q1"
        )
        try:
            resp = requests.get(test_url)
            resp.raise_for_status()
            print("API Test successful.")
        except Exception as e:
            print(f"API Test failed: {e}")

    def fetch_data(self, chunk_size: int = 20, path: str = "./datasets/OECD/") -> 'OECDAPI':
        """
        Downloads raw data from the OECD API for each indicator defined in the configuration.
        Data are downloaded in chunks over the time period from self.start to self.end.
        The chunk_size is defined as the number of periods (e.g., quarters if freq="Q").
        For each indicator, the filter string is built from the configuration values in the following order:
          TRANSACTION . SECTOR . INSTR_ASSET . ACTIVITY . TABLE_IDENTIFIER . COUNTERPART_SECTOR . EXPENDITURE
        The downloaded raw data for each indicator is saved as a CSV file in the specified path.
        """
        # Create a period range based on start, end, and the provided frequency.
        period_range = pd.period_range(start=self.start, end=self.end, freq=self.freq)
        
        # Process each indicator in the configuration.
        for col, conf in self.config.items():
            all_chunks = []
            time_chunks = []
            # Build the time chunks list.
            for i in range(0, len(period_range), chunk_size):
                chunk = period_range[i : i + chunk_size]
                if self.freq == "Q":
                    chunk_start = f"{chunk[0].year}-Q{chunk[0].quarter}"
                    chunk_end = f"{chunk[-1].year}-Q{chunk[-1].quarter}"
                elif self.freq == "Y":
                    # For yearly, simply use the year as a string.
                    chunk_start = str(chunk[0])
                    chunk_end = str(chunk[-1])
                elif self.freq == "M":
                    # For monthly data, you might want a format like "YYYY-MM"
                    chunk_start = chunk[0].strftime("%Y-%m")
                    chunk_end = chunk[-1].strftime("%Y-%m")
                else:
                    raise ValueError(f"Unsupported frequency: {self.freq}")
                time_chunks.append((chunk_start, chunk_end))
            print(f"\nFor indicator '{col}', time chunks to process: {time_chunks}")

            # Build the filter string using the configuration keys.
            # Order: TRANSACTION . SECTOR . INSTR_ASSET . ACTIVITY . TABLE_IDENTIFIER . COUNTERPART_SECTOR . EXPENDITURE
            filter_parts = [
                conf.get("TRANSACTION", "."),
                conf.get("SECTOR", "."),
                conf.get("INSTR_ASSET", "."),
                conf.get("ACTIVITY", "."),
                conf.get("TABLE_IDENTIFIER", "."),
                conf.get("COUNTERPART_SECTOR", "."),
                conf.get("EXPENDITURE", ".")
            ]
            base_filter = ".".join(filter_parts)
            # Construct the full URL using the agency, dataset, version, and filter string.
            full_url = (
                self.base_url +
                conf.get("AGENCY", "") + "," +
                conf.get("DATASET", "") + "," +
                conf.get("VERSION", "") + "/" +
                base_filter
            )
            print(f"Fetching data for '{col}' using URL: {full_url}")

            # Download the data in chunks.
            for chunk_start, chunk_end in tqdm(time_chunks, desc=f"Downloading {col} Data"):
                query_url = (
                    f"{full_url}?startPeriod={chunk_start}&endPeriod={chunk_end}"
                    "&dimensionAtObservation=AllDimensions&format=csv"
                )
                try:
                    resp = requests.get(query_url)
                    resp.raise_for_status()
                    chunk_df = pd.read_csv(StringIO(resp.text))
                    all_chunks.append(chunk_df)
                    print(f"  Chunk {chunk_start} to {chunk_end}: {chunk_df.shape[0]} rows")
                except Exception as e:
                    print(f"Error for chunk {chunk_start} to {chunk_end} for indicator '{col}': {e}")
                time.sleep(5)  # Pause between requests to help avoid rate limiting.
            # Concatenate all chunks for this indicator.
            if all_chunks:
                indicator_df = pd.concat(all_chunks, ignore_index=True)
            else:
                indicator_df = pd.DataFrame(columns=["TIME_PERIOD", "REF_AREA", conf.get("TRANSACTION", ".")])
            csv_filename = f"{path}{col}.csv"
            indicator_df.to_csv(csv_filename, index=False)
            print(f"Data for indicator '{col}' saved to {csv_filename}")
        return self

    def create_dataframe(self, cols: List[str] = None, path: str = "./datasets/OECD/") -> pd.DataFrame:
        """
        Merges the individual indicator CSV files (created by fetch_data) into a combined DataFrame.
        The final DataFrame will contain:
          - date: Time period (formatted as "YYYY-Qn")
          - country: REF_AREA
          - one column per indicator (using the configuration key names).
        A full grid for all specified countries and all periods in the date range is created to ensure missing
        observations are NaN.
        The final DataFrame is saved as "Queried_data.csv" in the specified path.
        """
        if cols is None:
            cols = list(self.config.keys())
        merged_df = None
        for indicator in tqdm(cols, desc="Merging Indicators"):
            csv_file = f"{path}{indicator}.csv"
            df = pd.read_csv(csv_file)
            conf = self.config[indicator]
            # Determine the column name expected to contain the observation value.
            # We use the TRANSACTION code as a fallback.
            value_col = conf.get("TRANSACTION", ".")
            if value_col not in df.columns:
                # If not found, default to the third column.
                value_col = df.columns[2]
            df.rename(columns={"TIME_PERIOD": "date", "REF_AREA": "country", value_col: indicator}, inplace=True)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=["country", "date"], how="outer")
        
        # Create a full grid for the specified countries and all periods in the date range.
        # Use the provided self.countries list if available.
        if not self.countries:
            raise ValueError("No countries provided. Please supply a list of countries.")
        period_range = pd.period_range(start=self.start, end=self.end, freq=self.freq)
        # Adjust formatting based on frequency; here we assume quarterly data.
        if self.freq == "Q":
            periods_str = [f"{p.year}-Q{p.quarter}" for p in period_range]
        else:
            periods_str = [str(p) for p in period_range]
        full_grid = pd.DataFrame(list(itertools.product(self.countries, periods_str)), columns=["country", "date"])
        final_df = pd.merge(full_grid, merged_df, on=["country", "date"], how="left")
        final_df.sort_values(by=["country", "date"], inplace=True)
        output_csv = f"{path}Queried_data.csv"
        final_df.to_csv(output_csv, index=False)
        print(f"\nFinal Combined DataFrame saved to {output_csv}")
        return final_df
