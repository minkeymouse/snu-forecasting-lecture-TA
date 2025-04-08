# oecddatabuilder.py
import os
import requests
import pandas as pd
from tqdm import tqdm
import time
from io import StringIO
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

class OECDAPI_Databuilder:
    def __init__(self, 
                 config: Dict = None, 
                 start: str = None, 
                 end: str = None, 
                 freq: str = "Q",
                 response_format: str = "csv",  # "csv", "json", or "xml",
                 dbpath: str="../datasets/OECD/",
                 base_url: str = None):
        """
        Initialize the OECDAPI with a configuration dictionary and a time period.
        For example, start="2019-Q1" and end="2020-Q3".
        
        freq: Frequency of data (e.g., "Q", "Y", "M").
        response_format: The desired response format; options include "csv", "json" and "xml".
        base_url: The API’s base URL.
        """
        self.start = start
        self.end = end
        self.config = config
        self.freq = freq.upper()
        self.response_format = response_format.lower()
        self.base_url = base_url
        self.indicators = list(self.config.keys())  # fixed: use list() on config keys
        self.dbpath = dbpath

        if not os.path.exists(self.dbpath):
            os.makedirs(self.dbpath)

        if self.start is None or self.end is None:
            raise ValueError("Both start and end periods must be provided.")

        # Collect the union of all REF_AREA values from the configuration.
        countries_set = set()
        for conf in config.values():
            ref_area = conf.get("REF_AREA", "")
            if ref_area:
                countries_set.update(ref_area.split("+"))
        self.countries = sorted(list(countries_set))
        print(f"Combined countries from configuration: {self.countries}")

    def fetch_data(self, chunk_size: int = 100):
        """
        Fetch data from the OECD API in chunks (to avoid very large queries or hitting rate limits)
        and then save the concatenated results as CSV files for each indicator.
        """
        period_range = pd.period_range(start=self.start, end=self.end, freq=self.freq)
        
        for indicator, conf in self.config.items():
            filter_order = list(conf.keys())
            all_chunks = []
            time_chunks = []
            # Build the list of time chunks.
            for i in range(0, len(period_range), chunk_size):
                chunk = period_range[i: i + chunk_size]
                if self.freq == "Q":
                    chunk_start = f"{chunk[0].year}-Q{chunk[0].quarter}"
                    chunk_end = f"{chunk[-1].year}-Q{chunk[-1].quarter}"
                elif self.freq == "Y":
                    chunk_start = str(chunk[0])
                    chunk_end = str(chunk[-1])
                elif self.freq == "M":
                    chunk_start = chunk[0].strftime("%Y-%m")
                    chunk_end = chunk[-1].strftime("%Y-%m")
                else:
                    raise ValueError(f"Unsupported frequency: {self.freq}")
                time_chunks.append((chunk_start, chunk_end))
            print(f"\nFor indicator '{indicator}', time chunks to process: {time_chunks}")

            filter_values = [conf.get(key, "") for key in filter_order]
            filter_url = ".".join(filter_values)

            # Construct the full URL for this indicator.
            full_url = (
                self.base_url +
                filter_url
            )
            print(f"Fetching data for '{indicator}' using URL: {full_url}")

            # Determine Accept header based on desired response format.
            headers = {}
            if self.response_format == "csv":
                headers["Accept"] = "application/vnd.sdmx.data+csv; charset=utf-8"
            elif self.response_format == "json":
                headers["Accept"] = "application/vnd.sdmx.data+json; charset=utf-8; version=2"
            elif self.response_format == "xml":
                headers["Accept"] = "application/vnd.sdmx.genericdata+xml; charset=utf-8; version=2.1"
            else:
                raise ValueError("response_format must be one of: csv, json, xml")

            # Process each time chunk.
            for chunk_start, chunk_end in tqdm(time_chunks, desc=f"Downloading {indicator} Data"):
                query_url = (
                    f"{full_url}?startPeriod={chunk_start}&endPeriod={chunk_end}"
                    "&dimensionAtObservation=TIME_PERIOD"
                )
                try:
                    resp = requests.get(query_url, headers=headers)
                    resp.raise_for_status()
                    
                    # Process the API response.
                    if self.response_format == "csv":
                        chunk_df = pd.read_csv(StringIO(resp.text))
                    elif self.response_format == "json":
                        chunk_df = pd.read_json(StringIO(resp.text))
                    elif self.response_format == "xml":
                        chunk_df = pd.read_xml(StringIO(resp.text))
                    else:
                        raise ValueError("Unsupported response_format")
                    
                    # Select only the needed columns.
                    # Note: Depending on the API, the actual column names might vary.
                    if {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}.issubset(chunk_df.columns):
                        chunk_df = chunk_df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]]
                    else:
                        print(f"Unexpected columns in the response: {chunk_df.columns.tolist()}")
                    
                    all_chunks.append(chunk_df)
                    print(f"  Chunk {chunk_start} to {chunk_end}: {chunk_df.shape[0]} rows")
                except Exception as e:
                    print(f"Error for chunk {chunk_start} to {chunk_end} for indicator '{indicator}': {e}")
                time.sleep(5)  # Pause between requests to avoid rate limiting.
            
            # Concatenate data for this indicator.
            if all_chunks:
                indicator_df = pd.concat(all_chunks, ignore_index=True)
            else:
                # Create an empty DataFrame if no data was fetched.
                indicator_df = pd.DataFrame(columns=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"])
            
            # Save the fetched data as CSV.
            csv_filename = os.path.join(self.dbpath, f"{indicator}.csv")
            indicator_df.to_csv(csv_filename, index=False)
            print(f"Data for indicator '{indicator}' saved to {csv_filename}")
        return self
    

    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a merged DataFrame that combines the data for all indicators.
        Each CSV file is expected to contain 'REF_AREA', 'TIME_PERIOD', and 'OBS_VALUE' columns.
        We rename 'TIME_PERIOD' to 'date' and 'REF_AREA' to 'country', and then merge using these keys.
        """
        if not self.indicators:
            raise ValueError("No indicators found. Please check your configuration.")
        
        merged_df = None  # Start with an empty structure
        for indicator in self.indicators:
            csv_file = os.path.join(self.dbpath, f"{indicator}.csv")
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"No file fetched for {csv_file}: {e}!")
                continue
            
            # Rename the standard columns to universal names and the observation column to the indicator name.
            df = df.rename(columns={
                "TIME_PERIOD": "date", 
                "REF_AREA": "country", 
                "OBS_VALUE": indicator
            })
            # If the merged_df is empty, initialize it.
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=["date", "country"], how="outer")
        
        if merged_df is None:
            raise ValueError("No data was loaded from any CSV file.")
        
        # Optional cleaning: sort by date and country, and reset the index.
        merged_df = merged_df.sort_values(by=["date", "country"]).reset_index(drop=True)
        # Convert 'date' to datetime or period if needed:
        # For example, if dates are in the format 'YYYY-Qn', you might parse them using custom logic.
        
        # Additional cleaning steps can be added here.
        
        final_df = merged_df
        return final_df
