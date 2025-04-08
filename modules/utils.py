# utils.py
import requests
import pprint
from typing import List, Dict, Optional
import recipe

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
        print("API connection successful.")
    except Exception as e:
        print(f"API Test failed: {e}")

def update_receipe(self, urls: Dict = None):
    if urls.keys() not in receipe.keys():
        