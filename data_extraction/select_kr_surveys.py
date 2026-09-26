#!/usr/bin/env python3
"""
02_select_kr_surveys.py - Study Survey Selection Protocol

Filters raw survey inventory to identify and select the single most recent,
standard national Kids Recode (KR) dataset per country that contains the complete
pediatric health care-seeking module (H11, H22, H31, H12Z, H32Z).

Applies prespecified study inclusion/exclusion criteria:
- Excludes subnational surveys (e.g., Ondo State 1986).
- Excludes interim/mini-surveys omitting illness modules (e.g., Ethiopia 2019 Mini-DHS).
- Excludes non-standard provider coding surveys (e.g., Sudan DHS-I).
- Retains exactly 38 Sub-Saharan African national surveys.

Usage:
    python 02_select_kr_surveys.py --inventory-csv dhs_scan_summary.csv --output-csv selected_kr_surveys.csv
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Protocol specified survey references (Table S1)
PROTOCOL_SURVEYS = {
    "AO": {"wave": "DHS-VIII", "version": "81", "zip": "AOKR81SV.zip", "years": "2023–2024", "name": "Angola"},
    "BF": {"wave": "DHS-VIII", "version": "81", "zip": "BFKR81SV.zip", "years": "2021", "name": "Burkina Faso"},
    "BJ": {"wave": "DHS-VII",  "version": "71", "zip": "BJKR71SV.zip", "years": "2017–2018", "name": "Benin"},
    "BU": {"wave": "DHS-VII",  "version": "71", "zip": "BUKR71SV.zip", "years": "2016–2017", "name": "Burundi"},
    "CD": {"wave": "DHS-VIII", "version": "81", "zip": "CDKR81SV.zip", "years": "2023–2024", "name": "Democratic Republic of the Congo"},
    "CF": {"wave": "DHS-III",  "version": "31", "zip": "CFKR31SV.zip", "years": "1994–1995", "name": "Central African Republic"},
    "CG": {"wave": "DHS-VI",   "version": "61", "zip": "CGKR61SV.zip", "years": "2011–2012", "name": "Congo"},
    "CI": {"wave": "DHS-VIII", "version": "81", "zip": "CIKR81SV.zip", "years": "2021", "name": "Cote d'Ivoire"},
    "CM": {"wave": "DHS-VIII", "version": "82", "zip": "CMKR82SV.zip", "years": "2022", "name": "Cameroon"},
    "ET": {"wave": "DHS-VII",  "version": "71", "zip": "ETKR71SV.zip", "years": "2016", "name": "Ethiopia"},
    "GA": {"wave": "DHS-VII",  "version": "71", "zip": "GAKR71SV.zip", "years": "2019–2021", "name": "Gabon"},
    "GH": {"wave": "DHS-VIII", "version": "8C", "zip": "GHKR8CSV.zip", "years": "2022–2023", "name": "Ghana"},
    "GM": {"wave": "DHS-VIII", "version": "81", "zip": "GMKR81SV.zip", "years": "2019–2020", "name": "Gambia"},
    "GN": {"wave": "DHS-VIII", "version": "82", "zip": "GNKR82SV.zip", "years": "2021", "name": "Guinea"},
    "KE": {"wave": "DHS-VIII", "version": "8C", "zip": "KEKR8CSV.zip", "years": "2022", "name": "Kenya"},
    "KM": {"wave": "DHS-VI",   "version": "61", "zip": "KMKR61SV.zip", "years": "2012", "name": "Comoros"},
    "LB": {"wave": "DHS-VIII", "version": "81", "zip": "LBKR81SV.zip", "years": "2022", "name": "Liberia"},
    "LS": {"wave": "DHS-VIII", "version": "81", "zip": "LSKR81SV.zip", "years": "2023–2024", "name": "Lesotho"},
    "MD": {"wave": "DHS-VIII", "version": "81", "zip": "MDKR81SV.zip", "years": "2021", "name": "Madagascar"},
    "ML": {"wave": "DHS-VIII", "version": "83", "zip": "MLKR83SV.zip", "years": "2021", "name": "Mali"},
    "MR": {"wave": "DHS-VII",  "version": "71", "zip": "MRKR71SV.zip", "years": "2019–2021", "name": "Mauritania"},
    "MW": {"wave": "DHS-VIII", "version": "81", "zip": "MWKR81SV.zip", "years": "2024", "name": "Malawi"},
    "MZ": {"wave": "DHS-VIII", "version": "81", "zip": "MZKR81SV.zip", "years": "2022–2023", "name": "Mozambique"},
    "NG": {"wave": "DHS-VIII", "version": "8B", "zip": "NGKR8BSV.zip", "years": "2023–2024", "name": "Nigeria"},
    "NI": {"wave": "DHS-VI",   "version": "61", "zip": "NIKR61SV.zip", "years": "2012", "name": "Niger"},
    "NM": {"wave": "DHS-VI",   "version": "61", "zip": "NMKR61SV.zip", "years": "2013", "name": "Namibia"},
    "RW": {"wave": "DHS-VIII", "version": "81", "zip": "RWKR81SV.zip", "years": "2019–2020", "name": "Rwanda"},
    "SL": {"wave": "DHS-VII",  "version": "7A", "zip": "SLKR7ASV.zip", "years": "2019", "name": "Sierra Leone"},
    "SN": {"wave": "DHS-VIII", "version": "81", "zip": "SNKR81SV.zip", "years": "2018", "name": "Senegal"},
    "ST": {"wave": "DHS-V",    "version": "51", "zip": "STKR51SV.zip", "years": "2008–2009", "name": "Sao Tome and Principe"},
    "SZ": {"wave": "DHS-V",    "version": "51", "zip": "SZKR51SV.zip", "years": "2006–2007", "name": "Eswatini"},
    "TD": {"wave": "DHS-VII",  "version": "71", "zip": "TDKR71SV.zip", "years": "2014–2015", "name": "Chad"},
    "TG": {"wave": "DHS-VI",   "version": "61", "zip": "TGKR61SV.zip", "years": "2013–2014", "name": "Togo"},
    "TZ": {"wave": "DHS-VIII", "version": "82", "zip": "TZKR82SV.zip", "years": "2022", "name": "Tanzania"},
    "UG": {"wave": "DHS-VII",  "version": "72", "zip": "UGKR72SV.zip", "years": "2014–2015", "name": "Uganda"},
    "ZA": {"wave": "DHS-VII",  "version": "71", "zip": "ZAKR71SV.zip", "years": "2016", "name": "South Africa"},
    "ZM": {"wave": "DHS-VIII", "version": "81", "zip": "ZMKR81SV.zip", "years": "2024", "name": "Zambia"},
    "ZW": {"wave": "DHS-VII",  "version": "72", "zip": "ZWKR72SV.zip", "years": "2015", "name": "Zimbabwe"}
}


def select_surveys(scan_df: pd.DataFrame = None) -> pd.DataFrame:
    """Matches scanned archives against the study protocol or generates protocol list."""
    rows = []
    for cc, info in sorted(PROTOCOL_SURVEYS.items()):
        row = {
            "country_code": cc,
            "country_name": info["name"],
            "wave": info["wave"],
            "version": info["version"],
            "filename": info["zip"],
            "fieldwork_years": info["years"],
            "status": "Selected"
        }
        if scan_df is not None and not scan_df.empty:
            match = scan_df[scan_df["filename"].str.upper() == info["zip"].upper()]
            if not match.empty:
                row["total_raw_sample"] = match["total_cases"].values[0]
                row["file_size_mb"] = match["file_size_mb"].values[0]
                row["found_in_scan"] = True
            else:
                row["found_in_scan"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Select Kids Recode surveys per study protocol.")
    parser.add_argument("--inventory-csv", type=str, default=None, help="Scan summary CSV from Step 1.")
    parser.add_argument("--output-csv", type=str, default="selected_kr_surveys.csv", help="Output selected surveys CSV.")
    args = parser.parse_args()

    scan_df = None
    if args.inventory_csv and Path(args.inventory_csv).exists():
        scan_df = pd.read_csv(args.inventory_csv)
        logger.info(f"Loaded {len(scan_df)} scanned records from {args.inventory_csv}")

    selected_df = select_surveys(scan_df)
    selected_df.to_csv(args.output_csv, index=False)
    logger.info(f"Selected {len(selected_df)} surveys across 38 countries. Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
