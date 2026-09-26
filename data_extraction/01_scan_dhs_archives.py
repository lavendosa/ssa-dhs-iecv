#!/usr/bin/env python3
"""
01_scan_dhs_archives.py - DHS Raw Archive Scanner

Scans a directory of DHS ZIP archives (containing SPSS .SAV files),
extracts recode types, survey waves, country identifiers, and column metadata
without extracting full archives to disk.

Usage:
    python 01_scan_dhs_archives.py --data-dir /path/to/dhs_zips --output-csv survey_scan_results.csv
"""

import sys
import os
import io
import zipfile
import argparse
import logging
from pathlib import Path
import pandas as pd
import pyreadstat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Standard DHS Country and Recode Mappings
DHS_COUNTRIES = {
    "AO": "Angola", "BF": "Burkina Faso", "BJ": "Benin", "BU": "Burundi",
    "CD": "Democratic Republic of the Congo", "CF": "Central African Republic",
    "CG": "Congo", "CI": "Cote d'Ivoire", "CM": "Cameroon", "ET": "Ethiopia",
    "GA": "Gabon", "GH": "Ghana", "GM": "Gambia", "GN": "Guinea", "KE": "Kenya",
    "KM": "Comoros", "LB": "Liberia", "LS": "Lesotho", "MD": "Madagascar",
    "ML": "Mali", "MR": "Mauritania", "MW": "Malawi", "MZ": "Mozambique",
    "NG": "Nigeria", "NI": "Niger", "NM": "Namibia", "RW": "Rwanda",
    "SL": "Sierra Leone", "SN": "Senegal", "ST": "Sao Tome and Principe",
    "SZ": "Eswatini", "TD": "Chad", "TG": "Togo", "TZ": "Tanzania",
    "UG": "Uganda", "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe"
}

DHS_RECODES = {
    "KR": "Kids Recode",
    "IR": "Individual Women Recode",
    "HR": "Household Recode",
    "PR": "Household Member Recode",
    "BR": "Births Recode",
    "MR": "Men Recode",
    "CR": "Couples Recode"
}


def parse_dhs_filename(filename: str) -> dict:
    """Parses standard DHS archive stem into country, recode, and wave."""
    stem = Path(filename).stem.upper()
    if len(stem) >= 6:
        country_code = stem[:2]
        recode_type = stem[2:4]
        version = stem[4:-2] if stem.endswith("SV") else stem[4:]
        
        # Determine survey wave
        wave = "Unknown"
        if len(version) > 0 and version[0].isdigit():
            v_digit = int(version[0])
            waves = {1: "DHS-I", 2: "DHS-II", 3: "DHS-III", 4: "DHS-IV",
                     5: "DHS-V", 6: "DHS-VI", 7: "DHS-VII", 8: "DHS-VIII"}
            wave = waves.get(v_digit, f"Phase-{v_digit}")

        return {
            "filename": filename,
            "country_code": country_code,
            "country_name": DHS_COUNTRIES.get(country_code, f"Unknown ({country_code})"),
            "recode_type": recode_type,
            "recode_desc": DHS_RECODES.get(recode_type, f"Other ({recode_type})"),
            "version": version,
            "wave": wave
        }
    return {"filename": filename, "country_code": "Unknown", "recode_type": "Unknown"}


def scan_directory(data_dir: Path) -> pd.DataFrame:
    """Scans all ZIP archives in the specified directory."""
    records = []
    zip_files = sorted(list(data_dir.glob("*.zip")) + list(data_dir.glob("*.ZIP")))
    logger.info(f"Found {len(zip_files)} ZIP archives in {data_dir}")

    for idx, zpath in enumerate(zip_files, 1):
        parsed = parse_dhs_filename(zpath.name)
        file_size_mb = round(zpath.stat().st_size / (1024 * 1024), 2)
        parsed["file_size_mb"] = file_size_mb
        parsed["has_sav"] = False
        parsed["total_cases"] = None
        parsed["total_variables"] = None

        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                sav_members = [m for m in zf.namelist() if m.upper().endswith(".SAV")]
                if sav_members:
                    parsed["has_sav"] = True
                    target_sav = sav_members[0]
                    parsed["sav_member_name"] = target_sav
                    # Read metadata only (extremely fast, no memory overhead)
                    sav_bytes = zf.read(target_sav)
                    _, meta = pyreadstat.read_sav(io.BytesIO(sav_bytes), metadataonly=True)
                    parsed["total_cases"] = meta.number_rows
                    parsed["total_variables"] = meta.number_columns
        except Exception as e:
            parsed["error"] = str(e)
            logger.warning(f"Error inspecting {zpath.name}: {e}")

        records.append(parsed)
        if idx % 100 == 0 or idx == len(zip_files):
            logger.info(f"Scanned {idx}/{len(zip_files)} archives...")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Scan DHS ZIP archives and extract metadata.")
    parser.add_argument("--data-dir", type=str, default="./data/raw_dhs", help="Directory containing DHS ZIP files.")
    parser.add_argument("--output-csv", type=str, default="dhs_scan_summary.csv", help="Output CSV path.")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory '{data_path}' does not exist. Creating empty scan template.")
        # Create directory and output placeholder
        data_path.mkdir(parents=True, exist_ok=True)

    df_scan = scan_directory(data_path)
    df_scan.to_csv(args.output_csv, index=False)
    logger.info(f"Scan complete. Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
