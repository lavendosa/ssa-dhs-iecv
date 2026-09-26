# Obtaining Demographic and Health Surveys (DHS) Microdata

This document describes how an independent researcher can obtain the raw Demographic and Health Surveys (DHS) microdata analyzed in this study in full compliance with the DHS Data Sharing Policy.

---

## 1. Ethical Governance and Data Use Restrictions

The microdata analyzed in this study are proprietary data managed by **The DHS Program** (funded by the United States Agency for International Development [USAID] and implemented by ICF). 

Under the DHS Data Sharing Agreement:
- **Redistribution of raw microdata is strictly prohibited.** No unauthorized third party may redistribute raw or derived individual-level microdata.
- Datasets are made available **free of charge** to legitimate researchers for non-commercial academic and public health research.
- All DHS data are strictly anonymized: cluster geographic coordinates are displaced (jittered by up to 2 km in urban areas and 5 km in rural areas, with 1% displaced up to 10 km) to prevent household identification.

---

## 2. Step-by-Step Registration and Access Procedure

### Step 1: Create a DHS User Account
1. Visit the DHS Program Portal: [https://dhsprogram.com](https://dhsprogram.com).
2. Click **"Log In"** (or directly visit [https://dhsprogram.com/data/new-user-registration.cfm](https://dhsprogram.com/data/new-user-registration.cfm)).
3. Fill out the registration form providing your institutional affiliation, academic email address, and professional background.

### Step 2: Register a New Research Project
1. Once logged in, navigate to **"Data -> Download Datasets"**.
2. Select **"Register for New Project"**.
3. Provide project details:
   - **Project Title:** e.g., *Machine Learning Prediction of Unmet Pediatric Healthcare Need in Sub-Saharan Africa*
   - **Description / Research Abstract:** State that you are conducting secondary analysis of pediatric illness care-seeking modules (fever, diarrhea, acute respiratory symptoms) across Sub-Saharan African countries to evaluate healthcare access and clinical prediction models.
   - **Target Countries:** Select the 38 sub-Saharan African countries catalogued in `data_extraction/dhs_survey_inventory.csv`.

### Step 3: Approval and Access Grant
- Most standard DHS project applications are automatically reviewed and approved within **24 to 48 business hours**.
- You will receive an official approval email with access credentials.

### Step 4: Download the Required Datasets
1. Log in to the DHS Download Manager: [https://dhsprogram.com/data/dataset_admin/index.cfm](https://dhsprogram.com/data/dataset_admin/index.cfm).
2. For each of the 38 countries listed in `dhs_survey_inventory.csv`, select the specified survey wave and download the **Kids Recode (KR)** dataset in **SPSS format (`.SAV`)**.
   - Standard filename format: `[CC]KR[version]SV.ZIP` (e.g., `AOKR81SV.ZIP` for Angola DHS-VIII 2023–2024, `CDKR81SV.ZIP` for DR Congo DHS-VIII 2023–2024, `NGKR8BSV.ZIP` for Nigeria DHS-VIII 2023–2024).
   - Alternatively, Stata (`.DTA`) format can be used with slight adjustments to script file readers.
3. Save all downloaded `.ZIP` or `.SAV` files into a local folder (e.g., `./data/raw_dhs/`).

---

## 3. Included Survey Inventory (38 Countries)

The complete inventory of the 38 selected national surveys is documented in [`dhs_survey_inventory.csv`](file:///data_extraction/dhs_survey_inventory.csv) and summarized below:

| ISO Code | Country | DHS Wave | Recode | SPSS ZIP Archive | Survey Years |
|:---|:---|:---|:---|:---|:---|
| AO | Angola | DHS-VIII | 81 | `AOKR81SV.zip` | 2023–2024 |
| BF | Burkina Faso | DHS-VIII | 81 | `BFKR81SV.zip` | 2021 |
| BJ | Benin | DHS-VII | 71 | `BJKR71SV.zip` | 2017–2018 |
| BU | Burundi | DHS-VII | 71 | `BUKR71SV.zip` | 2016–2017 |
| CD | Democratic Republic of the Congo | DHS-VIII | 81 | `CDKR81SV.zip` | 2023–2024 |
| CF | Central African Republic | DHS-III | 31 | `CFKR31SV.zip` | 1994–1995 |
| CG | Congo | DHS-VI | 61 | `CGKR61SV.zip` | 2011–2012 |
| CI | Côte d’Ivoire | DHS-VIII | 81 | `CIKR81SV.zip` | 2021 |
| CM | Cameroon | DHS-VIII | 82 | `CMKR82SV.zip` | 2022 |
| ET | Ethiopia | DHS-VII | 71 | `ETKR71SV.zip` | 2016 |
| GA | Gabon | DHS-VII | 71 | `GAKR71SV.zip` | 2019–2021 |
| GH | Ghana | DHS-VIII | 8C | `GHKR8CSV.zip` | 2022–2023 |
| GM | Gambia | DHS-VIII | 81 | `GMKR81SV.zip` | 2019–2020 |
| GN | Guinea | DHS-VIII | 82 | `GNKR82SV.zip` | 2021 |
| KE | Kenya | DHS-VIII | 8C | `KEKR8CSV.zip` | 2022 |
| KM | Comoros | DHS-VI | 61 | `KMKR61SV.zip` | 2012 |
| LB | Liberia | DHS-VIII | 81 | `LBKR81SV.zip` | 2022 |
| LS | Lesotho | DHS-VIII | 81 | `LSKR81SV.zip` | 2023–2024 |
| MD | Madagascar | DHS-VIII | 81 | `MDKR81SV.zip` | 2021 |
| ML | Mali | DHS-VIII | 83 | `MLKR83SV.zip` | 2021 |
| MR | Mauritania | DHS-VII | 71 | `MRKR71SV.zip` | 2019–2021 |
| MW | Malawi | DHS-VIII | 81 | `MWKR81SV.zip` | 2024 |
| MZ | Mozambique | DHS-VIII | 81 | `MZKR81SV.zip` | 2022–2023 |
| NG | Nigeria | DHS-VIII | 8B | `NGKR8BSV.zip` | 2023–2024 |
| NI | Niger | DHS-VI | 61 | `NIKR61SV.zip` | 2012 |
| NM | Namibia | DHS-VI | 61 | `NMKR61SV.zip` | 2013 |
| RW | Rwanda | DHS-VIII | 81 | `RWKR81SV.zip` | 2019–2020 |
| SL | Sierra Leone | DHS-VII | 7A | `SLKR7ASV.zip` | 2019 |
| SN | Senegal | DHS-VIII | 81 | `SNKR81SV.zip` | 2018 |
| ST | São Tomé and Príncipe | DHS-V | 51 | `STKR51SV.zip` | 2008–2009 |
| SZ | Eswatini | DHS-V | 51 | `SZKR51SV.zip` | 2006–2007 |
| TD | Chad | DHS-VII | 71 | `TDKR71SV.zip` | 2014–2015 |
| TG | Togo | DHS-VI | 61 | `TGKR61SV.zip` | 2013–2014 |
| TZ | United Republic of Tanzania | DHS-VIII | 82 | `TZKR82SV.zip` | 2022 |
| UG | Uganda | DHS-VII | 72 | `UGKR72SV.zip` | 2014–2015 |
| ZA | South Africa | DHS-VII | 71 | `ZAKR71SV.zip` | 2016 |
| ZM | Zambia | DHS-VIII | 81 | `ZMKR81SV.zip` | 2024 |
| ZW | Zimbabwe | DHS-VII | 72 | `ZWKR72SV.zip` | 2015 |

---

## 4. Setting the Local Data Directory

Once your raw archive files are downloaded, specify their location via an environment variable or command-line flag:

```bash
export DHS_RAW_DATA_DIR="/path/to/dhs_archives"
```

Alternatively, to test the entire pipeline without downloading the full 4.5 GB DHS dataset, run the included **Synthetic Data Generator** (`harmonization/synthetic_data_generator.py`), which constructs an identical schema cohort matching all marginal and conditional properties.
