# Data Included in This Private Repository

This folder contains the DACON/ETRI competition data files that were found on
the local computer and copied into the private GitHub repository.

## Folder Layout

| Path | Contents |
| --- | --- |
| `raw/` | Raw `ch2025_*.parquet` sensor/lifelog tables. |
| `meta/` | Label file, sample submission, and metrics description. |
| `probe/` | Schema probe generated during the original project. |

## Raw Sensor Tables

| File | Role |
| --- | --- |
| `raw/ch2025_mACStatus.parquet` | Smartphone charging status. |
| `raw/ch2025_mActivity.parquet` | Smartphone activity state. |
| `raw/ch2025_mBle.parquet` | BLE scan/context records. |
| `raw/ch2025_mGps.parquet` | GPS/location records. |
| `raw/ch2025_mLight.parquet` | Smartphone light sensor records. |
| `raw/ch2025_mScreenStatus.parquet` | Smartphone screen-use status. |
| `raw/ch2025_mUsageStats.parquet` | App usage records. |
| `raw/ch2025_mWifi.parquet` | Wi-Fi scan/context records. |
| `raw/ch2025_wHr.parquet` | Wearable heart-rate records. |
| `raw/ch2025_wLight.parquet` | Wearable light sensor records. |
| `raw/ch2025_wPedo.parquet` | Wearable pedometer/activity records. |

## Metadata and Labels

| File | Role |
| --- | --- |
| `meta/ch2026_metrics_train.csv` | Training labels for Q1-Q3 and S1-S4. |
| `meta/ch2026_submission_sample.csv` | DACON submission template. |
| `meta/ch2026_metrics_description.pdf` | Competition metric/label description. |
| `probe/raw_parquet_schema_probe.csv` | Local schema summary from the original work. |

## Missing File Note

The schema probe from the original project references
`ch2025_mAmbience.parquet`, but that parquet file was not found in
`C:\Users\jungheeyoung\AppData\Local\Temp` when this repository was updated.
Only the previously generated head preview exists in the old project folder.
