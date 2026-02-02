# 🌾 Agriverse GEE Data & Inference Microservice

This microservice is a complete pipeline for agricultural monitoring. It handles everything from raw satellite data acquisition (LULC & Sentinel-2) via **Google Earth Engine (GEE)** to **Tiled Ensemble Inference** for crop mapping.

---

## 🛠️ Microservice Components

The microservice consolidates three main workflows into a single containerized environment:

### 1. LULC Downloader (`gee_punjab_lulc_downloader_v2.py`)
- **Purpose:** Fetches Dynamic World V1 land cover probabilities.
- **How it works:** 
  - Iterates through a shapefile of Punjab districts.
  - Computes a `mean` composite for the Rabi start period.
  - **Scaling:** Multiplies values by `10,000` and saves as `uint16` to preserve sub-pixel probability precision without using floating-point storage.
- **Output:** `output/Rabi_25_26/{District_Name}/{District_Name}_LULC_2526_final.tif`

### 2. Sentinel-2 Downloader (`gee_bulk_s2_rabi_downloader.py`)
- **Purpose:** Acquires monthly median composites for spectral analysis.
- **Dynamic Logic:** Automatically calculates all Rabi months from **October 2025** up to the **current month** (limit May 2026). This ensures that as new imagery becomes available (e.g., February, March), the script will automatically include them.
- **Cropland Masking:** To optimize storage and processing, this script uses a **Dynamic World Cropland Mask**. It only downloads Sentinel-2 pixels where the LULC label indicates crops.
- **Bands:** Includes `B11` (SWIR1) + indices (`NDVI`, `EVI`, `SAVI`, `NDWI`).
- **Scaling:** Data is scaled by `10,000` (`int16`).

### 3. Tiled Rabi Inference (`gee_bulk_rabi_inference.py`)
- **Purpose:** Runs a machine learning ensemble (XGBoost + CatBoost) to generate crop maps.
- **Workflow:** 
  1. Reads district boundaries from `pak_admin2.shp`.
  2. Locates the pre-downloaded LULC and Masked Sentinel-2 TIFs.
  3. Processes imagery in **1200x1200px tiles** to ensure stability on low-memory systems.
  4. Merges tiles into a finalized **GeoTIFF Crop Map**.

---

## 🚀 How to Run

### 1. Prerequisites
- **GEE Authentication:** Place your `salehazainab-key.json` in the root folder.
- **Resources:** Place your district shapefiles in `district shapefiles/` and your trained models in `models/rabi/`.

### 2. Start the Service
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

---

## 📡 API Workflow (Step-by-Step)

Follow these steps for automated processing. You can use **Windows PowerShell** or **Bash/Curl**.

### Step 1: Download LULC Data
Fetches the 8-band probability stack (Dynamic World) used for masking and inference.

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/download/lulc" -Method Post -ContentType "application/json" -Body '{"districts": ["ALL"]}'
```

*💡 Tip: Use `["ALL"]` to process every district in the shapefile automatically.*

**Curl:**
```bash
curl -X POST "http://localhost:8000/download/lulc" -H "Content-Type: application/json" -d '{"districts": ["Sheikhupura", "Islamabad"]}'
```

### Step 2: Download Masked Sentinel-2 Data
Automatically detects available Rabi months (Oct 2025 to Present) and downloads pixels masked by the cropland label.

**PowerShell (Bulk - All Districts):**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/download/sentinel" -Method Post -ContentType "application/json" -Body '{"districts": ["ALL"]}'
```

*Note: The script automatically **skips existing files**. If a download is interrupted, just run this again to resume.*

### Step 3: Run Tiled Inference
Generates the final 10m GeoTIFF crop map using the ensemble models.

**PowerShell (Bulk - All Districts):**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/inference/rabi" -Method Post -ContentType "application/json" -Body '{"districts": ["ALL"]}'
```

*Note: This will only succeed for districts that have **already downloaded** both LULC and Sentinel-2 data. If data is missing for a district, it will log an error for that specific district and move to the next one.*

---

## 📊 Monitoring & Results

### List Districts & Data Status
Use this to see which districts are valid and which ones already have data (with timestamps).

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/districts" -Method Get | ConvertTo-Json -Depth 5
```

---

### Check All Tasks
**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/tasks" -Method Get | ConvertTo-Json -Depth 5
```

### Check Specific Task
**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/task/<TASK_ID>" -Method Get
```

### Output Locations
- **Raw Downloads:** `./output/Rabi_25_26/{District}/`
- **Final Crop Maps:** `./result/Rabi_25_26/{District}/`

---

## 🧪 Testing with Islamabad
To verify your setup quickly, use Islamabad as a test case:
1. `docker-compose up -d`
2. Run the Step 1 (LULC) command for "Islamabad".
3. Run Step 2 (Sentinel) for "Islamabad".
4. Run Step 3 (Inference) for "Islamabad".

---

## 📂 Directory Structure (Volumes)

- `/app/output`: Raw GEE downloads.
- `/app/result`: Final GeoTIFF crop maps.
- `/app/models`: Trained XGBoost/CatBoost models.
- `/app/district shapefiles`: Administrative boundaries.

---

---

## 🎨 Crop Map Legend & Format

The output is a **Cloud Optimized GeoTIFF (COG)** in **EPSG:3857** (Web Mercator) with a **10m resolution**. This format is optimized for web mapping and high-speed tiling.

| Value | Class Name |
| :--- | :--- |
| **0** | Wheat |
| **1** | Rice |
| **2** | Maize |
| **3** | Cotton |
| **4** | Sugarcane |
| **5** | Vegetation / Other Crops |
| **6** | Bare / Fallow |
| **7** | Water |
| **255** | No Data (Masked Area) |

---

## 🛡️ Best Practices
- **Persistence:** The container is set to `unless-stopped` to ensure tasks continue if the host restarts.
- **Memory:** Tiling is configured for 1200px. If running on low-memory systems, you can reduce `tile_size` in `gee_bulk_rabi_inference.py`.
- **Dynamic Updates:** The Sentinel-2 downloader automatically checks for "current" imagery. When February imagery becomes available on GEE, simply running Step 2 again will download it.
