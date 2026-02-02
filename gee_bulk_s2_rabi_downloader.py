import ee
import geemap
import os
import json
import shapefile
import time
from datetime import datetime

# ==============================================================================
# 🔐 AUTHENTICATION
# ==============================================================================
possible_key_paths = [
    "/app/salehazainab-key.json",
    "c:/Users/GreenAI/Documents/Agriverse 2.1/gee_download_workflow/salehazainab-key.json"
]
KEY_FILE = next((p for p in possible_key_paths if os.path.exists(p)), None)

if not KEY_FILE:
    exit("❌ NO SERVICE ACCOUNT KEY FOUND.")

with open(KEY_FILE, 'r') as f:
    key_data = json.load(f)
    SERVICE_ACCOUNT = key_data['client_email']
    PROJECT_ID = key_data.get('project_id', 'salehazainab')

ee.Initialize(ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE), project=PROJECT_ID)

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
# DYNAMIC RABI MONTHS (October 2025 to May 2026)
def calculate_rabi_months():
    rabi_months = []
    # Start: October 2025
    # End: May 2026 or Current Month (whichever is earlier)
    current_date = datetime.now()
    rabi_start = datetime(2025, 10, 1)
    # Define absolute Rabi end date
    rabi_end = datetime(2026, 5, 31)
    
    # We want to check up to the current month if it's within Rabi season
    limit_date = min(current_date, rabi_end)
    
    year, month = rabi_start.year, rabi_start.month
    while datetime(year, month, 1) <= limit_date:
        rabi_months.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return rabi_months

MONTHS = calculate_rabi_months()
print(f"📅 Target Rabi Months: {MONTHS}")
SHAPEFILE_PATH = "/app/district shapefiles/pak_admin2.shp"
ROOT_OUTPUT_DIR = "/app/output/Rabi_25_26"

os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

def shape_to_ee_geometry(shape):
    """Robust conversion of pyshp parts to ee.Geometry."""
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    all_polys = []
    for i in range(len(shape.parts)):
        ring = points[parts[i]:parts[i+1]]
        ring_list = [list(p) for p in ring]
        if ring_list[0] != ring_list[-1]:
            ring_list.append(ring_list[0])
        all_polys.append(ring_list)
    return ee.Geometry.MultiPolygon([all_polys])

def get_punjab_records():
    sf = shapefile.Reader(SHAPEFILE_PATH)
    fields = [f[0] for f in sf.fields[1:]]
    records = []
    adm1_idx = fields.index('adm1_name')
    adm2_idx = fields.index('adm2_name')
    for i, rec in enumerate(sf.records()):
        if rec[adm1_idx] in ["Punjab", "Islamabad", "F.C.T."]:
            records.append({'name': rec[adm2_idx], 'shape': sf.shape(i)})
    return sorted(records, key=lambda x: x['name'])

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = image.expression('2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))', {
        'B8': image.select('B8'), 'B4': image.select('B4'), 'B2': image.select('B2')
    }).rename('EVI')
    savi = image.expression('((B8 - B4) / (B8 + B4 + 0.5)) * 1.5', {
        'B8': image.select('B8'), 'B4': image.select('B4')
    }).rename('SAVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([ndvi, evi, savi, ndwi])

def get_cropland_mask(geom):
    """Generates a binary mask based on Dynamic World (class 4 = crops)"""
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
        .filterBounds(geom) \
        .filterDate("2024-01-01", "2025-06-01") \
        .select('label') \
        .mode()
    return dw.eq(4)

def download_district_s2(district_name, geom):
    dist_dir = os.path.join(ROOT_OUTPUT_DIR, district_name.replace(" ", "_"))
    os.makedirs(dist_dir, exist_ok=True)
    
    # Generate the mask once for the district
    crop_mask = get_cropland_mask(geom)
    
    print(f"\n🌍 Starting Masked Sentinel-2 Bulk for {district_name}...")
    
    for month_str in MONTHS:
        filename = os.path.join(dist_dir, f"{district_name}_{month_str}.tif")
        if os.path.exists(filename) and os.path.getsize(filename) > 1000:
            print(f"   ⏩ {month_str} already exists. Skipping.")
            continue

        try:
            start_date = f"{month_str}-01"
            y, m = map(int, month_str.split('-'))
            end_date = f"{y}-{m+1:02d}-01" if m < 12 else f"{y+1}-01-01"
            
            s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                      .filterBounds(geom)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                      .map(mask_s2_clouds)
                      .map(add_indices))
            
            count = s2_col.size().getInfo()
            if count == 0:
                print(f"   ⚠️ No data for {month_str}")
                continue

            # Composite, Select, and Apply Mask
            composite = s2_col.median().clip(geom).select(['B11', 'EVI', 'NDVI', 'NDWI', 'SAVI'])
            
            # Apply the Cropland Mask (Pixels outside cropland become NoData)
            masked_img = composite.updateMask(crop_mask)
            
            # Scale to int16 (reflectance/indices * 10,000)
            final_img = masked_img.multiply(10000).int16()

            print(f"   🛰️ Downloading {month_str} (Count: {count})...")
            geemap.download_ee_image(
                final_img,
                filename=filename,
                scale=10,
                region=geom,
                crs='EPSG:4326',
                overwrite=True,
                max_tile_size=2,
                max_tile_dim=1024
            )
            print(f"   ✅ Saved: {month_str}")
            
        except Exception as e:
            print(f"   ❌ Error in {month_str}: {e}")

if __name__ == "__main__":
    start_time = datetime.now()
    districts = get_punjab_records()
    print(f"🚀 PUNJAB SENTINEL-2 BULK DOWNLOAD (RABI 25-26)")
    print(f"📋 Found {len(districts)} districts to process.")

    for d in districts:
        name = d['name']
        geom = shape_to_ee_geometry(d['shape'])
        download_district_s2(name, geom)
        time.sleep(2)

    print(f"\n🎉 ALL SENTINEL-2 DOWNLOADS FINISHED.")
    print(f"🕒 Runtime: {datetime.now() - start_time}")
