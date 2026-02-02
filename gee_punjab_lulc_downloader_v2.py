import ee
import geemap
import os
import json
import shapefile
import time
from datetime import datetime
import numpy as np

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
# Dynamic World lag is small (~2-4 days). We'll set a safe end date.
LULC_START = "2025-11-01" # Increased window to ensure data
LULC_END = "2026-01-25"     # Latest known available date
ROOT_OUTPUT_DIR = "/app/output/Rabi_25_26"
SHAPEFILE_PATH = "/app/district shapefiles/pak_admin2.shp"

os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

# Bands we need for our Rabi model
DW_BANDS = ['bare', 'built', 'crops', 'flooded_vegetation', 'grass', 'shrub_and_scrub', 'trees', 'water']

def shape_to_ee_geometry(shape):
    """
    Robust conversion of pyshp parts to ee.Geometry.
    Correctly recognizes separate polygons in a MultiPolygon.
    """
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    
    all_polys = []
    for i in range(len(shape.parts)):
        ring = points[parts[i]:parts[i+1]]
        # Convert to list of lists for EE
        ring_list = [list(p) for p in ring]
        # Ensure Closed
        if ring_list[0] != ring_list[-1]:
            ring_list.append(ring_list[0])
        all_polys.append(ring_list)
    
    # We return as a MultiPolygon to handle islands and holes correctly
    # Note: Earth Engine interprets overlapping rings as holes automatically 
    # if they are in the same Polygon list, but for simplicity we use MultiPolygon.
    return ee.Geometry.MultiPolygon([all_polys])

def download_lulc_v3(district_name, geom):
    try:
        dist_dir = os.path.join(ROOT_OUTPUT_DIR, district_name.replace(" ", "_"))
        os.makedirs(dist_dir, exist_ok=True)
        filename = os.path.join(dist_dir, f"{district_name}_LULC_2526_final.tif")
        
        if os.path.exists(filename):
            print(f"   ⏩ {district_name} exists. Skipping.")
            return

        print(f"📦 Processing {district_name} (Range: {LULC_START} to {LULC_END})...")

        # 1. Fetch DW
        dw_col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                  .filterBounds(geom)
                  .filterDate(LULC_START, LULC_END)
                  .select(DW_BANDS))
        
        count = dw_col.size().getInfo()
        if count == 0:
            print(f"   ⚠️ No data for {district_name} in {LULC_START} to {LULC_END}")
            return

        # 2. Mean & Scale to uint16 (0-10,000)
        # We multiply by 10,000 to keep 4 decimal precision
        lulc_img = dw_col.mean().clip(geom).multiply(10000).uint16()

        print(f"📦 {district_name} ({count} images) -> Downloading uint16 (Scale: 10,000x)...")
        geemap.download_ee_image(
            lulc_img,
            filename=filename,
            scale=10,
            region=geom,
            crs='EPSG:4326',
            overwrite=True,
            max_tile_size=2,
            max_tile_dim=1024
        )
        if os.path.exists(filename):
            print(f"   ✅ Saved: {filename} ({os.path.getsize(filename)/1024/1024:.2f} MB)")

    except Exception as e:
        print(f"   ❌ Failed {district_name}: {e}")

def get_punjab_records():
    sf = shapefile.Reader(SHAPEFILE_PATH)
    fields = [f[0] for f in sf.fields[1:]]
    records = []
    
    adm1_idx = fields.index('adm1_name')
    adm2_idx = fields.index('adm2_name')
    
    for i, rec in enumerate(sf.records()):
        if rec[adm1_idx] in ["Punjab", "Islamabad", "F.C.T."]:
            records.append({
                'name': rec[adm2_idx],
                'shape': sf.shape(i)
            })
    return sorted(records, key=lambda x: x['name'])

if __name__ == "__main__":
    print(f"🚀 PUNJAB LULC BULK (ADM2 LOCAL SHAPES) - uint16 10,000x MODE")
    districts = get_punjab_records()
    print(f"📋 Processing {len(districts)} districts...")

    for d in districts:
        name = d['name']
        geom = shape_to_ee_geometry(d['shape'])
        download_lulc_v3(name, geom)
        time.sleep(1)

    print("\n🎉 DONE.")
