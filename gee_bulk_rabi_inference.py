import os
import json
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from catboost import CatBoostClassifier
from datetime import datetime
import xgboost as xgb
import gc
import glob
import shapefile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
SEASON = "rabi"
BASE_DIR = "/app"
INPUT_DIR = f"{BASE_DIR}/output/Rabi_25_26"
MODEL_PATH = f"{BASE_DIR}/models/{SEASON}"
RESULT_DIR = f"{BASE_DIR}/result/Rabi_25_26"
SHAPEFILE_PATH = "/app/district shapefiles/pak_admin2.shp"

# Frontend expects: 0=Wheat, 1=Rice, 2=Maize, 3=Cotton, 4=Sugarcane, 5=Vegetation, 6=Bare/Fallow, 7=Water
FIXED_CLASS_MAPPING = {
    'Wheat': 0,
    'Rice': 1,
    'Maize': 2,
    'Cotton': 3,
    'Sugarcane': 4,
    'Vegetation / Other Crops': 5,
    'Bare / Fallow': 6,
    'Water': 7
}

_models = {}

def load_models():
    if _models: return
    print("🧠 Loading models into memory...", flush=True)
    _models['xgb'] = joblib.load(f"{MODEL_PATH}/xgb/xgb_model.pkl")
    _models['scaler'] = joblib.load(f"{MODEL_PATH}/xgb/scaler.pkl")
    _models['cat_le'] = joblib.load(f"{MODEL_PATH}/catboost/label_encoder.pkl")
    _models['cat'] = CatBoostClassifier()
    _models['cat'].load_model(f"{MODEL_PATH}/catboost/catboost_model.cbm")
    with open(f"{MODEL_PATH}/xgb/feature_columns.json", "r") as f:
        _models['xgb_features'] = json.load(f)
    with open(f"{MODEL_PATH}/catboost/feature_columns.json", "r") as f:
        _models['cat_features'] = json.load(f)

def run_inference(df_chunk):
    if df_chunk.empty: return [], []
    
    # Scale correction: integer columns / 10,000.0
    feat_cols = [c for c in df_chunk.columns if c not in ['Latitude', 'Longitude']]
    df_chunk[feat_cols] = df_chunk[feat_cols].astype(np.float32) / 10000.0
    
    df_chunk.replace([np.inf, -np.inf], 0, inplace=True)
    df_chunk.fillna(0, inplace=True)
    
    weight_xgb, weight_cat = 0.4, 0.6
    
    # XGBoost
    X_xgb = df_chunk.reindex(columns=_models['xgb_features']).fillna(0).astype(np.float32)
    X_xgb_scaled = _models['scaler'].transform(X_xgb)
    xgb_probs = _models['xgb'].predict_proba(X_xgb_scaled).astype(np.float32)
    
    # CatBoost
    X_cat = df_chunk.reindex(columns=_models['cat_features']).fillna(0).astype(np.float32)
    cat_probs = _models['cat'].predict_proba(X_cat).astype(np.float32)
    
    ensemble_probs = (weight_cat * cat_probs + weight_xgb * xgb_probs)
    indices = np.argmax(ensemble_probs, axis=1)
    preds = _models['cat_le'].inverse_transform(indices)
    confs = np.max(ensemble_probs, axis=1).astype(np.float32)
    
    return preds, confs

def process_district(district_name):
    print(f"\n🚜 Starting Dynamic Inference for {district_name}...")
    dist_input_dir = os.path.join(INPUT_DIR, district_name.replace(" ", "_"))
    dist_result_dir = os.path.join(RESULT_DIR, district_name.replace(" ", "_"))
    tiles_dir = os.path.join(dist_result_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    output_crop_map = os.path.join(dist_result_dir, f"{district_name}_Rabi_2526_CropMap.tif")
    if os.path.exists(output_crop_map):
        print(f"   ⏩ {district_name} map already exists. Skipping.")
        return

    # 🔎 DYNAMICALLY FIND SENTINEL INPUTS (YYYY-MM.tif)
    pattern = os.path.join(dist_input_dir, f"{district_name}_202[0-9]-[0-9][0-9].tif")
    month_files = sorted(glob.glob(pattern))
    
    if not month_files:
        print(f"   ⚠️ Skipping {district_name}. No Sentinel inputs found in {dist_input_dir}")
        return

    print(f"   📂 Found {len(month_files)} months: {[os.path.basename(f) for f in month_files]}")

    lulc_file = f"{dist_input_dir}/{district_name}_LULC_2526_final.tif"
    if not os.path.exists(lulc_file):
        print(f"   ⚠️ Skipping {district_name}. Missing LULC file.")
        return

    # Load available months
    src_files = [rasterio.open(f) for f in month_files]
    
    # Logic: If we have < 8 months, replicate the latest available month to fill M1-M8
    # If we have [M1, M2, M3, M4], the list becomes [M1, M2, M3, M4, M4, M4, M4, M4]
    latest_src = src_files[-1]
    full_src_list = src_files + [latest_src] * (8 - len(src_files))
    # Cap at 8 if we somehow had more
    full_src_list = full_src_list[:8]
    
    lulc_src = rasterio.open(lulc_file)
    ref_src = latest_src # Use most recent month as reference for geometry
    width, height = ref_src.width, ref_src.height
    transform, crs = ref_src.transform, ref_src.crs
    
    tile_size = 1200
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            h, w = min(tile_size, height-y), min(tile_size, width-x)
            window = Window(x, y, w, h)
            tile_tif = os.path.join(tiles_dir, f"tile_{y}_{x}.tif")
            if os.path.exists(tile_tif): continue

            # Mask using Jan NDVI
            mask_data = ref_src.read(3, window=window)
            valid_mask = np.isfinite(mask_data) & (mask_data != 0)
            
            if not np.any(valid_mask):
                with rasterio.open(tile_tif, 'w', driver='GTiff', height=h, width=w, count=1, 
                                  dtype='uint8', crs=crs, transform=rasterio.windows.transform(window, transform), nodata=255) as out:
                    out.write(np.full((h, w), 255, dtype='uint8'), 1)
                continue

            rows, cols = np.where(valid_mask)
            lons, lats = rasterio.windows.transform(window, transform) * (cols, rows)
            tile_df = pd.DataFrame({'Latitude': lats, 'Longitude': lons})
            
            # Extract 8 months (including proxies)
            band_names = ['B11', 'EVI', 'NDVI', 'NDWI', 'SAVI']
            for m_idx, s_src in enumerate(full_src_list, start=1):
                for b_idx, b_name in enumerate(band_names, start=1):
                    data = s_src.read(b_idx, window=window)
                    tile_df[f"{b_name}_M{m_idx}"] = data[rows, cols].astype(np.int32)

            # LULC
            dw_bands = ['bare', 'built', 'crops', 'flooded_vegetation', 'grass', 'shrub_and_scrub', 'trees', 'water']
            for b_idx, b_name in enumerate(dw_bands, start=1):
                data = lulc_src.read(b_idx, window=window)
                tile_df[f"DW_{b_name}"] = data[rows, cols].astype(np.int32)
            
            # NDVI calculated features
            ndvi_cols = [f"NDVI_M{m}" for m in range(1, 9)]
            nv = tile_df[ndvi_cols].values.astype(np.float32) / 10000.0
            tile_df['ndvi_max'] = (np.max(nv, axis=1)*10000).astype(np.int32)
            tile_df['ndvi_min'] = (np.min(nv, axis=1)*10000).astype(np.int32)
            tile_df['ndvi_amplitude'] = tile_df['ndvi_max'] - tile_df['ndvi_min']
            tile_df['ndvi_auc'] = (np.sum(nv, axis=1)*10000).astype(np.int32)
            tile_df['ndvi_time_of_max_idx'] = (np.argmax(nv, axis=1) + 1).astype(np.uint8)

            preds, _ = run_inference(tile_df)
            
            grid = np.full((h, w), 255, dtype='uint8')
            grid[rows, cols] = [FIXED_CLASS_MAPPING.get(p, 255) for p in preds]
            
            with rasterio.open(tile_tif, 'w', driver='GTiff', height=h, width=w, count=1, 
                              dtype='uint8', crs=crs, transform=rasterio.windows.transform(window, transform), nodata=255) as out:
                out.write(grid, 1)

def generate_cog(temp_wgs84_path, output_cog_path):
    """Matches the tile_merger.py architecture for professional COG generation."""
    print(f"🔁 Reprojecting to EPSG:3857 (Web Mercator) at 10m resolution...")
    repro_tif = output_cog_path.replace(".tif", "_repro.tif")
    
    with rasterio.open(temp_wgs84_path) as src:
        # 1. Calculate the ideal 10m transform in Web Mercator
        transform_3857, width_3857, height_3857 = calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds, resolution=10
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": "EPSG:3857",
            "transform": transform_3857,
            "width": width_3857,
            "height": height_3857,
            "compress": "DEFLATE",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "nodata": 255,
            "dtype": "uint8"
        })
        
        # 2. Perform the Reprojection
        with rasterio.open(repro_tif, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform_3857,
                dst_crs="EPSG:3857",
                resampling=Resampling.nearest
            )
            # 3. Add same Metadata tags as tile_merger.py
            dst.update_tags(**{f"class_{idx}": cls for cls, idx in FIXED_CLASS_MAPPING.items()})
            dst.update_tags(OVR_RESAMPLING_ALG="NEAREST", AREA_OR_POINT="Area")

    # 4. Final COG Translation (Internal tiling and overviews)
    print("🛠 Finalizing Cloud Optimized GeoTIFF...")
    cog_profile = cog_profiles.get("lzw")
    cog_translate(repro_tif, output_cog_path, cog_profile, in_memory=False, quiet=True)
    
    # Clean up intermediate files
    if os.path.exists(repro_tif): os.remove(repro_tif)
    print(f"✅ COG saved to: {output_cog_path}")

def process_district(district_name):
    print(f"\n🚜 Starting Inference for {district_name}...")
    dist_input_dir = os.path.join(INPUT_DIR, district_name.replace(" ", "_"))
    dist_result_dir = os.path.join(RESULT_DIR, district_name.replace(" ", "_"))
    tiles_dir = os.path.join(dist_result_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    output_crop_map = os.path.join(dist_result_dir, f"{district_name}_Rabi_2526_CropMap.tif")
    if os.path.exists(output_crop_map):
        print(f"   ⏩ {district_name} map already exists. Skipping.")
        return

    # 🔎 Dynamic Detection of months
    pattern = os.path.join(dist_input_dir, f"{district_name}_202[0-9]-[0-9][0-9].tif")
    month_files = sorted(glob.glob(pattern))
    
    if not month_files:
        print(f"   ⚠️ Skipping {district_name}. No Sentinel inputs found.")
        return

    lulc_file = f"{dist_input_dir}/{district_name}_LULC_2526_final.tif"
    if not os.path.exists(lulc_file):
        print(f"   ⚠️ Skipping {district_name}. Missing LULC file.")
        return

    # Prepare Source Handles
    src_files = [rasterio.open(f) for f in month_files]
    latest_src = src_files[-1]
    full_src_list = src_files + [latest_src] * (8 - len(src_files))
    full_src_list = full_src_list[:8]
    
    lulc_src = rasterio.open(lulc_file)
    ref_src = latest_src
    width, height = ref_src.width, ref_src.height
    transform, crs = ref_src.transform, ref_src.crs
    
    tile_size = 1200
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            h, w = min(tile_size, height-y), min(tile_size, width-x)
            window = Window(x, y, w, h)
            tile_tif = os.path.join(tiles_dir, f"tile_{y}_{x}.tif")
            if os.path.exists(tile_tif): continue

            # Masking using NDVI
            mask_data = ref_src.read(3, window=window)
            valid_mask = (mask_data != 0) & np.isfinite(mask_data)
            
            if not np.any(valid_mask):
                with rasterio.open(tile_tif, 'w', driver='GTiff', height=h, width=w, count=1, 
                                  dtype='uint8', crs=crs, transform=rasterio.windows.transform(window, transform), nodata=255) as out:
                    out.write(np.full((h, w), 255, dtype='uint8'), 1)
                continue

            rows, cols = np.where(valid_mask)
            lons, lats = rasterio.windows.transform(window, transform) * (cols, rows)
            tile_df = pd.DataFrame({'Latitude': lats, 'Longitude': lons})
            
            # Feature extraction
            band_names = ['B11', 'EVI', 'NDVI', 'NDWI', 'SAVI']
            for m_idx, s_src in enumerate(full_src_list, start=1):
                for b_idx, b_name in enumerate(band_names, start=1):
                    data = s_src.read(b_idx, window=window)
                    tile_df[f"{b_name}_M{m_idx}"] = data[rows, cols].astype(np.int32)

            dw_bands = ['bare', 'built', 'crops', 'flooded_vegetation', 'grass', 'shrub_and_scrub', 'trees', 'water']
            for b_idx, b_name in enumerate(dw_bands, start=1):
                data = lulc_src.read(b_idx, window=window)
                tile_df[f"DW_{b_name}"] = data[rows, cols].astype(np.int32)
            
            nv = tile_df[[f"NDVI_M{m}" for m in range(1, 9)]].values.astype(np.float32) / 10000.0
            tile_df['ndvi_max'] = (np.max(nv, axis=1)*10000).astype(np.int32)
            tile_df['ndvi_min'] = (np.min(nv, axis=1)*10000).astype(np.int32)
            tile_df['ndvi_amplitude'] = tile_df['ndvi_max'] - tile_df['ndvi_min']
            tile_df['ndvi_auc'] = (np.sum(nv, axis=1)*10000).astype(np.int32)
            tile_df['ndvi_time_of_max_idx'] = (np.argmax(nv, axis=1) + 1).astype(np.uint8)

            preds, _ = run_inference(tile_df)
            
            grid = np.full((h, w), 255, dtype='uint8')
            grid[rows, cols] = [FIXED_CLASS_MAPPING.get(p, 255) for p in preds]
            
            with rasterio.open(tile_tif, 'w', driver='GTiff', height=h, width=w, count=1, 
                              dtype='uint8', crs=crs, transform=rasterio.windows.transform(window, transform), nodata=255) as out:
                out.write(grid, 1)

    # 🏗️ Merge tiles into temporary master
    temp_master = os.path.join(dist_result_dir, "wgs84_master.tif")
    print(f"🏗️ Merging tiles for {district_name}...")
    with rasterio.open(temp_master, 'w', driver='GTiff', height=height, width=width, count=1, 
                      dtype='uint8', crs=crs, transform=transform, nodata=255) as dst:
        tile_files = glob.glob(f"{tiles_dir}/tile_*.tif")
        for tif in tile_files:
            with rasterio.open(tif) as ts:
                name_parts = os.path.basename(tif).replace('.tif','').split('_')
                dst.write(ts.read(1), 1, window=Window(int(name_parts[2]), int(name_parts[1]), ts.width, ts.height))

    # 🚀 Generate the Professional COG
    generate_cog(temp_master, output_crop_map)

    # Cleanup
    if os.path.exists(temp_master): os.remove(temp_master)
    print(f"✅ DISTRICT {district_name} COMPLETE.")

if __name__ == "__main__":
    load_models()
    sf = shapefile.Reader(SHAPEFILE_PATH)
    fields = [f[0] for f in sf.fields[1:]]
    adm1_idx = fields.index('adm1_name')
    adm2_idx = fields.index('adm2_name')
    districts = sorted([rec[adm2_idx] for rec in sf.records() if rec[adm1_idx] in ["Punjab", "Islamabad", "F.C.T."]])
    
    for dname in districts:
        process_district(dname)
