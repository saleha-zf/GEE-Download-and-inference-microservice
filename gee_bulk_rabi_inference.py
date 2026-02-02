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

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
SEASON = "rabi"
BASE_DIR = "/app"
INPUT_DIR = f"{BASE_DIR}/output/Rabi_25_26"
MODEL_PATH = f"{BASE_DIR}/models/{SEASON}"
RESULT_DIR = f"{BASE_DIR}/result/Rabi_25_26"
SHAPEFILE_PATH = "/app/district shapefiles/pak_admin2.shp"

FIXED_CLASS_MAPPING = {
    'Bare / Fallow': 0, 'Maize': 1, 'Sugarcane': 2,
    'Vegetation / Other Crops': 3, 'Water': 4, 'Wheat': 5
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
    
    weight_xgb, weight_cat = 0.6, 0.4
    
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
    print(f"\n🚜 Starting Inference for {district_name}...")
    dist_input_dir = os.path.join(INPUT_DIR, district_name.replace(" ", "_"))
    dist_result_dir = os.path.join(RESULT_DIR, district_name.replace(" ", "_"))
    tiles_dir = os.path.join(dist_result_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    output_crop_map = os.path.join(dist_result_dir, f"{district_name}_Rabi_2526_CropMap.tif")
    if os.path.exists(output_crop_map):
        print(f"   ⏩ {district_name} map already exists. Skipping.")
        return

    # Inputs configuration
    month_files = [
        f"{dist_input_dir}/{district_name}_2025-10.tif",
        f"{dist_input_dir}/{district_name}_2025-11.tif",
        f"{dist_input_dir}/{district_name}_2025-12.tif",
        f"{dist_input_dir}/{district_name}_2026-01.tif"
    ]
    # Check if all 4 months exist
    if not all(os.path.exists(f) for f in month_files):
        missing = [f for f in month_files if not os.path.exists(f)]
        print(f"   ⚠️ Skipping {district_name}. Missing Sentinel inputs: {missing}")
        return

    lulc_file = f"{dist_input_dir}/{district_name}_LULC_2526_final.tif"
    if not os.path.exists(lulc_file):
        print(f"   ⚠️ Skipping {district_name}. Missing LULC file.")
        return

    # Load 4 months, proxy the rest from M4 (Persistence)
    src_files = [rasterio.open(f) for f in month_files]
    # Replicate M4 for M5-M8
    full_src_list = src_files + [src_files[3]] * 4
    
    lulc_src = rasterio.open(lulc_file)
    ref_src = src_files[3] # Jan as reference
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

    # Merge tiles
    print(f"🏗️ Merging {district_name} tiles...")
    tile_files = glob.glob(f"{tiles_dir}/tile_*.tif")
    with rasterio.open(output_crop_map, 'w', driver='GTiff', height=height, width=width, count=1, 
                      dtype='uint8', crs=crs, transform=transform, nodata=255, compress='lzw', tiled=True) as dst:
        for tif in tile_files:
            with rasterio.open(tif) as ts:
                coords = os.path.basename(tif).replace('.tif','').split('_')
                dst.write(ts.read(1), 1, window=Window(int(coords[2]), int(coords[1]), ts.width, ts.height))
    print(f"✅ {district_name} COMPLETE.")

if __name__ == "__main__":
    load_models()
    sf = shapefile.Reader(SHAPEFILE_PATH)
    fields = [f[0] for f in sf.fields[1:]]
    adm1_idx = fields.index('adm1_name')
    adm2_idx = fields.index('adm2_name')
    districts = sorted([rec[adm2_idx] for rec in sf.records() if rec[adm1_idx] in ["Punjab", "Islamabad", "F.C.T."]])
    
    for dname in districts:
        process_district(dname)
