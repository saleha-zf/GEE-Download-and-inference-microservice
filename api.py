import os
import json
import time
import uuid
import threading
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import ee

# Import existing workflow scripts
import gee_punjab_lulc_downloader_v2 as lulc_downloader
import gee_bulk_s2_rabi_downloader as s2_downloader
import gee_bulk_rabi_inference as rabi_inference

app = FastAPI(
    title="Agriverse GEE & Crop Inference Microservice",
    description="Full pipeline: Download LULC, Sentinel-2, and run Tiled Ensemble Inference for Rabi season.",
    version="1.1.0"
)

# Shared state for task monitoring
tasks = {}

class DownloadRequest(BaseModel):
    districts: List[str]
    season: str = "Rabi_25_26"
    months: Optional[List[str]] = None

class InferenceRequest(BaseModel):
    districts: List[str]
    season: str = "rabi"

@app.get("/")
def read_root():
    return {
        "message": "Agriverse GEE & Inference Microservice Active",
        "endpoints": {
            "GET /districts": "List all districts in shapefile and check download status",
            "POST /download/lulc": "Download Dynamic World LULC probability stacks",
            "POST /download/sentinel": "Download monthly Sentinel-2 indices",
            "POST /inference/rabi": "Run Tiled Ensemble Inference for Rabi season",
            "GET /tasks": "List all background tasks",
            "GET /task/{id}": "Check status of a specific task"
        }
    }

@app.get("/districts")
def list_districts():
    """Lists all available districts and their download status."""
    try:
        # 1. Get all districts from shapefile
        all_recs = lulc_downloader.get_punjab_records()
        district_names = [d['name'] for d in all_recs]
        
        status_report = {}
        
        # 2. Check output directory for existing files
        output_root = "/app/output/Rabi_25_26"
        if not os.path.exists(output_root):
            # If nothing downloaded yet, return names with empty status
            return {d: {"status": "NO_DATA"} for d in district_names}

        for dname in district_names:
            safe_name = dname.replace(" ", "_")
            dist_dir = os.path.join(output_root, safe_name)
            
            d_status = {"status": "NO_DATA", "files": {}}
            
            if os.path.exists(dist_dir):
                files = [f for f in os.listdir(dist_dir) if f.endswith(".tif")]
                if files:
                    d_status["status"] = "PARTIAL_DATA"
                    for f in files:
                        fpath = os.path.join(dist_dir, f)
                        mtime = os.path.getmtime(fpath)
                        timestamp = time.ctime(mtime)
                        d_status["files"][f] = timestamp
                    
                    # Check if LULC and at least one S2 exist for "COMPLETED" label
                    has_lulc = any("LULC" in f for f in files)
                    has_s2 = any("-" in f or "2025" in f or "2026" in f for f in files if "LULC" not in f)
                    if has_lulc and has_s2:
                        d_status["status"] = "DOWNLOADED"
            
            status_report[dname] = d_status
            
        return {
            "total_districts": len(district_names),
            "districts": status_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
def list_tasks():
    return tasks

@app.get("/task/{task_id}")
def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

# --- BACKGROUND WORKERS ---

def run_lulc_worker(task_id: str, districts: List[str]):
    tasks[task_id]["status"] = "RUNNING"
    try:
        all_recs = lulc_downloader.get_punjab_records()
        
        if districts == ["ALL"]:
            target_recs = all_recs
        else:
            target_recs = [d for d in all_recs if d['name'] in districts]
        
        if not target_recs:
            tasks[task_id]["status"] = "FAILED"
            tasks[task_id]["error"] = "No matching districts found."
            return

        for d in target_recs:
            tasks[task_id]["current_district"] = d['name']
            geom = lulc_downloader.shape_to_ee_geometry(d['shape'])
            lulc_downloader.download_lulc_v3(d['name'], geom)
            
        tasks[task_id]["status"] = "COMPLETED"
        tasks[task_id]["current_district"] = None
    except Exception as e:
        tasks[task_id]["status"] = "FAILED"
        tasks[task_id]["error"] = str(e)

def run_s2_worker(task_id: str, districts: List[str], months: Optional[List[str]]):
    tasks[task_id]["status"] = "RUNNING"
    # Overwrite the global MONTHS in the downloader module ONLY if provided
    if months:
        s2_downloader.MONTHS = months
    
    # Expose the months being used for logging
    tasks[task_id]["target_months"] = s2_downloader.MONTHS
    try:
        all_recs = s2_downloader.get_punjab_records()
        
        if districts == ["ALL"]:
            target_recs = all_recs
        else:
            target_recs = [d for d in all_recs if d['name'] in districts]
        
        if not target_recs:
            tasks[task_id]["status"] = "FAILED"
            tasks[task_id]["error"] = "No matching districts found."
            return

        for d in target_recs:
            tasks[task_id]["current_district"] = d['name']
            geom = s2_downloader.shape_to_ee_geometry(d['shape'])
            s2_downloader.download_district_s2(d['name'], geom)
            
        tasks[task_id]["status"] = "COMPLETED"
        tasks[task_id]["current_district"] = None
    except Exception as e:
        tasks[task_id]["status"] = "FAILED"
        tasks[task_id]["error"] = str(e)

def run_inference_worker(task_id: str, districts: List[str]):
    tasks[task_id]["status"] = "RUNNING"
    try:
        # Lazy load models only when needed to save base memory
        rabi_inference.load_models()
        
        if districts == ["ALL"]:
            # Get all names directly from shapefile logic in inference script
            import shapefile
            sf = shapefile.Reader(rabi_inference.SHAPEFILE_PATH)
            fields = [f[0] for f in sf.fields[1:]]
            adm1_idx = fields.index('adm1_name')
            adm2_idx = fields.index('adm2_name')
            target_districts = sorted([rec[adm2_idx] for rec in sf.records() if rec[adm1_idx] in ["Punjab", "Islamabad", "F.C.T."]])
        else:
            target_districts = districts
        
        for dname in target_districts:
            tasks[task_id]["current_district"] = dname
            rabi_inference.process_district(dname)
            
        tasks[task_id]["status"] = "COMPLETED"
        tasks[task_id]["current_district"] = None
    except Exception as e:
        tasks[task_id]["status"] = "FAILED"
        tasks[task_id]["error"] = str(e)

# --- API ENDPOINTS ---

@app.post("/download/lulc")
def start_lulc(request: DownloadRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "type": "LULC_DOWNLOAD",
        "status": "QUEUED",
        "districts": request.districts,
        "timestamp": time.ctime()
    }
    background_tasks.add_task(run_lulc_worker, task_id, request.districts)
    return {"task_id": task_id, "status": "QUEUED"}

@app.post("/download/sentinel")
def start_s2(request: DownloadRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "type": "SENTINEL_DOWNLOAD",
        "status": "QUEUED",
        "districts": request.districts,
        "months": request.months,
        "timestamp": time.ctime()
    }
    background_tasks.add_task(run_s2_worker, task_id, request.districts, request.months)
    return {"task_id": task_id, "status": "QUEUED"}

@app.post("/inference/rabi")
def start_inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "type": "RABI_INFERENCE",
        "status": "QUEUED",
        "districts": request.districts,
        "timestamp": time.ctime()
    }
    background_tasks.add_task(run_inference_worker, task_id, request.districts)
    return {"task_id": task_id, "status": "QUEUED"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
