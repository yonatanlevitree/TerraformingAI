from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np
from ..state import optimizer, goal_terrain

router = APIRouter()

class WellParams(BaseModel):
    x0: float
    y0: float
    depth: float
    volume: float

@router.get("/")
async def get_wells() -> List[Dict[str, Any]]:
    """Get all placed wells"""
    wells_data = []
    for i, well in enumerate(optimizer.wells):
        wells_data.append({
            "well_id": i,
            "x0": well.x0,
            "y0": well.y0,
            "depth": well.depth,
            "volume": well.volume,
            "monetary_cost": well.monetary_cost(),
            "time_cost": well.time_cost()
        })
    return wells_data

@router.post("/")
async def add_well(params: WellParams) -> Dict[str, Any]:
    """Add a new well"""
    from terrain_optimization import TerrainWell
    well = TerrainWell(params.x0, params.y0, params.depth, params.volume)
    new_terrain = optimizer.apply_well(well)
    
    return {
        "well_id": len(optimizer.wells) - 1,
        "x0": well.x0,
        "y0": well.y0,
        "depth": well.depth,
        "volume": well.volume,
        "monetary_cost": well.monetary_cost(),
        "time_cost": well.time_cost(),
        "new_terrain": new_terrain.tolist()
    }

@router.get("/{well_id}")
async def get_well(well_id: int) -> Dict[str, Any]:
    """Get details for a specific well"""
    if well_id < 0 or well_id >= len(optimizer.wells):
        raise HTTPException(status_code=404, detail="Well not found")
    
    well = optimizer.wells[well_id]
    return {
        "well_id": well_id,
        "x0": well.x0,
        "y0": well.y0,
        "depth": well.depth,
        "volume": well.volume,
        "monetary_cost": well.monetary_cost(),
        "time_cost": well.time_cost()
    }

@router.get("/preview")
async def preview_well(x0: float, y0: float, depth: float, volume: float) -> Dict[str, Any]:
    """Preview the effect of placing a well without committing"""
    if goal_terrain is None:
        raise HTTPException(status_code=400, detail="Goal terrain not set")
    
    from terrain_optimization import TerrainWell
    well = TerrainWell(x0, y0, depth, volume)
    
    # Get current terrain
    if len(optimizer.surface_snapshots) > 0:
        current_terrain = optimizer.surface_snapshots[-1]
    else:
        current_terrain = optimizer.original_surface
    
    # Apply well temporarily
    new_terrain = optimizer.apply_well(well)
    
    # Calculate discrepancy
    discrepancy = goal_terrain - new_terrain
    discrepancy_norm = np.linalg.norm(discrepancy)
    
    return {
        "well": {
            "x0": well.x0,
            "y0": well.y0,
            "depth": well.depth,
            "volume": well.volume,
            "monetary_cost": well.monetary_cost(),
            "time_cost": well.time_cost()
        },
        "new_terrain": new_terrain.tolist(),
        "discrepancy_norm": float(discrepancy_norm)
    } 