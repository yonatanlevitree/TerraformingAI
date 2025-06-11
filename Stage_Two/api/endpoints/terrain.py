from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import numpy as np
from ..state import optimizer, goal_terrain

router = APIRouter()

@router.get("/initial")
async def get_initial_terrain() -> Dict[str, Any]:
    """Get the initial terrain data"""
    return {
        "x_vals": optimizer.x_vals.tolist(),
        "y_vals": optimizer.y_vals.tolist(),
        "heights": optimizer.original_surface.tolist(),
        "grid_size": optimizer.grid_size,
        "z_max": optimizer.z_max
    }

@router.get("/goal")
async def get_goal_terrain() -> Dict[str, Any]:
    """Get the goal terrain data"""
    global goal_terrain
    if goal_terrain is None:
        goal_terrain = optimizer.generate_goal_terrain(method='custom')
    
    return {
        "x_vals": optimizer.x_vals.tolist(),
        "y_vals": optimizer.y_vals.tolist(),
        "heights": goal_terrain.tolist(),
        "grid_size": optimizer.grid_size,
        "z_max": optimizer.z_max
    }

@router.get("/current")
async def get_current_terrain() -> Dict[str, Any]:
    """Get the current terrain state after applied wells"""
    if len(optimizer.surface_snapshots) > 0:
        current_terrain = optimizer.surface_snapshots[-1]
    else:
        current_terrain = optimizer.original_surface
    
    return {
        "x_vals": optimizer.x_vals.tolist(),
        "y_vals": optimizer.y_vals.tolist(),
        "heights": current_terrain.tolist(),
        "grid_size": optimizer.grid_size,
        "z_max": optimizer.z_max
    }

@router.get("/discrepancy")
async def get_terrain_discrepancy() -> Dict[str, Any]:
    """Get the difference between goal and current terrain"""
    if goal_terrain is None:
        raise HTTPException(status_code=400, detail="Goal terrain not set")
    
    if len(optimizer.surface_snapshots) > 0:
        current_terrain = optimizer.surface_snapshots[-1]
    else:
        current_terrain = optimizer.original_surface
    
    discrepancy = goal_terrain - current_terrain
    return {
        "x_vals": optimizer.x_vals.tolist(),
        "y_vals": optimizer.y_vals.tolist(),
        "heights": discrepancy.tolist(),
        "norm": float(np.linalg.norm(discrepancy))
    } 