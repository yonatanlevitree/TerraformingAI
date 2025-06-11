from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np
from ..state import optimizer, goal_terrain

router = APIRouter()

class OptimizationParams(BaseModel):
    max_iterations: int = 10
    max_budget: float = 100000
    max_time: float = 10000

@router.post("/single")
async def optimize_single_well(x: float, y: float) -> Dict[str, Any]:
    """Optimize a single well at the given location"""
    if goal_terrain is None:
        raise HTTPException(status_code=400, detail="Goal terrain not set")
    
    # Get current terrain
    if len(optimizer.surface_snapshots) > 0:
        current_terrain = optimizer.surface_snapshots[-1]
    else:
        current_terrain = optimizer.original_surface
    
    # Optimize parameters for this location
    depth, volume = optimizer.optimize_well_params(x, y, current_terrain, goal_terrain)
    
    # Create and apply the well
    from terrain_optimization import TerrainWell
    well = TerrainWell(x, y, depth, volume)
    new_terrain = optimizer.apply_well(well)
    
    return {
        "well_id": len(optimizer.wells) - 1,
        "x0": well.x0,
        "y0": well.y0,
        "depth": well.depth,
        "volume": well.volume,
        "monetary_cost": well.monetary_cost(),
        "time_cost": well.time_cost(),
        "new_terrain": new_terrain.tolist(),
        "discrepancy_norm": float(np.linalg.norm(goal_terrain - new_terrain))
    }

@router.post("/terrain")
async def optimize_terrain(params: OptimizationParams) -> Dict[str, Any]:
    """Run full terrain optimization with constraints"""
    if goal_terrain is None:
        raise HTTPException(status_code=400, detail="Goal terrain not set")
    
    # Run the optimization
    final_terrain, wells, monetary_cost, time_cost, terrain_history = optimizer.iterative_optimization(
        goal_terrain,
        max_iterations=params.max_iterations,
        max_monetary_budget=params.max_budget,
        max_time_budget=params.max_time
    )
    
    return {
        "status": "completed",
        "iterations": len(wells),
        "monetary_cost": float(monetary_cost),
        "time_cost": float(time_cost),
        "well_count": len(wells),
        "final_discrepancy_norm": float(np.linalg.norm(goal_terrain - final_terrain))
    }

@router.get("/history")
async def get_optimization_history() -> Dict[str, Any]:
    """Get the step-by-step history of the optimization process"""
    if goal_terrain is None:
        raise HTTPException(status_code=400, detail="No optimization has been run yet")
    
    terrain_states = []
    
    # Add initial state
    terrain_states.append({
        "stage_name": "Initial Terrain",
        "heights": optimizer.original_surface.tolist(),
        "wells": [],
        "discrepancy_norm": float(np.linalg.norm(goal_terrain - optimizer.original_surface)),
        "total_monetary_cost": 0.0,
        "total_time_cost": 0.0
    })
    
    # Add states for each well placement
    cumulative_cost = 0.0
    cumulative_time = 0.0
    
    for i, terrain in enumerate(optimizer.surface_snapshots[1:], 1):
        well = optimizer.wells[i-1]
        cumulative_cost += well.monetary_cost()
        cumulative_time += well.time_cost()
        
        discrepancy = goal_terrain - terrain
        discrepancy_norm = np.linalg.norm(discrepancy)
        
        wells_so_far = []
        for j, w in enumerate(optimizer.wells[:i]):
            wells_so_far.append({
                "well_id": j,
                "x0": w.x0,
                "y0": w.y0,
                "depth": w.depth,
                "volume": w.volume
            })
        
        terrain_states.append({
            "stage_name": f"After {i} Well(s)",
            "heights": terrain.tolist(),
            "wells": wells_so_far,
            "discrepancy_norm": float(discrepancy_norm),
            "total_monetary_cost": float(cumulative_cost),
            "total_time_cost": float(cumulative_time)
        })
    
    return {
        "iteration": len(optimizer.wells),
        "terrain_states": terrain_states,
        "goal_terrain": {
            "heights": goal_terrain.tolist()
        }
    } 