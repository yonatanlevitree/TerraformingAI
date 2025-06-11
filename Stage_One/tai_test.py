import numpy as np
import torch
from copy import deepcopy
import json
from system import *

from scipy.optimize import minimize
from scipy.ndimage import maximum_filter, minimum_filter

def optimize_well_parameters(terrain, wells, depth_bounds, volume_bounds):
    initial_params = np.array([param for well in wells for param in (well.depth, well.volume)])
    bounds = [(depth_bounds[0], depth_bounds[1]), (volume_bounds[0], volume_bounds[1])] * len(wells)

    def objective(params):
        for i, well in enumerate(wells):
            well.depth, well.volume = params[i*2], params[i*2+1]
        modified_terrain = terrain.apply_wells(wells)
        discrepancy = modified_terrain - terrain.goal_terrain
        overshoot_weight = 20
        undershoot_weight = 1
        loss = torch.where(discrepancy > 0, overshoot_weight * discrepancy**2, undershoot_weight * discrepancy**2)
        return torch.mean(loss).item()

    result = minimize(objective, initial_params, method='Nelder-Mead', bounds=bounds, options={'maxiter': 1000, 'disp': False})
    for i, well in enumerate(wells):
        well.depth, well.volume = result.x[i*2], result.x[i*2+1]

def points_of_maximum_discrepancy(initial_terrain, goal_terrain, min_distance=5, num_peaks=4):
    discrepancy = (goal_terrain - initial_terrain).cpu().numpy()
    local_max = maximum_filter(discrepancy, size=min_distance) == discrepancy
    local_max *= discrepancy > minimum_filter(discrepancy, size=min_distance)
    max_coords = np.array(np.where(local_max)).T
    if len(max_coords) > 0:
        max_values = discrepancy[tuple(max_coords.T)]
        sorted_indices = np.argsort(-max_values)
        return max_coords[sorted_indices][:num_peaks]
    else:
        return np.array([])

def main(terrain_size=100, noise_level=0.5, smoothness=10, depth_bounds=(5, 30), volume_bounds=(10, 500)):
    terrain = Terrain(terrain_size, noise_level, smoothness, epsilon=0.001, device='cpu', regenerate=True)
    init_ter = deepcopy(terrain.initial_terrain)

    max_points = points_of_maximum_discrepancy(terrain.initial_terrain, terrain.goal_terrain)
    if len(max_points) == 0:
        print(json.dumps({"status": "no_peaks_found"}))
        return

    wells = [
        Well(x0, y0, np.random.uniform(*depth_bounds), np.random.uniform(*volume_bounds))
        for x0, y0 in max_points
    ]

    optimize_well_parameters(terrain, wells, depth_bounds, volume_bounds)
    optimized_terrain = terrain.apply_wells(wells)

    print(json.dumps({
        "status": "success",
        "terrain_summary": {
            "initial_mean": float(init_ter.mean().item()),
            "goal_mean": float(terrain.goal_terrain.mean().item()),
            "final_mean": float(optimized_terrain.mean().item()),
            "num_wells": int(len(wells)),
        },
        "well_parameters": [
            {
                "x": int(w.x0),
                "y": int(w.y0),
                "depth": float(w.depth),
                "volume": float(w.volume)
            }
            for w in wells
        ]
    }))

if __name__ == "__main__":
    main()
