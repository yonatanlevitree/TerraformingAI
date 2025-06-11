#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import numpy
import json
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, gaussian_filter, minimum_filter
from scipy.signal import find_peaks
from scipy.ndimage.measurements import label
import os
from copy import deepcopy
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from system import *


# In[2]:


## Cell block for injection logic
# Optimize one injection
def optimize_well_parameters(terrain, wells, depth_bounds, volume_bounds):
    initial_params = np.array([param for well in wells for param in (well.depth, well.volume)])
    bounds = [(depth_bounds[0], depth_bounds[1]), (volume_bounds[0], volume_bounds[1])] * len(wells)

    def objective(params):
        for i, well in enumerate(wells):
            well.depth, well.volume = params[i*2], params[i*2+1]
        modified_terrain = terrain.apply_wells(wells)
        discrepancy = modified_terrain - terrain.goal_terrain

        # Define weights for the loss function
        overshoot_weight = 20  # Higher weight for overshooting
        undershoot_weight = 1  # Lower weight for undershooting

        # Compute the asymmetric loss
        loss = torch.where(discrepancy > 0,
                           overshoot_weight * discrepancy ** 2,
                           undershoot_weight * discrepancy ** 2)
        return torch.mean(loss).item()

    result = minimize(objective, initial_params, method='Nelder-Mead', bounds=bounds, options={'maxiter': 10000, 'disp': True})
    for i, well in enumerate(wells):
        well.depth, well.volume = result.x[i*2], result.x[i*2+1]

# For now we start injection with where max discrepany is found
def points_of_maximum_discrepancy(initial_terrain, goal_terrain, min_distance=5, num_peaks=4):
    discrepancy = (goal_terrain - initial_terrain).cpu().numpy()  # Convert tensor to numpy array
    local_max = maximum_filter(discrepancy, size=min_distance) == discrepancy

    local_max *= discrepancy > minimum_filter(discrepancy, size=min_distance)
    max_coords = np.array(np.where(local_max)).T

    if len(max_coords) > 0:
        max_values = discrepancy[tuple(max_coords.T)]
        sorted_indices = np.argsort(-max_values)
        sorted_max_coords = max_coords[sorted_indices][:num_peaks]
    else:
        sorted_max_coords = np.array([])  # No maxima found

    return sorted_max_coords

def plot_final_terrains(original_terrain, optimized_terrain, goal_terrain, total_size, center_terrain_size=None):
    """
    Plots the final comparison of terrains in 3D.

    :param (new) center_terrain_size: The dimension of the central subregion to visualize (e.g., 100).
                               If None, the full grid [0..total_size-1, 0..total_size-1] is shown.
    """
    # Determine if we are using a subregion or the full terrain when showing 3D visulization
    if center_terrain_size is not None and 0 < center_terrain_size <= total_size:
        # Calculate the start and end indices for the subregion
        start_idx = total_size // 2 - center_terrain_size // 2
        end_idx = start_idx + center_terrain_size

        # Sub-slice each terrain
        orig_sub = original_terrain[start_idx:end_idx, start_idx:end_idx]
        opt_sub = optimized_terrain[start_idx:end_idx, start_idx:end_idx]
        goal_sub = goal_terrain[start_idx:end_idx, start_idx:end_idx]

        # Here we create coordinate arrays for the subregion
        x = y = np.linspace(start_idx, end_idx - 1, center_terrain_size)

        # And plot the subregion only
        terrain_to_plot = {
            "original": orig_sub,
            "optimized": opt_sub,
            "goal": goal_sub,
            "size": center_terrain_size
        }
    else:
        # Show the full terrain
        x = y = np.linspace(0, total_size - 1, total_size)
        terrain_to_plot = {
            "original": original_terrain,
            "optimized": optimized_terrain,
            "goal": goal_terrain,
            "size": total_size
        }

    X, Y = np.meshgrid(x, y)

    # Now plot the subregion or full region in 3D depending on parameter center_terrain_size 
    fig = go.Figure()

    # Original Terrain - fully opaque, colored blue
    fig.add_trace(go.Surface(
        z=terrain_to_plot["original"].cpu().numpy(),
        x=X, y=Y,
        colorscale='Blues', 
        opacity=1.0, 
        name='Original Terrain (Blue)'
    ))

    # Optimized Terrain - 50% transparency, colored red
    fig.add_trace(go.Surface(
        z=terrain_to_plot["optimized"].cpu().numpy(),
        x=X, y=Y,
        colorscale='Reds', 
        opacity=0.5, 
        name='Optimized Terrain (Red)'
    ))

    # Goal Terrain - 50% transparency, colored green
    fig.add_trace(go.Surface(
        z=terrain_to_plot["goal"].cpu().numpy(),
        x=X, y=Y,
        colorscale='Greens',
        opacity=0.5, 
        name='Goal Terrain (Green)'
    ))

    # Update plot layout
    fig.update_layout(
        title='Comparison of Terrains',
        scene=dict(
            xaxis_title='X coordinate',
            yaxis_title='Y coordinate',
            zaxis_title='Elevation'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()


# In[89]:


# Experimenting with monetary and time constraints added
def main():
    terrain_size = 400
    noise_level = 0.5
    smoothness = 10
    max_iterations = 100  # Increase for better result
    depth_bounds = (5, 30)
    volume_bounds = (10, 500)
    epsilon = 0.001  # Adjusted epsilon for different local fidelity
    window_size = 1  # Adjusted window size for sliding window

    # Define total cost constraints
    # Decrease these values if you want to see the algorithm running into an early stop because of budget constraints
    total_monetary_cost_limit = 50000000  # Example value for p, set for 50 million for now
    total_time_cost_limit = 500000        # Example value for t, set for 500k for now

    terrain = Terrain(terrain_size, noise_level, smoothness, epsilon=epsilon, device='cpu', regenerate=True)

    init_ter = deepcopy(terrain.initial_terrain)
    all_wells = []

    # Initialize cumulative costs
    cumulative_monetary_cost = 0.0
    cumulative_time_cost = 0.0

    previous_max_norm = float('inf')
    previous_terrain = None
    iteration_completed = 0  # To track how many iterations were completed successfully

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        max_points = points_of_maximum_discrepancy(terrain.initial_terrain, terrain.goal_terrain)
        if len(max_points) == 0:
            print("No more discrepancy points found. Optimization complete.")
            break

        # Initialize wells for this iteration
        wells = [
            Well(x0, y0, np.random.uniform(depth_bounds[0], depth_bounds[1]),
                 np.random.uniform(volume_bounds[0], volume_bounds[1]))
            for x0, y0 in max_points
        ]

        # Estimate the cost of adding these wells
        estimated_monetary_cost = sum([well.monetary_cost() for well in wells])
        estimated_time_cost = sum([well.time_cost() for well in wells])

        print(f"Estimated Monetary Cost for Iteration {iteration + 1}: {estimated_monetary_cost:.2f}")
        print(f"Estimated Time Cost for Iteration {iteration + 1}: {estimated_time_cost:.2f}")

        # Check if adding these wells would exceed the total cost constraints
        if (cumulative_monetary_cost + estimated_monetary_cost > total_monetary_cost_limit) or \
           (cumulative_time_cost + estimated_time_cost > total_time_cost_limit):
            print("Adding these wells would exceed the total cost constraints.")
            print("Skipping this iteration and stopping optimization.")
            break

        # Optimize well parameters
        optimize_well_parameters(terrain, wells, depth_bounds, volume_bounds)
        all_wells.extend(wells)

        # Update cumulative costs
        actual_monetary_cost = sum([well.monetary_cost() for well in wells])
        actual_time_cost = sum([well.time_cost() for well in wells])
        cumulative_monetary_cost += actual_monetary_cost
        cumulative_time_cost += actual_time_cost

        print(f"Actual Monetary Cost added in Iteration {iteration + 1}: {actual_monetary_cost:.2f}")
        print(f"Actual Time Cost added in Iteration {iteration + 1}: {actual_time_cost:.2f}")
        print(f"Cumulative Monetary Cost: {cumulative_monetary_cost:.2f} / {total_monetary_cost_limit}")
        print(f"Cumulative Time Cost: {cumulative_time_cost:.2f} / {total_time_cost_limit}")

        # Save the current terrain before applying new wells
        previous_terrain = deepcopy(terrain.initial_terrain)

        # Apply the optimized wells
        terrain.initial_terrain = terrain.apply_wells(wells)

        # Check the local fidelity constraint
        passed, max_norm = terrain.check_local_fidelity(terrain.initial_terrain, window_size=window_size)
        # Log the current state
        print(f"Max local norm = {max_norm:.4f}, Passed fidelity: {passed}")

        # Early Stop Condition: If max_norm has increased compared to the previous iteration
        if max_norm > previous_max_norm:
            print(f"Early stopping at iteration {iteration + 1} due to increasing max local norm.")
            if previous_terrain is not None:
                terrain.initial_terrain = previous_terrain
                # Remove the last set of wells as they caused the increase
                all_wells = all_wells[:-len(wells)]
                # Adjust cumulative costs
                cumulative_monetary_cost -= actual_monetary_cost
                cumulative_time_cost -= actual_time_cost
                iteration_completed = iteration
            break
        else:
            previous_max_norm = max_norm
            iteration_completed = iteration + 1

        # If the fidelity constraint is met, stop early
        if passed:
            print(f"Fidelity constraint met at iteration {iteration + 1}!")
            break

        # Plot the progress after each iteration
        #terrain.plot_terrains(init_ter, terrain.goal_terrain, terrain.initial_terrain, all_wells, max_points,
        #                      f"Iteration {iteration + 1}")

    # Final Logging
    if iteration_completed < max_iterations:
        print(f"\nOptimization stopped early at iteration {iteration_completed}.")
    else:
        print("\nOptimization completed all iterations.")
    """
        # Plot the final terrains
        # plot_final_terrains(init_ter, terrain.initial_terrain, terrain.goal_terrain, terrain_size)
        plot_final_terrains(
        original_terrain=init_ter,
        optimized_terrain=terrain.initial_terrain,
        goal_terrain=terrain.goal_terrain,
        total_size=terrain_size,
        # MODIFY THIS PARAMETER TO DETERMINE HOW MUCH OF THE FULL TERRAIN YOU WANT TO VISUALIZE
        center_terrain_size=100  # or None if you want the full terrain
    )
    """
    print(json.dumps({
        "terrain_summary": {
            "initial_mean": init_ter.mean().item(),
            "goal_mean": terrain.goal_terrain.mean().item(),
            "final_mean": terrain.initial_terrain.mean().item(),
            "iterations": iteration_completed,
            "cost_monetary": cumulative_monetary_cost,
            "cost_time": cumulative_time_cost
        }
    }))

if __name__ == "__main__":
    main()