#!/usr/bin/env python3
"""
Terrain Optimization through Strategic Slurry Injection

This script demonstrates how to transform terrain surfaces through underground slurry injection,
using an optimization approach similar to Stage One but with more realistic 3D geological modeling.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# Add the src directory to the path to import our modules
sys.path.append('src')

# Import terrain optimization module
from terrain_optimization import TerrainOptimizer, TerrainWell

def main():
    """Main function to run the terrain optimization demonstration"""
    print("="*80)
    print("Terrain Optimization through Strategic Slurry Injection")
    print("="*80)
    
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    # 1. Initialize the Terrain Optimizer
    print("\nInitializing Terrain Optimizer...")
    grid_size = 50  # Use a smaller grid for faster computation
    optimizer = TerrainOptimizer(grid_size=grid_size, z_max=60, num_cores=5, seed=42)
    
    # 2. Plot the initial surface
    print("Generating initial terrain from geocore data...")
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(optimizer.x_vals, optimizer.y_vals)
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, optimizer.original_surface, cmap='viridis', alpha=0.7)
    ax.set_title('Initial Terrain Surface')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height (m)')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('initial_terrain.png')
    plt.close()
    print("Initial terrain visualization saved to 'initial_terrain.png'")
    
    # 3. Generate or load goal terrain
    print("\nGenerating goal terrain...")
    stage_one_path = "../../Newest_Attempt/Stage_One/goal_terrain.npy"
    
    # Alternative approach: try loading from file, if that fails, generate a smooth terrain
    # goal_terrain = optimizer.generate_goal_terrain(method='file', file_path=stage_one_path)
    # Use the custom goal terrain generation
    goal_terrain = optimizer.generate_goal_terrain(method='custom')
    
    # Plot the goal terrain
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, goal_terrain, cmap='plasma', alpha=0.7)
    ax.set_title('Goal Terrain Surface')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height (m)')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('goal_terrain.png')
    plt.close()
    print("Goal terrain visualization saved to 'goal_terrain.png'")
    
    # 4. Analyze Initial Discrepancy
    print("\nAnalyzing initial terrain discrepancy...")
    initial_discrepancy = goal_terrain - optimizer.original_surface
    
    # Plot the discrepancy
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, initial_discrepancy, cmap='coolwarm', alpha=0.7)
    ax.set_title('Initial Discrepancy (Goal - Original)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height Difference (m)')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('initial_discrepancy.png')
    plt.close()
    print("Initial discrepancy visualization saved to 'initial_discrepancy.png'")
    
    # Find the worst discrepancy points
    worst_points = optimizer.get_worst_discrepancy_points(optimizer.original_surface, goal_terrain, n_points=5)
    print(f"Top 5 worst discrepancy points: {worst_points}")
    
    # 5. Run Single Well Optimization
    print("\nRunning single well optimization at worst point...")
    x0, y0 = worst_points[0]
    
    # Optimize well parameters for this location
    print(f"Optimizing well at ({x0:.1f}, {y0:.1f})...")
    start_time = time.perf_counter()
    depth, volume = optimizer.optimize_well_params(x0, y0, optimizer.original_surface, goal_terrain)
    end_time = time.perf_counter()
    
    print(f"Optimization took {end_time - start_time:.2f} seconds")
    print(f"Optimized parameters: Depth = {depth:.2f}m, Volume = {volume:.2f}m³")
    
    # Create and apply the well
    well = TerrainWell(x0, y0, depth, volume)
    new_terrain = optimizer.apply_well(well)
    
    # Calculate costs
    monetary_cost = well.monetary_cost()
    time_cost = well.time_cost()
    print(f"Well costs: ${monetary_cost:.2f}, Time: {time_cost:.2f} hours")
    
    # Plot the result
    plt.figure(figsize=(15, 5))
    
    # Original terrain
    ax1 = plt.subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, optimizer.original_surface, cmap='viridis', alpha=0.7)
    ax1.set_title('Original Terrain')
    ax1.set_zlim(0, max(np.max(goal_terrain), np.max(optimizer.original_surface)) + 5)
    
    # New terrain after one well
    ax2 = plt.subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, new_terrain, cmap='inferno', alpha=0.7)
    ax2.scatter([x0], [y0], [0], marker='^', s=100, color='red')
    ax2.set_title('Terrain After One Well')
    ax2.set_zlim(0, max(np.max(goal_terrain), np.max(optimizer.original_surface)) + 5)
    
    # Remaining discrepancy
    discrepancy = goal_terrain - new_terrain
    ax3 = plt.subplot(1, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, discrepancy, cmap='coolwarm', alpha=0.7)
    ax3.set_title('Remaining Discrepancy')
    
    plt.tight_layout()
    plt.savefig('single_well_result.png')
    plt.close()
    print("Single well optimization result saved to 'single_well_result.png'")
    
    # Calculate improvement
    initial_norm = np.linalg.norm(initial_discrepancy)
    current_norm = np.linalg.norm(discrepancy)
    improvement = 100 * (initial_norm - current_norm) / initial_norm
    print(f"Discrepancy improvement: {improvement:.2f}%")
    
    # 6. Run Full Iterative Optimization
    print("\nRunning full iterative optimization...")
    
    # Reset the optimizer to start fresh
    optimizer = TerrainOptimizer(grid_size=grid_size, z_max=60, num_cores=5, seed=42)
    
    # Set optimization parameters
    max_iterations = 10
    max_monetary_budget = 100000  # Maximum budget in monetary units
    max_time_budget = 10000      # Maximum budget in time units
    
    print("Starting optimization process...")
    start_time = time.perf_counter()
    
    # Run the optimization
    final_terrain, wells, monetary_cost, time_cost, terrain_history = optimizer.iterative_optimization(
        goal_terrain, 
        max_iterations=max_iterations,
        max_monetary_budget=max_monetary_budget,
        max_time_budget=max_time_budget
    )
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
    print(f"Final monetary cost: ${monetary_cost:.2f}")
    print(f"Final time cost: {time_cost:.2f} hours")
    print(f"Number of wells placed: {len(wells)}")
    
    # 7. Visualize Each Step of the Optimization Process
    print("\nGenerating step-by-step visualizations...")
    
    # Create a directory for step visualizations if it doesn't exist
    step_dir = "optimization_steps"
    if not os.path.exists(step_dir):
        os.makedirs(step_dir)
    
    # Create X, Y coordinate grids for plotting
    X, Y = np.meshgrid(optimizer.x_vals, optimizer.y_vals)
    
    # Plot each step of the optimization process
    for i, step_terrain in enumerate(terrain_history):
        plt.figure(figsize=(15, 10))
        
        # Create a 2x2 grid of subplots
        ax1 = plt.subplot(2, 2, 1, projection='3d')
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        ax3 = plt.subplot(2, 2, 3, projection='3d')
        ax4 = plt.subplot(2, 2, 4, projection='3d')
        
        # Plot initial terrain
        surf1 = ax1.plot_surface(X, Y, optimizer.original_surface, cmap='viridis', alpha=0.7)
        ax1.set_title('Initial Terrain')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Height (m)')
        
        # Plot goal terrain
        surf2 = ax2.plot_surface(X, Y, goal_terrain, cmap='plasma', alpha=0.7)
        ax2.set_title('Goal Terrain')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Height (m)')
        
        # Plot current terrain at this step
        surf3 = ax3.plot_surface(X, Y, step_terrain, cmap='inferno', alpha=0.7)
        step_title = 'Initial Terrain' if i == 0 else f'After {i} Well(s)'
        ax3.set_title(f'Current Terrain: {step_title}')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Height (m)')
        
        # Plot current discrepancy
        discrepancy = goal_terrain - step_terrain
        surf4 = ax4.plot_surface(X, Y, discrepancy, cmap='coolwarm', alpha=0.7)
        ax4.set_title(f'Discrepancy (Goal - Current)')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Height Difference (m)')
        
        # Add well locations to the current terrain plot
        if i > 0:
            for j, well in enumerate(wells[:i]):
                ax3.scatter([well.x0], [well.y0], [step_terrain[np.abs(optimizer.y_vals - well.y0).argmin(), 
                                                             np.abs(optimizer.x_vals - well.x0).argmin()]], 
                           marker='^', s=100, color='red')
        
        # Set consistent z-axis limits for better comparison
        z_min = min(np.min(optimizer.original_surface), np.min(goal_terrain), np.min(step_terrain))
        z_max = max(np.max(optimizer.original_surface), np.max(goal_terrain), np.max(step_terrain))
        ax1.set_zlim(z_min, z_max)
        ax2.set_zlim(z_min, z_max)
        ax3.set_zlim(z_min, z_max)
        
        # Add a colorbar for the discrepancy plot
        plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
        
        # Add a title for the entire figure
        plt.suptitle(f'Terrain Optimization Process - Step {i}', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
        plt.savefig(f'{step_dir}/step_{i:02d}.png')
        plt.close()
        
        print(f"  Saved visualization for step {i}")
    
    print(f"Step-by-step visualizations saved to '{step_dir}/' directory")
    
    # 8. Visualize Final Optimization Results
    print("\nGenerating final optimization result visualization...")
    
    # Save the optimization results plot
    optimizer.plot_optimization_results(goal_terrain, final_terrain)
    plt.savefig('optimization_results.png')
    plt.close()
    print("Optimization results visualization saved to 'optimization_results.png'")
    
    # 9. Well Characteristics
    print("\nWell characteristics:")
    print("-" * 70)
    print(f"{'Well #':^6} | {'X (m)':^10} | {'Y (m)':^10} | {'Depth (m)':^10} | {'Volume (m³)':^12} | {'Cost ($)':^10} | {'Time (h)':^8}")
    print("-" * 70)
    
    for i, well in enumerate(wells):
        print(f"{i+1:^6} | {well.x0:^10.2f} | {well.y0:^10.2f} | {well.depth:^10.2f} | {well.volume:^12.2f} | {well.monetary_cost():^10.2f} | {well.time_cost():^8.2f}")
    
    print("-" * 70)
    print(f"{'Total':^6} | {'':<10} | {'':<10} | {'':<10} | {'':<12} | {monetary_cost:^10.2f} | {time_cost:^8.2f}")
    
    # 10. Create 3D Animation (if running in interactive mode)
    try:
        print("\nGenerating 3D animation...")
        fig = optimizer.create_3D_animation()
        fig.write_html('terrain_animation.html')
        print("3D animation saved to 'terrain_animation.html'")
    except Exception as e:
        print(f"Could not generate 3D animation: {e}")
    
    print("\nTerrain optimization process completed successfully!")

if __name__ == "__main__":
    main() 