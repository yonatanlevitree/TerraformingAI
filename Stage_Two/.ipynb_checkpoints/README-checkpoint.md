# Terrain Optimization via Strategic Slurry Injection

This project extends the terrain transformation capabilities from Stage One to a more sophisticated 3D geological model in Stage Two. It demonstrates how to optimize terrain surfaces through strategic underground slurry injection.

## Project Structure

- `src/` - Core implementation files
  - `geocore_utils.py` - Generation of random geocores with layered density profiles
  - `interpolation_utils.py` - 3D interpolation of density and pressure fields
  - `injection_utils.py` - Slurry injection and pressure propagation simulation
  - `plotting_utils.py` - Visualization utilities
  - `terrain_optimization.py` - Main optimization logic
  - `run_terrain_optimization.py` - Python script to demonstrate the terrain optimization process

## Features

1. **3D Geological Modeling**
   - Generate realistic geocores with layered density profiles
   - Interpolate 3D density and pressure fields
   - Extract terrain surfaces from 3D pressure data

2. **Terrain Transformation**
   - Define initial and goal terrains
   - Identify optimal locations for well placement
   - Optimize well parameters (depth, volume) to achieve desired terrain changes

3. **Optimization with Constraints**
   - Monetary and time budget constraints
   - Physics-based slurry injection and pressure propagation
   - Asymmetric loss function to penalize overshooting and undershooting
   - Multi-starting point optimization to avoid local minima
   - Well spacing controls to prevent clustering

4. **Visualization**
   - 3D terrain surfaces
   - Underground pressure fields
   - Well locations and characteristics
   - Animated transformation process
   - Step-by-step visualization of the optimization process

## How It Works

### Terrain Generation
1. Random geocores are generated with varying density profiles
2. These geocores are interpolated to create a continuous 3D density field
3. The density field is converted to a pressure field based on gravitational effects
4. The initial terrain surface is extracted based on a pressure threshold

### Optimization Process
1. The goal terrain is defined (either loaded from Stage One or generated)
2. The system identifies points with the largest discrepancy between current and goal terrains
3. For each point:
   - Well parameters (depth, volume) are optimized using the Nelder-Mead algorithm
   - Multiple starting points are used to avoid local minima
   - The slurry is injected at the optimal location and depth
   - Pressure field and terrain surface are updated
   - Costs are calculated and checked against budget constraints
4. The process repeats until the goal terrain is achieved or constraints are exceeded

### Slurry Injection Model
The injection model simulates how slurry spreads through the subsurface using:
- Pressure-guided flow
- Breadth-first search for spreading patterns
- Physics-based pressure updates based on slurry distribution

## Usage

### Running the Demonstration Script
To run the demonstration script:
1. Ensure all dependencies are installed
2. Navigate to the Stage_Two directory
3. Run the script:
   ```
   python src/run_terrain_optimization.py
   ```
4. View the generated visualizations:
   - `initial_terrain.png` - The starting terrain surface
   - `goal_terrain.png` - The target terrain we want to achieve
   - `initial_discrepancy.png` - The difference between goal and initial terrains
   - `single_well_result.png` - Result after optimizing and applying a single well
   - `optimization_results.png` - Final results after full optimization
   - `optimization_steps/` - Directory containing step-by-step visualizations of the process
   - `terrain_animation.html` - Interactive 3D animation of the optimization process

### Using the TerrainOptimizer
You can also use the `TerrainOptimizer` class in your own code:

```python
from src.terrain_optimization import TerrainOptimizer

# Initialize the optimizer
optimizer = TerrainOptimizer(grid_size=100, z_max=60, num_cores=5, seed=42)

# Generate or load goal terrain
goal_terrain = optimizer.generate_goal_terrain(method='smooth')

# Run optimization
final_terrain, wells, monetary_cost, time_cost, terrain_history = optimizer.iterative_optimization(
    goal_terrain, 
    max_iterations=10,
    max_monetary_budget=10000,
    max_time_budget=1000
)

# Visualize results
optimizer.plot_optimization_results(goal_terrain, final_terrain)
```

## Optimization Parameters

The terrain optimization process uses several key parameters that can be adjusted:

- `grid_size`: Resolution of the terrain (default: 50)
- `z_max`: Maximum depth for the simulation (default: 60m)
- `num_cores`: Number of geocores to generate (default: 5)
- `max_iterations`: Maximum number of wells to place (default: 10)
- `max_monetary_budget`: Maximum money to spend (default: 100,000)
- `max_time_budget`: Maximum time budget in hours (default: 10,000)
- `min_well_spacing`: Minimum distance between wells (default: 15m)

The well parameters are optimized within these bounds:
- Depth: 5-50 meters
- Volume: 10-500 cubic meters

## Interpreting Results

The optimization process generates several visualizations to help interpret the results:

### Single Well Analysis
The `single_well_result.png` shows the effect of a single optimized well and helps to understand:
- How slurry injection affects the local terrain
- The relationship between injection depth, volume, and surface change
- Remaining discrepancies that need to be addressed

### Step-by-Step Visualizations
The `optimization_steps/` directory contains a sequence of images showing:
- Initial terrain
- Goal terrain
- Current terrain after each well
- Current discrepancy map
- Well locations

These visualizations help understand the progressive transformation of the terrain and the strategic placement of wells.

### Final Results
The `optimization_results.png` provides a comprehensive view of:
- Initial vs. final terrain comparison
- Well locations and characteristics
- Remaining discrepancies
- Total cost (monetary and time)

### Performance Metrics
The optimization process tracks these metrics:
- Discrepancy reduction percentage
- Monetary and time costs
- Number of wells required

## Dependencies

- NumPy
- SciPy
- Matplotlib
- Plotly
- PyTorch

## Relationship to Stage One

This implementation builds on the optimization approach from Stage One:
- Both use the same concept of optimizing well parameters for terrain transformation
- Both use the Nelder-Mead algorithm for optimization
- Both include monetary and time constraints

However, Stage Two adds:
- Realistic 3D geological modeling with geocores
- Physics-based slurry injection and pressure propagation
- More sophisticated visualization with 3D animations
- Improved optimization strategies (multiple starting points, well spacing)
- Step-by-step visualization of the process 