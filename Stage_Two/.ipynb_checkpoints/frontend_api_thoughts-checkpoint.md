# Terraforming AI: Front-End & API Implementation Guide -- Thoughts

This document outlines a possible approach for creating a front-end visualization and API layer for the Terraforming AI simulation system. It covers key data structures, API endpoints, implementation strategies, and technology recommendations.

## Key Data Structures for Front-End Visualization

### 1. Terrain Data
The primary data structure is the terrain surface represented as a 2D grid:
```json
{
  "x_vals": [0, 2, 4, ..., 100],  // Array of x coordinates 
  "y_vals": [0, 2, 4, ..., 100],  // Array of y coordinates
  "heights": [  // 2D array of terrain heights
    [10.5, 12.3, 9.8, ...],
    [11.2, 13.4, 10.1, ...],
    ...
  ],
  "grid_size": 50,  // Resolution of the grid
  "z_max": 60       // Maximum depth
}
```

### 2. Well Data
Each well is represented with:
```json
{
  "well_id": 1,
  "x0": 45.2,        // x-coordinate
  "y0": 32.7,        // y-coordinate
  "depth": 25.3,     // depth of injection in meters
  "volume": 320.5,   // volume of slurry in cubic meters
  "monetary_cost": 3525.0,  // calculated cost in dollars
  "time_cost": 253.0,       // calculated time in hours
  "surface_impact": {       // approximate surface impact dimensions
    "radius": 15.2,
    "max_height_change": 3.8
  }
}
```

### 3. Optimization Process Data
For the step-by-step visualization:
```json
{
  "iteration": 2,
  "terrain_states": [
    {
      "stage_name": "Initial Terrain",
      "heights": [...],  // 2D height array
      "wells": []        // No wells initially
    },
    {
      "stage_name": "After 1 Well",
      "heights": [...],  // 2D height array after 1st well
      "wells": [
        {"well_id": 1, "x0": 45.2, "y0": 32.7, "depth": 25.3, "volume": 320.5}
      ],
      "discrepancy_norm": 120.3
    },
    {
      "stage_name": "After 2 Wells",
      "heights": [...],  // Updated heights
      "wells": [
        {"well_id": 1, "x0": 45.2, "y0": 32.7, "depth": 25.3, "volume": 320.5},
        {"well_id": 2, "x0": 15.8, "y0": 65.3, "depth": 32.1, "volume": 250.0}
      ],
      "discrepancy_norm": 85.6,
      "total_monetary_cost": 5250.0,
      "total_time_cost": 470.0
    }
  ],
  "goal_terrain": {
    "heights": [...]  // 2D height array of goal terrain
  }
}
```

### 4. 3D Visualization Data 
For the 3D pressure/slurry visualization:
```json
{
  "pressure_snapshots": [
    {
      "iteration": 0,
      "isosurface_data": {
        "x": [...],  // Flattened x coordinates
        "y": [...],  // Flattened y coordinates
        "z": [...],  // Flattened z coordinates
        "values": [...],  // Pressure values
        "min_pressure": 0,
        "max_pressure": 17000
      },
      "wells": []
    },
    {
      "iteration": 1,
      "isosurface_data": {...},
      "wells": [
        {"well_id": 1, "x0": 45.2, "y0": 32.7, "depth": 25.3, "volume": 320.5}
      ]
    }
  ],
  "geocores": [
    {"x": 25.5, "y": 37.8, "z_vals": [0, 5, 10, ...], "density_vals": [...]},
    {"x": 75.2, "y": 42.3, "z_vals": [0, 5, 10, ...], "density_vals": [...]}
  ]
}
```

## API Structure Recommendations

### 1. Terrain Endpoints

```
GET /api/terrain/initial
```
Returns the initial terrain data

```
GET /api/terrain/goal
```
Returns the goal terrain data

```
GET /api/terrain/current
```
Returns the current state of the terrain after applied wells

```
GET /api/terrain/discrepancy
```
Returns the difference between goal and current terrain

### 2. Wells Endpoints

```
GET /api/wells
```
Returns all placed wells

```
POST /api/wells
```
Add a new well with parameters (x0, y0, depth, volume)

```
GET /api/wells/{well_id}
```
Get details for a specific well

```
GET /api/wells/preview?x={x}&y={y}&depth={depth}&volume={volume}
```
Preview the effect of placing a well without committing

### 3. Optimization Endpoints

```
POST /api/optimize/single?x={x}&y={y}
```
Optimize a single well at the given location

```
POST /api/optimize/terrain?max_iterations={10}&max_budget={10000}&max_time={1000}
```
Run full terrain optimization with constraints

```
GET /api/optimize/history
```
Get the step-by-step history of the optimization process

### 4. Visualization Endpoints

```
GET /api/visualize/3d
```
Get data for 3D visualization with pressure isosurfaces

```
GET /api/visualize/animation
```
Get data for animation of the transformation process

## Implementation Approach

Based on the existing codebase, here's how you could implement this API:

### 1. Create a Python Flask/FastAPI wrapper

```python
from fastapi import FastAPI
from src.terrain_optimization import TerrainOptimizer, TerrainWell
import numpy as np

app = FastAPI()
optimizer = TerrainOptimizer(grid_size=50, z_max=60, num_cores=5, seed=42)

@app.get("/api/terrain/initial")
def get_initial_terrain():
    return {
        "x_vals": optimizer.x_vals.tolist(),
        "y_vals": optimizer.y_vals.tolist(),
        "heights": optimizer.original_surface.tolist(),
        "grid_size": optimizer.grid_size,
        "z_max": optimizer.z_max
    }

@app.get("/api/terrain/goal")
def get_goal_terrain():
    # Generate or load goal terrain if not already available
    if not hasattr(app, "goal_terrain"):
        app.goal_terrain = optimizer.generate_goal_terrain(method='custom')
    
    return {
        "x_vals": optimizer.x_vals.tolist(),
        "y_vals": optimizer.y_vals.tolist(),
        "heights": app.goal_terrain.tolist(),
        "grid_size": optimizer.grid_size
    }

@app.post("/api/wells")
def add_well(x0: float, y0: float, depth: float, volume: float):
    well = TerrainWell(x0, y0, depth, volume)
    new_terrain = optimizer.apply_well(well)
    return {
        "well_id": len(optimizer.wells),
        "x0": well.x0,
        "y0": well.y0, 
        "depth": well.depth,
        "volume": well.volume,
        "monetary_cost": well.monetary_cost(),
        "time_cost": well.time_cost(),
        "new_terrain": new_terrain.tolist()
    }

@app.get("/api/wells")
def get_wells():
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

@app.post("/api/optimize/single")
def optimize_single_well(x: float, y: float):
    if not hasattr(app, "goal_terrain"):
        app.goal_terrain = optimizer.generate_goal_terrain(method='custom')
        
    # Get current terrain
    if len(optimizer.wells) > 0:
        current_terrain = optimizer.surface_snapshots[-1]
    else:
        current_terrain = optimizer.original_surface
    
    # Optimize parameters for this location
    depth, volume = optimizer.optimize_well_params(x, y, current_terrain, app.goal_terrain)
    
    # Create and apply the well
    well = TerrainWell(x, y, depth, volume)
    new_terrain = optimizer.apply_well(well)
    
    return {
        "well_id": len(optimizer.wells) - 1,  # The well has already been added to optimizer.wells
        "x0": well.x0,
        "y0": well.y0, 
        "depth": well.depth,
        "volume": well.volume,
        "monetary_cost": well.monetary_cost(),
        "time_cost": well.time_cost(),
        "new_terrain": new_terrain.tolist()
    }
```

### 2. Serialize the 3D data for visualization

```python
@app.get("/api/visualize/3d")
def get_3d_data(iteration: int = -1):
    if iteration < 0 or iteration >= len(optimizer.pressure_snapshots):
        iteration = len(optimizer.pressure_snapshots) - 1
        
    from src.plotting_utils import flatten_isosurface_data
    
    X, Y, Z, Pvals = flatten_isosurface_data(
        optimizer.pressure_snapshots[iteration],
        optimizer.x_vals,
        optimizer.y_vals, 
        optimizer.z_vals
    )
    
    wells_data = []
    for i, well in enumerate(optimizer.wells[:iteration+1]):
        wells_data.append({
            "well_id": i,
            "x0": well.x0,
            "y0": well.y0,
            "depth": well.depth,
            "volume": well.volume
        })
    
    return {
        "iteration": iteration,
        "isosurface_data": {
            "x": X,
            "y": Y, 
            "z": Z,
            "values": Pvals,
            "min_pressure": 0,
            "max_pressure": 17000
        },
        "wells": wells_data
    }

@app.get("/api/optimize/history")
def get_optimization_history():
    if not hasattr(app, "goal_terrain"):
        return {"error": "No optimization has been run yet"}
    
    terrain_states = []
    
    # Add initial state
    terrain_states.append({
        "stage_name": "Initial Terrain",
        "heights": optimizer.original_surface.tolist(),
        "wells": [],
        "discrepancy_norm": 0.0,
        "total_monetary_cost": 0.0,
        "total_time_cost": 0.0
    })
    
    # Add states for each well placement
    cumulative_cost = 0.0
    cumulative_time = 0.0
    
    for i, terrain in enumerate(optimizer.surface_snapshots[1:], 1):
        well = optimizer.wells[i-1]  # wells are 0-indexed, but we start from state 1
        cumulative_cost += well.monetary_cost()
        cumulative_time += well.time_cost()
        
        # Get discrepancy between current and goal
        if hasattr(app, "goal_terrain"):
            discrepancy = app.goal_terrain - terrain
            discrepancy_norm = np.linalg.norm(discrepancy)
        else:
            discrepancy_norm = 0.0
            
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
            "heights": app.goal_terrain.tolist() if hasattr(app, "goal_terrain") else []
        }
    }
```

### 3. Run Full Optimization Process

```python
@app.post("/api/optimize/terrain")
def optimize_terrain(max_iterations: int = 10, max_budget: float = 10000, max_time: float = 1000):
    # Generate a goal terrain if not already available
    if not hasattr(app, "goal_terrain"):
        app.goal_terrain = optimizer.generate_goal_terrain(method='custom')
    
    # Reset the optimizer to start fresh
    global optimizer
    optimizer = TerrainOptimizer(grid_size=50, z_max=60, num_cores=5, seed=42)
    
    # Run the optimization
    final_terrain, wells, monetary_cost, time_cost, terrain_history = optimizer.iterative_optimization(
        app.goal_terrain, 
        max_iterations=max_iterations,
        max_monetary_budget=max_budget,
        max_time_budget=max_time
    )
    
    # Return summary of results
    return {
        "status": "completed",
        "iterations": len(wells),
        "monetary_cost": float(monetary_cost),
        "time_cost": float(time_cost),
        "well_count": len(wells),
        "final_discrepancy_norm": float(np.linalg.norm(app.goal_terrain - final_terrain))
    }
```

## Front-end Technologies to Consider

### 1. 3D Visualization
- **Plotly.js**: Already used in the Python code, simplest to adapt
- **Three.js**: For more custom WebGL visualizations 
- **CesiumJS**: If you need geospatial visualization capabilities

### 2. User Interface
- **React** or **Vue.js**: For component-based UI development
- **Material-UI** or **Chakra UI**: For quick UI development with pre-built components
- **D3.js**: For custom data visualizations and charts

### 3. State Management
- **Redux** or **Context API** for React
- **Vuex** for Vue.js

### 4. API Interaction
- **Axios** or **Fetch API** for handling HTTP requests
- **React Query** for data fetching, caching, and state management

## Next Steps

1. **Create a basic API wrapper**
   - Set up FastAPI or Flask with the endpoints outlined above
   - Implement serialization functions to convert NumPy arrays to JSON
   - Set up CORS for cross-origin requests if needed
   - Add basic error handling

2. **Develop the front-end components**
   - Create a terrain visualization component using Plotly.js or Three.js
   - Build well placement and editing UI
   - Implement the optimization controls
   - Create a dashboard for monitoring costs and progress

3. **Connect front-end to the API**
   - Implement API service layers
   - Handle async operations and loading states
   - Set up state management for terrain and wells data

4. **Add interactivity features**
   - Well placement via clicking on the terrain
   - Parameter adjustment via sliders/inputs
   - Animation controls for the optimization process
   - View switching between 2D and 3D representations

5. **Testing and optimization**
   - Test API endpoints with different parameters
   - Optimize 3D rendering performance
   - Ensure responsive design for different screen sizes
   - Add loading indicators for long-running operations

## Suggested Project Structure

```
terraforming-ai/
├── api/
│   ├── main.py             # FastAPI app
│   ├── endpoints/          # API endpoint implementations
│   └── models/             # Pydantic models for request/response
├── src/                    # Existing Python code
├── client/                 # Front-end code
│   ├── src/
│   │   ├── components/     # React/Vue components
│   │   ├── services/       # API service layer
│   │   ├── store/          # State management
│   │   └── visualizations/ # 3D visualization components
│   └── public/             # Static assets
└── notebooks/              # Jupyter notebooks for experimentation
```

## Deployment Considerations

1. **Backend**: 
   - Deploy the FastAPI application with Uvicorn/Gunicorn
   - Consider containerization with Docker for easier deployment
   - Use a reverse proxy like Nginx for production deployment

2. **Frontend**:
   - Build and serve as static files
   - Deploy to a CDN for improved performance
   - Consider serverless hosting options (Netlify, Vercel, etc.)

3. **Performance**:
   - Optimize 3D visualizations for web performance
   - Consider using WebAssembly for computation-heavy parts
   - Implement caching strategies for API responses
