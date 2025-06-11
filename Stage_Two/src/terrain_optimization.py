import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Stage Two utilities
from geocore_utils import generate_random_geocores
from interpolation_utils import build_3D_from_geocores, build_3D_pressure
from injection_utils import inject_slurry, update_pressure, compute_volume_used
from plotting_utils import flatten_isosurface_data, create_geocore_traces

class TerrainWell:
    def __init__(self, x0, y0, depth, volume):
        """
        Initialize a well for terrain modification.
        
        Args:
            x0, y0: Surface coordinates for the well
            depth: Depth of the injection (m)
            volume: Volume of slurry to inject (cubic meters)
        """
        self.x0 = x0
        self.y0 = y0
        self.depth = depth
        self.volume = volume
        
        # Define cost coefficients (as in Stage One)
        self.a = 100  # Monetary cost coefficient for depth
        self.b = 50   # Monetary cost coefficient for volume
        self.c = 10   # Time cost coefficient for depth
        self.d = 5    # Time cost coefficient for volume
    
    def monetary_cost(self):
        """Calculate the monetary cost of the well"""
        return self.a * self.depth + self.b * self.volume
    
    def time_cost(self):
        """Calculate the time cost of the well"""
        return self.c * self.depth + self.d * self.volume

class TerrainOptimizer:
    def __init__(self, grid_size=100, z_max=60, num_cores=5, seed=42):
        """
        Initialize the terrain optimizer with 3D geological data.
        
        Args:
            grid_size: Number of grid points in x and y dimensions
            z_max: Maximum depth for geocores
            num_cores: Number of geocores to generate
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.z_max = z_max
        self.seed = seed
        
        # Set up the grid
        self.x_vals = np.linspace(0, 100, grid_size)
        self.y_vals = np.linspace(0, 100, grid_size)
        self.z_vals = np.linspace(0, z_max, grid_size)
        
        # Generate geocores
        self.geocores = generate_random_geocores(
            num_cores=num_cores, 
            z_max=z_max, 
            z_points=50, 
            layer_count=4, 
            seed=seed
        )
        
        # Interpolate 3D density and pressure maps
        self.density_3D = build_3D_from_geocores(
            self.geocores, 
            self.x_vals, 
            self.y_vals, 
            self.z_vals
        )
        
        self.pressure_3D = build_3D_pressure(self.density_3D, self.z_vals)
        
        # Store original data
        self.original_pressure_3D = self.pressure_3D.copy()
        self.original_surface = self.extract_surface(self.pressure_3D)
        
        # Initialize slurry concentration to zeros
        self.slurry_3D = np.zeros_like(self.pressure_3D)
        
        # Store simulation steps
        self.pressure_snapshots = [self.pressure_3D.copy()]
        self.surface_snapshots = [self.original_surface.copy()]
        
        # Store wells
        self.wells = []
    
    def extract_surface(self, pressure_3D, threshold=5000):
        """
        Extract a 2D surface from 3D pressure data.
        The surface is defined as the height where pressure exceeds the threshold.
        
        Returns:
            2D numpy array representing the surface height
        """
        surface = np.zeros((self.grid_size, self.grid_size))
        
        for ix in range(self.grid_size):
            for iy in range(self.grid_size):
                # Find the first z-value where pressure exceeds threshold
                for iz in range(self.grid_size):
                    if pressure_3D[ix, iy, iz] >= threshold:
                        # Convert to a height value (invert z-axis)
                        surface[ix, iy] = self.z_max - self.z_vals[iz]
                        break
                else:
                    # If no value exceeds threshold, set to minimum height
                    surface[ix, iy] = 0
        
        return surface
    
    def generate_goal_terrain(self, method='random', file_path=None):
        """
        Generate a goal terrain.
        
        Args:
            method: 'random', 'smooth', 'custom', or 'file' to load from file
            file_path: Path to load terrain if method='file'
            
        Returns:
            2D numpy array representing the goal terrain
        """
        if method == 'file' and file_path and os.path.exists(file_path):
            try:
                goal_terrain = np.load(file_path)
                # Resize if necessary
                if goal_terrain.shape != (self.grid_size, self.grid_size):
                    from scipy.ndimage import zoom
                    zoom_factor = (self.grid_size/goal_terrain.shape[0], 
                                  self.grid_size/goal_terrain.shape[1])
                    goal_terrain = zoom(goal_terrain, zoom_factor)
                return goal_terrain
            except Exception as e:
                print(f"Error loading terrain from file: {e}")
                # Fall back to custom method
                method = 'custom'
                
        # Start with the original surface
        goal_terrain = self.original_surface.copy()
        
        if method == 'random':
            # Add random perturbations
            np.random.seed(self.seed + 1)  # Different seed than geocores
            noise = np.random.rand(self.grid_size, self.grid_size) * 5  # 0-5m noise
            goal_terrain += noise
            
            # Apply some smoothing
            from scipy.ndimage import gaussian_filter
            goal_terrain = gaussian_filter(goal_terrain, sigma=2)
            
        elif method == 'smooth':
            # Create a smoother terrain with some interesting features
            from scipy.ndimage import gaussian_filter
            
            # Add a few "hills"
            for _ in range(3):
                x_center = np.random.randint(0, self.grid_size)
                y_center = np.random.randint(0, self.grid_size)
                radius = np.random.randint(10, 30)
                height = np.random.uniform(3, 8)
                
                x_grid, y_grid = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
                distance = np.sqrt((x_grid - x_center)**2 + (y_grid - y_center)**2)
                hill = height * np.exp(-0.5 * (distance / radius)**2)
                
                goal_terrain += hill
            
            # Smooth the result
            goal_terrain = gaussian_filter(goal_terrain, sigma=3)
            
        elif method == 'custom' or True:  # Default to custom if method not recognized
            # Create a significantly different terrain with distinct features
            from scipy.ndimage import gaussian_filter
            
            # Reset to a basic terrain with some overall slopes
            goal_terrain = np.zeros((self.grid_size, self.grid_size))
            
            # Create a primary gradient across the terrain (higher on one side)
            x_grid, y_grid = np.meshgrid(
                np.linspace(0, 1, self.grid_size),
                np.linspace(0, 1, self.grid_size)
            )
            
            # Add a diagonal slope
            gradient = 10 * (x_grid + y_grid)
            goal_terrain += gradient
            
            # Add several distinct hills and valleys
            num_features = 6
            np.random.seed(self.seed + 42)  # Completely different seed
            
            for i in range(num_features):
                x_center = np.random.randint(5, self.grid_size-5)
                y_center = np.random.randint(5, self.grid_size-5)
                radius = np.random.randint(5, 15)
                
                # Alternate between hills (positive) and valleys (negative)
                if i % 2 == 0:
                    height = np.random.uniform(4, 12)  # Hills
                else:
                    height = np.random.uniform(-10, -3)  # Valleys
                
                distance = np.sqrt((x_grid * self.grid_size - x_center)**2 + 
                                  (y_grid * self.grid_size - y_center)**2)
                feature = height * np.exp(-0.5 * (distance / radius)**2)
                
                goal_terrain += feature
            
            # Add a river-like depression
            river_y = np.random.randint(self.grid_size // 4, 3 * self.grid_size // 4)
            river_width = np.random.randint(2, 5)
            river_depth = np.random.uniform(-8, -4)
            
            for x in range(self.grid_size):
                # Meandering river with some randomness
                center = river_y + int(3 * np.sin(x / 10))
                for y in range(max(0, center - river_width), min(self.grid_size, center + river_width)):
                    # Create river depression with smooth edges
                    dist_from_center = abs(y - center)
                    depression = river_depth * (1 - dist_from_center / river_width)**2
                    goal_terrain[y, x] += depression
            
            # Apply appropriate smoothing to make it natural
            goal_terrain = gaussian_filter(goal_terrain, sigma=1.5)
            
            # Ensure it has a reasonable range similar to the original
            min_height = np.min(self.original_surface)
            range_height = np.max(self.original_surface) - min_height
            
            # Scale and shift to match original's range but with different pattern
            goal_terrain = ((goal_terrain - np.min(goal_terrain)) / 
                           (np.max(goal_terrain) - np.min(goal_terrain)) * range_height + min_height)
            
            # Add a significant height increase of at least 20 meters
            height_increase = 25.0  # 25 meters higher
            goal_terrain += height_increase
            
            print(f"Goal terrain average height: {np.mean(goal_terrain):.2f}m")
            print(f"Initial terrain average height: {np.mean(self.original_surface):.2f}m")
            print(f"Average height difference: {np.mean(goal_terrain - self.original_surface):.2f}m")
        
        return goal_terrain
    
    def apply_well(self, well):
        """
        Apply the effect of a well by injecting slurry at the specified location.
        
        Args:
            well: TerrainWell object
        """
        # Find the closest grid points to the well location
        ix = np.abs(self.x_vals - well.x0).argmin()
        iy = np.abs(self.y_vals - well.y0).argmin()
        
        # Convert depth to z-index (invert direction)
        iz = np.abs(self.z_vals - (self.z_max - well.depth)).argmin()
        
        # Create a copy of the current slurry state
        old_slurry_3D = self.slurry_3D.copy()
        
        # Inject slurry at the well location
        injection_location = (ix, iy, iz)
        self.slurry_3D, placed_volume = inject_slurry(
            self.slurry_3D, 
            self.pressure_3D, 
            well.volume, 
            injection_location
        )
        
        # Update pressure based on new slurry distribution
        self.pressure_3D = update_pressure(
            self.pressure_3D,
            old_slurry_3D,
            self.slurry_3D
        )
        
        # Extract the new surface
        new_surface = self.extract_surface(self.pressure_3D)
        
        # Save snapshots
        self.pressure_snapshots.append(self.pressure_3D.copy())
        self.surface_snapshots.append(new_surface)
        
        # Add well to the list
        self.wells.append(well)
        
        return new_surface
    
    def get_worst_discrepancy_points(self, current_terrain, goal_terrain, n_points=5, buffer=5):
        """
        Find the n points with the largest discrepancy between current and goal terrains.
        Avoids points near the boundary to prevent edge placement issues.
        
        Args:
            current_terrain: 2D array representing current terrain
            goal_terrain: 2D array representing goal terrain
            n_points: Number of worst points to return
            buffer: Minimum distance from terrain edges to consider (prevents boundary placement)
            
        Returns:
            List of (x, y) coordinates for worst discrepancy points
        """
        # Calculate discrepancy
        discrepancy = goal_terrain - current_terrain
        
        # Create a mask to exclude boundary points
        mask = np.ones_like(discrepancy, dtype=bool)
        mask[:buffer, :] = False  # Top buffer
        mask[-buffer:, :] = False  # Bottom buffer
        mask[:, :buffer] = False  # Left buffer
        mask[:, -buffer:] = False  # Right buffer
        
        # Apply the mask and set discrepancy in buffer zones to a low value
        masked_discrepancy = discrepancy.copy()
        masked_discrepancy[~mask] = -np.inf  # Ensure boundary points aren't selected
        
        # Find indices of n largest absolute discrepancies (we care about magnitude of difference)
        abs_masked_discrepancy = np.abs(masked_discrepancy)
        flat_indices = np.argsort(abs_masked_discrepancy.flatten())[-n_points:]
        
        # Convert to 2D indices
        worst_indices = np.unravel_index(flat_indices, discrepancy.shape)
        
        # Convert to real-world coordinates
        # Note: In numpy arrays, the first index is row (y) and second index is column (x)
        row_indices, col_indices = worst_indices
        worst_points = [(self.x_vals[col_idx], self.y_vals[row_idx]) 
                       for row_idx, col_idx in zip(row_indices, col_indices)]
        
        # If we couldn't find enough points (due to masking), try with a smaller buffer
        if len(worst_points) < n_points and buffer > 1:
            print(f"Warning: Could not find {n_points} points with buffer={buffer}. Trying with smaller buffer.")
            return self.get_worst_discrepancy_points(current_terrain, goal_terrain, n_points, buffer=buffer-1)
        
        return worst_points
    
    def optimize_well_params(self, x0, y0, current_terrain, goal_terrain):
        """
        Optimize the well parameters (depth, volume) for a given location.
        
        Args:
            x0, y0: Surface coordinates for the well
            current_terrain: 2D array representing current terrain
            goal_terrain: 2D array representing goal terrain
            
        Returns:
            Optimized (depth, volume) for the well
        """
        # Find grid indices for the well location
        ix = np.abs(self.x_vals - x0).argmin()
        iy = np.abs(self.y_vals - y0).argmin()
        
        # Define bounds for depth and volume
        bounds = [(5, 50), (10, 500)]  # (depth_min, depth_max), (volume_min, volume_max)
        
        # Try multiple initial starting points to avoid local minima
        best_params = None
        best_loss = float('inf')
        
        # Define different starting points to try
        starting_points = [
            [bounds[0][0], bounds[1][0]],  # Minimum depth, minimum volume
            [bounds[0][1], bounds[1][1]],  # Maximum depth, maximum volume
            [(bounds[0][0] + bounds[0][1]) / 2, (bounds[1][0] + bounds[1][1]) / 2],  # Middle values
            [bounds[0][0], bounds[1][1]],  # Minimum depth, maximum volume
            [bounds[0][1], bounds[1][0]]   # Maximum depth, minimum volume
        ]
        
        # Define the objective function for minimization
        def objective(params):
            depth, volume = params
            
            # Create a temporary well
            temp_well = TerrainWell(x0, y0, depth, volume)
            
            # Save current state
            prev_slurry = self.slurry_3D.copy()
            prev_pressure = self.pressure_3D.copy()
            
            # Apply the well
            new_terrain = self.apply_well(temp_well)
            
            # Calculate the resulting discrepancy in a local window around the well
            window_size = 15  # Increased window size to better capture well effects
            x_start = max(0, ix - window_size)
            x_end = min(self.grid_size - 1, ix + window_size)
            y_start = max(0, iy - window_size)
            y_end = min(self.grid_size - 1, iy + window_size)
            
            local_discrepancy = goal_terrain[x_start:x_end, y_start:y_end] - \
                               new_terrain[x_start:x_end, y_start:y_end]
            
            # Calculate loss - asymmetric to penalize overshooting more than undershooting
            undershooting = local_discrepancy[local_discrepancy > 0]
            overshooting = local_discrepancy[local_discrepancy < 0]
            
            loss_undershot = np.sum(undershooting**2) if len(undershooting) > 0 else 0
            loss_overshot = np.sum(overshooting**2) * 1.5 if len(overshooting) > 0 else 0  # Reduced penalty for overshooting
            
            # If the well made no difference, penalize highly
            if np.all(new_terrain == current_terrain):
                loss = float('inf')
            else:
                loss = loss_undershot + loss_overshot
            
            # Add cost penalty - reduced to encourage exploration of larger wells
            cost_weight = 0.0005  # Reduced weight for cost in the objective function 
            cost = temp_well.monetary_cost() * cost_weight
            
            # Restore previous state (this is a simulation only)
            self.slurry_3D = prev_slurry
            self.pressure_3D = prev_pressure
            self.pressure_snapshots.pop()  # Remove the simulated snapshot
            self.surface_snapshots.pop()   # Remove the simulated snapshot
            
            if hasattr(self, 'wells') and len(self.wells) > 0 and self.wells[-1] == temp_well:
                self.wells.pop()  # Remove the temporary well
            
            # Total loss combines terrain discrepancy and cost
            total_loss = loss + cost
            
            return total_loss
        
        # Try each starting point
        for i, initial_params in enumerate(starting_points):
            # Run optimization with this starting point
            result = minimize(
                objective, 
                initial_params, 
                method='Nelder-Mead', 
                bounds=bounds, 
                options={
                    'maxiter': 150,        # Increased from 100
                    'xatol': 1e-3,         # Reduced tolerance for more precision
                    'fatol': 1e-3,         # Reduced tolerance for more precision
                    'adaptive': True       # Use adaptive step size for better convergence
                }
            )
            
            # Check if this is better than previous results
            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
                print(f"  Found better parameters at starting point {i+1}: {best_params}, loss: {best_loss}")
        
        # If we found a good solution, return it
        if best_params is not None:
            return best_params
        else:
            print(f"Optimization failed for all starting points. Using default parameters.")
            return [10, 50]  # Default mid-range values if optimization fails
    
    def iterative_optimization(self, goal_terrain, max_iterations=10, 
                             max_monetary_budget=10000, max_time_budget=1000):
        """
        Iteratively optimize well placements to transform the terrain.
        
        Args:
            goal_terrain: 2D array representing the goal terrain
            max_iterations: Maximum number of wells to place
            max_monetary_budget: Maximum monetary budget
            max_time_budget: Maximum time budget
            
        Returns:
            Final terrain, list of wells, monetary cost, time cost, history of terrain states
        """
        current_terrain = self.original_surface.copy()
        monetary_cost = 0
        time_cost = 0
        
        # Store the history of terrain states after each injection
        terrain_history = [current_terrain.copy()]  # Start with initial terrain
        
        # For placing wells, maintain a list of placed locations
        placed_well_locations = []
        
        # Minimum spacing between wells (to avoid clustering)
        min_well_spacing = 15  # meters
        
        # Start the optimization loop
        for iteration in range(max_iterations):
            # Check if we've exceeded our budget constraints
            if monetary_cost >= max_monetary_budget:
                print(f"Stopping: Exceeded monetary budget (${monetary_cost:.2f})")
                break
            if time_cost >= max_time_budget:
                print(f"Stopping: Exceeded time budget ({time_cost:.2f} hours)")
                break
            
            # Find the worst discrepancy points
            n_candidates = min(20, (self.grid_size**2) // 4)  # Limit to reasonable number
            worst_points = self.get_worst_discrepancy_points(current_terrain, goal_terrain, n_points=n_candidates)
            
            # Filter out points that are too close to existing wells
            valid_points = []
            for point in worst_points:
                x, y = point
                too_close = False
                
                # Check distance to all previously placed wells
                for well_x, well_y in placed_well_locations:
                    distance = np.sqrt((x - well_x)**2 + (y - well_y)**2)
                    if distance < min_well_spacing:
                        too_close = True
                        break
                
                if not too_close:
                    valid_points.append(point)
            
            # If we have no valid points, reduce spacing requirement
            if not valid_points:
                print(f"No valid points found with spacing={min_well_spacing}. Reducing spacing.")
                min_well_spacing = max(5, min_well_spacing - 5)
                worst_points = self.get_worst_discrepancy_points(current_terrain, goal_terrain, n_points=n_candidates)
                valid_points = worst_points  # Accept any points with reduced spacing
            
            # Choose the worst valid point
            x0, y0 = valid_points[0]
            
            print(f"Iteration {iteration+1}/{max_iterations}: Optimizing well at ({x0:.1f}, {y0:.1f})")
            
            # Check if current terrain is already close to goal terrain at this point
            ix = np.abs(self.x_vals - x0).argmin()
            iy = np.abs(self.y_vals - y0).argmin()
            window_size = 5
            i_min, i_max = max(0, ix-window_size), min(self.grid_size-1, ix+window_size)
            j_min, j_max = max(0, iy-window_size), min(self.grid_size-1, iy+window_size)
            
            local_current = current_terrain[j_min:j_max+1, i_min:i_max+1]
            local_goal = goal_terrain[j_min:j_max+1, i_min:i_max+1]
            local_discrepancy = np.abs(local_goal - local_current)
            
            if np.max(local_discrepancy) < 0.5:  # If discrepancy is already low
                print(f"  Skipping well at ({x0:.1f}, {y0:.1f}) as terrain is already close to goal")
                continue
            
            # Optimize well parameters (depth and volume)
            depth, volume = self.optimize_well_params(x0, y0, current_terrain, goal_terrain)
            
            # Create and apply the optimized well
            well = TerrainWell(x0, y0, depth, volume)
            current_terrain = self.apply_well(well)
            
            # Calculate costs
            well_monetary_cost = well.monetary_cost()
            well_time_cost = well.time_cost()
            monetary_cost += well_monetary_cost
            time_cost += well_time_cost
            
            # Store the well location to avoid placing wells too close to each other
            placed_well_locations.append((x0, y0))
            
            # Calculate current discrepancy norm
            current_norm = np.linalg.norm(goal_terrain - current_terrain)
            
            print(f"  Depth: {depth:.2f}m, Volume: {volume:.2f}m³")
            print(f"  Cost: ${well_monetary_cost:.2f}, Time: {well_time_cost:.2f} hours")
            print(f"  Total Cost: ${monetary_cost:.2f}, Total Time: {time_cost:.2f} hours")
            print(f"  Discrepancy norm: {current_norm:.2f}")
            
            # Save the terrain state after this injection
            terrain_history.append(current_terrain.copy())
            
            # Check if we've reached sufficient accuracy
            if current_norm < 1.0:
                print(f"Stopping: Reached sufficient accuracy (norm={current_norm:.4f})")
                break
        
        return current_terrain, self.wells, monetary_cost, time_cost, terrain_history
    
    def plot_optimization_results(self, goal_terrain, final_terrain):
        """
        Plot the optimization results, including initial, goal, and final terrains.
        
        Args:
            goal_terrain: 2D array representing the goal terrain
            final_terrain: 2D array representing the final terrain
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        # Create X, Y coordinate grids
        X, Y = np.meshgrid(self.x_vals, self.y_vals)
        
        # Plot original terrain
        axes[0].plot_surface(X, Y, self.original_surface, cmap='viridis', alpha=0.7)
        axes[0].set_title('Original Terrain')
        
        # Plot goal terrain
        axes[1].plot_surface(X, Y, goal_terrain, cmap='plasma', alpha=0.7)
        axes[1].set_title('Goal Terrain')
        
        # Plot final terrain
        axes[2].plot_surface(X, Y, final_terrain, cmap='inferno', alpha=0.7)
        axes[2].set_title('Final Terrain')
        
        # Plot discrepancy
        discrepancy = goal_terrain - final_terrain
        axes[3].plot_surface(X, Y, discrepancy, cmap='coolwarm', alpha=0.7)
        axes[3].set_title('Discrepancy (Goal - Final)')
        
        # Add well locations to all plots
        for ax in axes[:3]:
            for well in self.wells:
                ax.scatter([well.x0], [well.y0], [0], 
                          marker='^', s=100, color='red', 
                          label='Well Locations')
        
        # Adjust plot settings
        for ax in axes:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Height (m)')
            ax.set_zlim(0, max(np.max(goal_terrain), np.max(self.original_surface)) + 5)
        
        plt.tight_layout()
        plt.show()
    
    def create_3D_animation(self):
        """
        Create a 3D animation of the terrain transformation using Plotly.
        """
        # Create frames for animation
        frames = []
        geocore_traces = create_geocore_traces(self.geocores)
        
        # Set pressure limits
        min_p, max_p = 0, 17000
        surface_count = 6
        colorscale = 'Viridis'
        
        # Create a frame for each pressure snapshot
        for idx, press_3D in enumerate(self.pressure_snapshots):
            X, Y, Z, Pvals = flatten_isosurface_data(
                press_3D, 
                self.x_vals, 
                self.y_vals, 
                self.z_vals,
                min_p=min_p, 
                max_p=max_p
            )
            
            iso_trace = go.Isosurface(
                x=X, y=Y, z=Z,
                value=Pvals,
                isomin=min_p,
                isomax=max_p,
                surface_count=surface_count,
                colorscale=colorscale,
                caps=dict(x_show=False, y_show=False, z_show=False),
                opacity=0.6,
                showscale=True,
                name=f"Press Iter {idx}"
            )
            
            # Add markers for wells that have been placed by this step
            if idx > 0:
                wells_x, wells_y, wells_z = [], [], []
                for well in self.wells[:idx]:
                    wells_x.append(well.x0)
                    wells_y.append(well.y0)
                    wells_z.append(self.z_max - well.depth)  # Convert to appropriate Z coordinate
                
                well_trace = go.Scatter3d(
                    x=wells_x,
                    y=wells_y,
                    z=wells_z,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='diamond'
                    ),
                    name=f"Wells (Step {idx})"
                )
                
                frame_data = [iso_trace, well_trace] + geocore_traces
            else:
                frame_data = [iso_trace] + geocore_traces
            
            frames.append(go.Frame(data=frame_data, name=f"frame{idx}"))
        
        # Create the figure with animation
        fig = go.Figure(data=frames[0].data, frames=frames)
        
        fig.update_layout(
            title="Animated 3D Pressure Isosurface Over Injection Steps",
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Depth (m)',
                zaxis=dict(autorange='reversed')
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                    }]
                }, {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }]
            }],
            sliders=[{
                'active': 0,
                'steps': [
                    {
                        'label': f'Step {i}',
                        'method': 'animate',
                        'args': [[f'frame{i}'], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }]
                    }
                    for i in range(len(frames))
                ]
            }]
        )
        
        return fig

def main():
    """
    Main function to run the terrain optimization process.
    """
    print("Starting terrain optimization based on 3D geocore data...")
    
    # Initialize terrain optimizer
    optimizer = TerrainOptimizer(grid_size=100, z_max=60, num_cores=5, seed=42)
    
    # Generate or load goal terrain
    # Try to load from Stage One if available
    stage_one_path = "../../../Newest_Attempt/Stage_One/goal_terrain.npy"
    goal_terrain = optimizer.generate_goal_terrain(method='file', file_path=stage_one_path)
    
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
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    print(f"Final monetary cost: ${monetary_cost:.2f}")
    print(f"Final time cost: {time_cost:.2f} hours")
    print(f"Number of wells placed: {len(wells)}")
    
    # Plot the optimization results
    print("Generating visualization...")
    optimizer.plot_optimization_results(goal_terrain, final_terrain)
    
    # Create 3D animation
    fig = optimizer.create_3D_animation()
    fig.show()
    
    print("Terrain optimization complete.")

if __name__ == "__main__":
    main() 