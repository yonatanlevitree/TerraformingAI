import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from terrain_optimization import TerrainWell
from geocore_utils import generate_random_geocores
from interpolation_utils import build_3D_from_geocores, build_3D_pressure
from injection_utils import inject_slurry, update_pressure

class GeneticSingleOptimizer:
    """
    A genetic algorithm optimizer for single well parameter optimization.
    This optimizer uses evolutionary strategies to find optimal well parameters
    (x, y, depth, volume) for a given location, considering 3D geological modeling
    and BFS-based fluid dynamics simulation.
    """
    
    def __init__(self, 
                 grid_size: int = 100,
                 z_max: float = 60,
                 num_cores: int = 5,
                 seed: int = 42,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 6,
                 elite_size: int = 3,
                 num_generations: int = 150):
        """
        Initialize the genetic single-well optimizer.
        
        Args:
            grid_size: Size of the terrain grid
            z_max: Maximum depth for geocores
            num_cores: Number of geocores to generate
            seed: Random seed for reproducibility
            population_size: Number of solutions in each generation
            mutation_rate: Probability of mutation for each parameter
            tournament_size: Number of solutions to compare in tournament selection
            elite_size: Number of best solutions to preserve in each generation
            num_generations: Number of generations per well placement
        """
        self.grid_size = grid_size
        self.z_max = z_max
        self.seed = seed
        
        # Genetic algorithm parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.num_generations = num_generations
        
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
        
        # Initialize terrain state
        self.goal_terrain = None
        self.current_terrain = None
    
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
    
    def set_terrain(self, goal_terrain: np.ndarray):
        """
        Set the goal terrain surface.
        
        Args:
            goal_terrain: Goal terrain surface
        """
        self.goal_terrain = goal_terrain
        self.current_terrain = self.original_surface.copy()
    
    def _initialize_population(self) -> List[TerrainWell]:
        """
        Initialize a population of wells with random parameters.
        
        Returns:
            List of TerrainWell objects with random parameters
        """
        population = []
        for _ in range(self.population_size):
            x = np.random.uniform(0, 100)  # x coordinate
            y = np.random.uniform(0, 100)  # y coordinate
            depth = np.random.uniform(5, 50)  # depth bounds
            volume = np.random.uniform(10, 500)  # volume bounds
            population.append(TerrainWell(x, y, depth, volume))
        return population
    
    def _calculate_fitness(self, well: TerrainWell) -> float:
        """
        Calculate fitness of a well configuration.
        
        Args:
            well: Well configuration to evaluate
            
        Returns:
            Fitness score (lower is better)
        """
        # Save current state
        prev_slurry = self.slurry_3D.copy()
        prev_pressure = self.pressure_3D.copy()
        
        # Find grid indices for the well location
        ix = np.abs(self.x_vals - well.x0).argmin()
        iy = np.abs(self.y_vals - well.y0).argmin()
        iz = np.abs(self.z_vals - (self.z_max - well.depth)).argmin()
        
        # Apply the well using BFS-based injection
        self.slurry_3D, placed_volume = inject_slurry(
            self.slurry_3D,
            self.pressure_3D,
            well.volume,
            (ix, iy, iz)
        )
        
        # Update pressure based on new slurry distribution
        self.pressure_3D = update_pressure(
            self.pressure_3D,
            prev_slurry,
            self.slurry_3D
        )
        
        # Extract the new surface
        new_surface = self.extract_surface(self.pressure_3D)
        
        # Calculate discrepancy
        discrepancy = self.goal_terrain - new_surface
        
        # Calculate loss with asymmetric weights
        overshooting = discrepancy[discrepancy < 0]
        undershooting = discrepancy[discrepancy > 0]
        
        loss_overshot = np.sum(overshooting**2) * 1.5 if len(overshooting) > 0 else 0
        loss_undershot = np.sum(undershooting**2) if len(undershooting) > 0 else 0
        
        # Add cost penalty
        cost_weight = 0.0005
        cost_penalty = well.monetary_cost() * cost_weight
        
        # Restore previous state
        self.slurry_3D = prev_slurry
        self.pressure_3D = prev_pressure
        
        return loss_overshot + loss_undershot + cost_penalty
    
    def _tournament_selection(self, population: List[TerrainWell], 
                            fitnesses: List[float]) -> TerrainWell:
        """
        Select a well using tournament selection.
        
        Args:
            population: List of well configurations
            fitnesses: List of fitness scores
            
        Returns:
            Selected well configuration
        """
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
        return TerrainWell(
            population[winner_idx].x0,
            population[winner_idx].y0,
            population[winner_idx].depth,
            population[winner_idx].volume
        )
    
    def _crossover(self, parent1: TerrainWell, parent2: TerrainWell) -> TerrainWell:
        """
        Perform crossover between two parent wells.
        
        Args:
            parent1: First parent well
            parent2: Second parent well
            
        Returns:
            Child well with mixed parameters
        """
        # Create a new well with parameters from either parent
        if np.random.random() < 0.5:
            x = parent1.x0
        else:
            x = parent2.x0
            
        if np.random.random() < 0.5:
            y = parent1.y0
        else:
            y = parent2.y0
            
        if np.random.random() < 0.5:
            depth = parent1.depth
        else:
            depth = parent2.depth
            
        if np.random.random() < 0.5:
            volume = parent1.volume
        else:
            volume = parent2.volume
            
        return TerrainWell(x, y, depth, volume)
    
    def _mutate(self, well: TerrainWell) -> TerrainWell:
        """
        Mutate a well's parameters.
        
        Args:
            well: Well to mutate
            
        Returns:
            Mutated well
        """
        x = well.x0
        y = well.y0
        depth = well.depth
        volume = well.volume
        
        # Define neighborhood size for x and y mutations (as percentage of grid size)
        neighborhood_size = 0.1  # 10% of grid size
        
        if np.random.random() < self.mutation_rate:
            # Mutate x within neighborhood of original value
            x_min = max(0, x - self.grid_size * neighborhood_size)
            x_max = min(self.grid_size, x + self.grid_size * neighborhood_size)
            x = np.random.uniform(x_min, x_max)
            
        if np.random.random() < self.mutation_rate:
            # Mutate y within neighborhood of original value
            y_min = max(0, y - self.grid_size * neighborhood_size)
            y_max = min(self.grid_size, y + self.grid_size * neighborhood_size)
            y = np.random.uniform(y_min, y_max)
            
        if np.random.random() < self.mutation_rate:
            depth = np.random.uniform(5, 50)
            
        if np.random.random() < self.mutation_rate:
            volume = np.random.uniform(10, 500)
            
        return TerrainWell(x, y, depth, volume)
    
    def optimize_well_params(self) -> Tuple[float, float, float, float]:
        """
        Optimize well parameters using genetic algorithm.
        
        Returns:
            Tuple of (optimized_x, optimized_y, optimized_depth, optimized_volume)
        """
        # Initialize population
        population = self._initialize_population()
        best_well = None
        best_fitness = float('inf')
        
        # Evolve population
        for generation in range(self.num_generations):
            # Calculate fitness for all wells
            fitnesses = [self._calculate_fitness(well) for well in population]
            
            # Update best well
            min_fitness_idx = np.argmin(fitnesses)
            if fitnesses[min_fitness_idx] < best_fitness:
                best_fitness = fitnesses[min_fitness_idx]
                best_well = population[min_fitness_idx]
            
            # Create new population
            new_population = []
            
            # Elitism: keep best solutions
            elite_indices = np.argsort(fitnesses)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Fill rest of population with offspring
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        if best_well is None:
            # If optimization failed, return default values
            return 50, 50, 27.5, 255  # Default mid-range values
        
        return best_well.x0, best_well.y0, best_well.depth, best_well.volume