from terrain_optimization import TerrainOptimizer

# Initialize the optimizer
optimizer = TerrainOptimizer(
    grid_size=50,
    z_max=100,
    num_cores=4,
    seed=42
)

# Initialize goal terrain
goal_terrain = None 