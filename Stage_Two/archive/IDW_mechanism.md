

Takes five vertical "drillsticks" (or geocores), each at a random \((x,y)\) location with densities specified at uniformly spaced depths from \(0\) to a chosen maximum depth, and interpolates their density values across a 3D volume for visualization.

---

## Generating Random Geocores

We first create five geocores, each defined by:

- **\((x,y)\)**: random coordinates in the range \([0, 500]\times[0, 500]\).  
- A **vertical axis** from \(z=0\) to \(z=z_\text{max}\), uniformly sampled into `z_points` segments.
- **Random densities** at each of these discrete depth points, simulating measured or observed data.

```python
def generate_random_geocores(num_cores=5, z_max=60, z_points=10):
    z_vals_uniform = np.linspace(0, z_max, z_points)
    geocores = []
    for _ in range(num_cores):
        gx = random.uniform(0, 500)
        gy = random.uniform(0, 500)
        density_vals = np.random.uniform(2000, 3000, z_points)
        geocores.append({
            'x': gx,
            'y': gy,
            'z_vals': z_vals_uniform,
            'density_vals': density_vals
        })
    return geocores
```

## Building the 3D Volume via IDW

This step constructs a volumetric density model by **Inverse Distance Weighting** (IDW) each geocore’s density in the *horizontal plane* and linearly interpolating *vertically* within each geocore.

1. **3D Grid**  
   - We define a regular grid in \(x\) (0 to 500), \(y\) (0 to 500), and \(z\) (0 to `z_max`)—with `Nx`, `Ny`, and `Nz` subdivisions, respectively.

2. **Horizontal Weight Computation**  
   - For each \((x,y)\) in the grid, compute the horizontal distance to each geocore \((g_x, g_y)\).  
   - Define IDW weights as 
     \[
       w_i = \frac{1}{(\text{distance}_i^\text{power} + \varepsilon)},
     \]
     where \(\varepsilon\) is a small value (e.g.\ \(10^{-6}\)) to avoid division by zero.

3. **Vertical Linear Interpolation**  
   - Each geocore has discrete density values at uniform depths \(\{z_0, z_1, \dots, z_{n}\}\).  
   - For a given \((x,y,z)\) in the grid, we find the density at that \(z\) for each geocore by linearly interpolating its piecewise depth–density data.

4. **Weighted Average**  
   - Combine all geocores’ densities with the IDW weights:
     \[
       \rho(x,y,z) \;=\; \frac{\displaystyle \sum_i w_i \,\rho_i(z)}
                            {\displaystyle \sum_i w_i}.
     \]

Below is the relevant code snippet:

```python
def build_3D_from_geocores(geocores, Nx=70, Ny=70, Nz=60, z_max=60.0, power=2.0):
    """
    Create a 3D density volume from the given list of geocores via simple IDW.
    Nx, Ny, Nz define the output grid. 'power=2.0' => inverse distance squared.
    """
    x_vals = np.linspace(0, 500, Nx)
    y_vals = np.linspace(0, 500, Ny)
    z_vals = np.linspace(0, z_max, Nz)
    
    density_3D = np.zeros((Nx, Ny, Nz), dtype=float)
    eps = 1e-6  # tiny offset for stability

    # Pre-extract geocore (x, y) for faster distance computations
    core_coords = [(gc['x'], gc['y']) for gc in geocores]

    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            # 1) Horizontal distance -> IDW weights
            dists = []
            for (gx, gy) in core_coords:
                dist_xy = np.sqrt((x - gx)**2 + (y - gy)**2)
                dists.append(dist_xy)
            weights = [1.0 / (dist_xy**power + eps) for dist_xy in dists]

            for iz, z in enumerate(z_vals):
                # 2) Get each geocore's density at this z
                rho_list = []
                for gc in geocores:
                    rho_i = linear_interpolate_z(gc['z_vals'], gc['density_vals'], z)
                    rho_list.append(rho_i)

                # 3) Weighted average
                num_sum = 0.0
                den_sum = 0.0
                for w, rho_i in zip(weights, rho_list):
                    num_sum += w * rho_i
                    den_sum += w

                density_3D[ix, iy, iz] = num_sum / den_sum

    return x_vals, y_vals, z_vals, density_3D

