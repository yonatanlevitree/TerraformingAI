import numpy as np
import random
import csv
import plotly.graph_objects as go

# -----------------------------------------------------
# A) Generating piecewise-layers in geocores
# -----------------------------------------------------

def create_random_layered_density(z_max, z_points, layer_count=4):
    """
    Generate a piecewise-linear density profile in [0..z_max].
    1) Randomly choose (layer_count - 1) internal boundary depths.
    2) Assign random density at each boundary (range ~ 20..30).
    3) Interpolate linearly for z_points.
    """
    internal_boundaries = np.sort(np.random.uniform(0, z_max, layer_count - 1))
    boundaries = np.concatenate(([0.0], internal_boundaries, [z_max]))
    
    densities = np.random.uniform(20, 30, layer_count + 1)
    
    z_vals_uniform = np.linspace(0, z_max, z_points)
    density_vals = np.zeros(z_points, dtype=float)
    
    for i in range(layer_count):
        z_min = boundaries[i]
        z_max_layer = boundaries[i+1]
        d_min = densities[i]
        d_max = densities[i+1]
        idx = np.where((z_vals_uniform >= z_min) & (z_vals_uniform <= z_max_layer))[0]
        
        if z_max_layer > z_min:
            frac = (z_vals_uniform[idx] - z_min)/(z_max_layer - z_min)
            density_vals[idx] = d_min + frac*(d_max - d_min)
        else:
            density_vals[idx] = d_min
    
    return z_vals_uniform, density_vals

def generate_random_geocores(num_cores=5, z_max=60, z_points=50, layer_count=4):
    """
    Create multiple geocores with piecewise-linear random densities.
    """
    geocores = []
    for _ in range(num_cores):
        gx = random.uniform(0, 500)
        gy = random.uniform(0, 500)
        z_vals, dens_vals = create_random_layered_density(z_max, z_points, layer_count)
        geocores.append({
            'x': gx,
            'y': gy,
            'z_vals': z_vals,
            'density_vals': dens_vals
        })
    return geocores

def linear_interpolate_z(z_vals, dens_vals, z_query):
    """
    Simple linear interpolation given (z_vals, dens_vals).
    """
    if z_query <= z_vals[0]:
        return dens_vals[0]
    if z_query >= z_vals[-1]:
        return dens_vals[-1]

    i = np.searchsorted(z_vals, z_query) - 1
    z0, z1 = z_vals[i], z_vals[i+1]
    d0, d1 = dens_vals[i], dens_vals[i+1]
    frac = (z_query - z0)/(z1 - z0)
    return d0 + frac*(d1 - d0)

# -----------------------------------------------------
# B) 3D Density via IDW
# -----------------------------------------------------

def build_3D_from_geocores(geocores, x_vals, y_vals, z_vals, power=2.0):
    """
    IDW interpolation to build density_3D over the grid [len(x_vals), len(y_vals), len(z_vals)].
    """
    Nx, Ny, Nz = len(x_vals), len(y_vals), len(z_vals)
    density_3D = np.zeros((Nx, Ny, Nz), dtype=float)
    eps = 1e-6

    # Precompute geocore coords
    core_coords = [(gc['x'], gc['y']) for gc in geocores]

    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            # Horizontal distances -> IDW weights
            dists = []
            for (gx, gy) in core_coords:
                dist_xy = np.sqrt((x - gx)**2 + (y - gy)**2)
                dists.append(dist_xy)
            weights = [1.0/(dist**power + eps) for dist in dists]

            for iz, z in enumerate(z_vals):
                # For each geocore, get density at this z
                rho_list = []
                for gc in geocores:
                    rho_i = linear_interpolate_z(gc['z_vals'], gc['density_vals'], z)
                    rho_list.append(rho_i)

                # Weighted average
                num_sum = 0.0
                den_sum = 0.0
                for w, rho_i in zip(weights, rho_list):
                    num_sum += w * rho_i
                    den_sum += w
                density_3D[ix, iy, iz] = num_sum / den_sum

    return density_3D

# -----------------------------------------------------
# C) 3D Pressure Calculation
# -----------------------------------------------------

def build_3D_pressure(density_3D, z_vals):
    """
    Overburden pressure at each cell: sum rho*g*dz from top to current iz.
    """
    Nx, Ny, Nz = density_3D.shape
    pressure_3D = np.zeros((Nx, Ny, Nz), dtype=float)

    g = 9.81  # m/s^2

    for ix in range(Nx):
        for iy in range(Ny):
            cum_pressure = 0.0
            for iz in range(Nz):
                if iz == 0:
                    dz = (z_vals[iz+1] - z_vals[iz]) if (iz+1 < Nz) else 0
                else:
                    dz = z_vals[iz] - z_vals[iz-1]

                rho_here = density_3D[ix, iy, iz]
                cum_pressure += rho_here * g * dz
                pressure_3D[ix, iy, iz] = cum_pressure

    return pressure_3D

# -----------------------------------------------------
# D) MAIN
# -----------------------------------------------------

def main():
    random.seed(9)
    # STEP 1) Create random geocores
    geocores = generate_random_geocores(num_cores=5, z_max=60, z_points=50, layer_count=4)

    # STEP 2) Build a "coarse" grid (70×70×60) for quick isosurface visualization
    Nx, Ny, Nz = 70, 70, 60
    x_vals_coarse = np.linspace(0, 500, Nx)
    y_vals_coarse = np.linspace(0, 500, Ny)
    z_vals_coarse = np.linspace(0, 60, Nz)
    
    # 2a) Compute density & pressure on coarse grid
    density_3D_coarse = build_3D_from_geocores(geocores, x_vals_coarse, y_vals_coarse, z_vals_coarse, power=2.0)
    pressure_3D_coarse = build_3D_pressure(density_3D_coarse, z_vals_coarse)

    # 2b) Save these "coarse" results to CSV
    with open("density_map.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x","y","z","density"])
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    writer.writerow([
                        x_vals_coarse[ix],
                        y_vals_coarse[iy],
                        z_vals_coarse[iz],
                        density_3D_coarse[ix, iy, iz]
                    ])

    with open("pressure_map.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x","y","z","pressure"])
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    writer.writerow([
                        x_vals_coarse[ix],
                        y_vals_coarse[iy],
                        z_vals_coarse[iz],
                        pressure_3D_coarse[ix, iy, iz]
                    ])

    print("Saved density_map.csv and pressure_map.csv (coarse grid).")

    # STEP 3) Visualize the coarse results as an isosurface
    Xc, Yc, Zc = [], [], []
    D_dens, D_pres = [], []
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                Xc.append(x_vals_coarse[ix])
                Yc.append(y_vals_coarse[iy])
                Zc.append(z_vals_coarse[iz])
                D_dens.append(density_3D_coarse[ix, iy, iz])
                D_pres.append(pressure_3D_coarse[ix, iy, iz])

    fig_density = go.Figure(data=go.Isosurface(
        x=Xc, y=Yc, z=Zc,
        value=D_dens,
        isomin=20,
        isomax=30,
        surface_count=5,
        colorscale='Turbo',
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6
    ))
    fig_density.update_layout(
        title="3D Density Isosurface (Coarse Grid)",
        scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Depth (m)',
        zaxis=dict(autorange='reversed')  # <--- flip in the visualization so 0 on top
    )
    )

    fig_pressure = go.Figure(data=go.Isosurface(
        x=Xc, y=Yc, z=Zc,
        value=D_pres,
        surface_count=6,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6
    ))
    fig_pressure.update_layout(
        title="3D Pressure Isosurface (Coarse Grid)",
        scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Depth (m)',
        zaxis=dict(autorange='reversed')  # <--- flip in the visualization so 0 on top
    )
    )

    # Add geocore lines
    for i, gc in enumerate(geocores):
        gx, gy = gc['x'], gc['y']
        z_line = gc['z_vals']
        fig_density.add_trace(go.Scatter3d(
            x=[gx]*len(z_line),
            y=[gy]*len(z_line),
            z=z_line,
            mode='lines+markers',
            marker=dict(size=3, color='black'),
            line=dict(color='black', width=4),
            name=f"Geocore {i+1}"
        ))
        fig_pressure.add_trace(go.Scatter3d(
            x=[gx]*len(z_line),
            y=[gy]*len(z_line),
            z=z_line,
            mode='lines+markers',
            marker=dict(size=3, color='black'),
            line=dict(color='black', width=4),
            name=f"Geocore {i+1}"
        ))
    
    # Show coarse-grid figures
    fig_density.show()
    fig_pressure.show()

    # STEP 4) Optionally build a 1×1×1 m grid
    build_fine_grid = False  # set True if you want the big CSV (potentially huge)
    if build_fine_grid:
        print("Building 10x10x10 grid. This may be large...")

        x_vals_fine = np.arange(0, 501, 10)
        y_vals_fine = np.arange(0, 501, 10)
        z_vals_fine = np.arange(0, 61, 10)

        density_3D_fine = build_3D_from_geocores(geocores, x_vals_fine, y_vals_fine, z_vals_fine, power=2.0)
        pressure_3D_fine = build_3D_pressure(density_3D_fine, z_vals_fine)

        # Save the 1-meter results
        out_file = "pressure_3D_10m.csv"
        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x","y","z","pressure"])
            Nx_f, Ny_f, Nz_f = len(x_vals_fine), len(y_vals_fine), len(z_vals_fine)
            for ix in range(Nx_f):
                Xp = x_vals_fine[ix]
                for iy in range(Ny_f):
                    Yp = y_vals_fine[iy]
                    for iz in range(Nz_f):
                        Zp = z_vals_fine[iz]
                        pval = pressure_3D_fine[ix, iy, iz]
                        writer.writerow([Xp, Yp, Zp, pval])

        print(f"Saved 10x10x10 grid to {out_file}")

if __name__ == "__main__":
    main()
