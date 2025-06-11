import numpy as np
import plotly.graph_objects as go

def generate_random_density_map(Nx=10, Ny=10, Nz=5, seed=42):
    """
    Creates a 3D grid of shape (Nx, Ny, Nz). 
    Each cell has a randomly assigned density (kg/m^3, for example).
    Returns:
      x_vals, y_vals, z_vals (1D arrays),
      density_3D (3D numpy array, shape [Nx, Ny, Nz])
    """
    np.random.seed(seed)  # fix seed for reproducibility

    # Let's define a small 2D horizontal domain, e.g. 0..100 in x and y
    x_vals = np.linspace(0, 100, Nx)
    y_vals = np.linspace(0, 100, Ny)

    # Suppose we define depth intervals up to 50 ft
    # For example, 5 layers: [0-10], [10-20], [20-30], [30-40], [40-50]
    z_layers = [0, 10, 20, 30, 40, 50]  # Nz=5 means 5 intervals => 6 boundary points

    # We'll store the actual 'representative' z for each layer, 
    # for the sake of plotting or referencing.
    # For instance, z_vals[k] might be the midpoint of each layer
    z_vals = []
    for k in range(Nz):
        midpoint = 0.5*(z_layers[k] + z_layers[k+1])
        z_vals.append(midpoint)
    z_vals = np.array(z_vals, dtype=float)

    # Build a 3D array for densities
    density_3D = np.zeros((Nx, Ny, Nz), dtype=float)

    # Option A: purely random densities (range e.g. 1500–3000 kg/m^3)
    # Option B: or a "layer approach" if you want. We'll do purely random here:
    for ix in range(Nx):
        for iy in range(Ny):
            # random density for each vertical layer
            # e.g. uniform(1500, 3000)
            density_3D[ix, iy, :] = np.random.uniform(1500, 3000, size=Nz)

    return x_vals, y_vals, z_vals, density_3D

def sample_geocore(density_3D, x_vals, y_vals, z_vals, x_coord, y_coord):
    """
    Given a geocore at (x_coord, y_coord), we find the nearest cell in (x_vals, y_vals),
    then extract the vertical density profile across all Nz layers.
    Returns a list of tuples: [(z0, density0), (z1, density1), ...].
    """
    Nx, Ny, Nz = density_3D.shape
    # nearest x index
    ix = np.argmin(np.abs(x_vals - x_coord))
    # nearest y index
    iy = np.argmin(np.abs(y_vals - y_coord))

    profile = []
    for iz, z_mid in enumerate(z_vals):
        profile.append((z_mid, density_3D[ix, iy, iz]))
    return profile

def main():
    # 1) Generate a random density map
    Nx, Ny, Nz = 10, 10, 5
    x_vals, y_vals, z_vals, density_3D = generate_random_density_map(Nx=Nx, Ny=Ny, Nz=Nz, seed=42)

    # 2) Pick a few geocore locations
    # e.g. geocore 1 at (x=20, y=30), geocore 2 at random
    geocore_locs = [(20, 30), (70, 80)]
    # or if you want random geocores:
    # np.random.seed(123)
    # geocore_locs = [(np.random.uniform(0,100), np.random.uniform(0,100)) for _ in range(3)]

    # 3) Sample each geocore
    all_geocores = []
    for (gx, gy) in geocore_locs:
        profile = sample_geocore(density_3D, x_vals, y_vals, z_vals, gx, gy)
        all_geocores.append(((gx, gy), profile))

    # 4) Print or store the geocore data
    for i, gc in enumerate(all_geocores):
        (gx, gy), profile = gc
        print(f"\nGeocore {i+1} at (x={gx}, y={gy}):")
        for (z_m, dens) in profile:
            print(f"  Depth midpoint = {z_m:.1f} ft, Density = {dens:.2f} kg/m^3")

    # 5) Visualize in 3D using Plotly
    # We'll create a scatter cloud: each point => (x, y, z) with color = density
    # We'll convert density_3D => big arrays X, Y, Z, D
    Xs = []
    Ys = []
    Zs = []
    Dvals = []
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                Xs.append(x_vals[ix])
                Ys.append(y_vals[iy])
                Zs.append(z_vals[iz])  # midpoint
                Dvals.append(density_3D[ix, iy, iz])

    # Create Plotly scatter3D
    # fig = go.Figure(data=[go.Scatter3d(
    #     x=Xs,
    #     y=Ys,
    #     z=Zs,
    #     mode='markers',
    #     marker=dict(
    #         size=4,
    #         color=Dvals,            # color by density
    #         colorscale='Viridis',
    #         showscale=True,
    #         colorbar=dict(title='Density (kg/m^3)')
    #     ),
    #     name='Random Density'
    # )])

    fig = go.Figure(data=go.Volume(
        x=np.array(Xs),
        y=np.array(Ys),
        z=np.array(Zs),
        value=np.array(Dvals),
        isomin=1500,
        isomax=3000,
        opacity=0.1,
        surface_count=15,
        colorscale='Viridis'
    ))
    # fig.show()


    # Add geocore markers as well
    # We'll show each geocore as a small vertical line or set of points
    for i, gc in enumerate(all_geocores):
        (gx, gy), profile = gc
        # profile => list of (z_mid, dens)
        # Let's just highlight them with big red markers
        z_line = [p[0] for p in profile]
        d_line = [p[1] for p in profile]  # not strictly used in position, but could color by it
        x_line = [gx]*len(profile)
        y_line = [gy]*len(profile)
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode='markers+lines',
            marker=dict(
                size=5,
                color='red'
            ),
            line=dict(
                color='red',
                width=4
            ),
            name=f'Geocore {i+1}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (ft)',
            yaxis_title='Y (ft)',
            zaxis_title='Depth midpoint (ft)',
            aspectmode='cube'
        ),
        title='Random Density Map with Geocores'
    )

    fig.show()

if __name__ == "__main__":
    main()
