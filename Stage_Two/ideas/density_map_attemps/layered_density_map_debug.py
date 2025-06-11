import numpy as np
import plotly.graph_objects as go

def boundary_surface_1(x, y):
    """
    Example boundary surface #1: moderate wave in x, y.
    We'll keep it shallower. (z ~ 15 +/- small variation)
    """
    return 15.0 + 3.0 * np.sin(0.015*x) + 2.0 * np.cos(0.02*y)

def boundary_surface_2(x, y):
    """
    Example boundary surface #2: deeper boundary, different wave pattern
    """
    return 30.0 + 5.0 * np.cos(0.01*x) - 3.0 * np.sin(0.015*y)

def boundary_surface_3(x, y):
    """
    Example boundary surface #3: the deepest boundary
    """
    return 45.0 + 4.0 * np.sin(0.01*x) + 3.0 * np.cos(0.01*y)

def generate_density_map(Nx=70, Ny=70, Nz=60):
    """
    Create a 3D grid of shape (Nx,Ny,Nz), with x in [0..500], y in [0..500],
    z in [0..60]. We'll define 3 boundary surfaces for a total of 4 layers:
       - Region 1: z < b1(x,y)         => density = 2000
       - Region 2: b1 < z < b2        => density = 2300
       - Region 3: b2 < z < b3        => density = 2600
       - Region 4: z > b3             => density = 2900
    """
    # 1) Define coordinate arrays
    x_vals = np.linspace(0, 500, Nx)
    y_vals = np.linspace(0, 500, Ny)
    z_vals = np.linspace(0, 60, Nz)

    # 2) Evaluate boundary surfaces on a 2D grid
    #    We'll shape them as (Nx,Ny), storing z-values
    X2d, Y2d = np.meshgrid(x_vals, y_vals, indexing='ij')
    b1_2d = boundary_surface_1(X2d, Y2d)  # shape Nx x Ny
    b2_2d = boundary_surface_2(X2d, Y2d)
    b3_2d = boundary_surface_3(X2d, Y2d)

    print("b1_2d range:", np.min(b1_2d), np.max(b1_2d))
    print("b2_2d range:", np.min(b2_2d), np.max(b2_2d))
    print("b3_2d range:", np.min(b3_2d), np.max(b3_2d))


    # If any boundary is out of order, we'll sort them so b1 < b2 < b3 per cell
    # That ensures consistent layering.
    # We'll do it by stacking and taking sorted along the last dimension
    stacked = np.stack([b1_2d, b2_2d, b3_2d], axis=-1)
    sorted_boundaries = np.sort(stacked, axis=-1)
    b1_2d = sorted_boundaries[:, :, 0]
    b2_2d = sorted_boundaries[:, :, 1]
    b3_2d = sorted_boundaries[:, :, 2]

    test_ix = Nx//2
    test_iy = Ny//2
    print("At center cell:", b1_2d[test_ix, test_iy], b2_2d[test_ix, test_iy], b3_2d[test_ix, test_iy])


    # 3) Build a 3D array of densities
    density_3D = np.zeros((Nx, Ny, Nz), dtype=float)

    for ix in range(Nx):
        for iy in range(Ny):
            z_b1 = b1_2d[ix, iy]
            z_b2 = b2_2d[ix, iy]
            z_b3 = b3_2d[ix, iy]
            for iz, zv in enumerate(z_vals):
                if zv < z_b1:
                    density_3D[ix, iy, iz] = 2000.0
                elif zv < z_b2:
                    density_3D[ix, iy, iz] = 2300.0
                elif zv < z_b3:
                    density_3D[ix, iy, iz] = 2600.0
                else:
                    density_3D[ix, iy, iz] = 2900.0

    test_ix = Nx//2
    test_iy = Ny//2
    dens_line = density_3D[test_ix, test_iy, :]
    print("Density profile at center cell =\n", dens_line)
    print("Final density_3D stats: min =", density_3D.min(), " max =", density_3D.max())


    return x_vals, y_vals, z_vals, density_3D

def sample_geocore(density_3D, x_vals, y_vals, z_vals, x_coord, y_coord):
    """
    Extract a vertical density profile at (x_coord, y_coord).
    We'll find the nearest grid cell in x,y, then build
    a list of (z, density) for all layers in z_vals.
    """
    Nx, Ny, Nz = density_3D.shape
    ix = np.argmin(np.abs(x_vals - x_coord))
    iy = np.argmin(np.abs(y_vals - y_coord))

    profile = []
    for iz, zv in enumerate(z_vals):
        dens = density_3D[ix, iy, iz]
        profile.append((zv, dens))
    return profile

def main():
    # 1) Generate the 4-layer density map
    Nx, Ny, Nz = 70, 70, 60
    x_vals, y_vals, z_vals, density_3D = generate_density_map(Nx, Ny, Nz)

    print("Debug Info:")
    print("  x range =", x_vals[0], "to", x_vals[-1])
    print("  y range =", y_vals[0], "to", y_vals[-1])
    print("  z range =", z_vals[0], "to", z_vals[-1])

    # 2) Flatten for Plotly
    X = []
    Y = []
    Z = []
    dens_vals = []
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                X.append(x_vals[ix])
                Y.append(y_vals[iy])
                Z.append(z_vals[iz])
                dens_vals.append(density_3D[ix, iy, iz])

    # 3) Create a custom discrete color scale for 4 distinct densities
    # We'll do 2000 => red, 2300 => green, 2600 => blue, 2900 => orange, for instance.
    # But isosurface uses continuous color mapping. We'll define the min->max = 2000->2900,
    # and place color stops at 2000, 2300, 2600, 2900. We'll normalize them:
    min_d, max_d = 2000.0, 2900.0
    # Normalized positions: 0 => 2000, 0.33 => 2300, 0.66 => 2600, 1.0 => 2900
    # We'll do piecewise linear in between
    custom_colorscale = [
        [0.00, "red"],
        [0.11, "red"],    # keep it red near 2000
        [0.33, "green"],
        [0.44, "green"],  # keep it green near 2300
        [0.66, "blue"],
        [0.77, "blue"],   # keep it near 2600
        [1.00, "orange"]  # near 2900
    ]

    # 4) Plot with Isosurface
    # We'll set isomin=2000, isomax=2900 and surface_count=3 => we should get surfaces around 2300,2600, ~ 2 boundaries, might see 3
    fig = go.Figure(data=go.Isosurface(
        x=X,
        y=Y,
        z=Z,
        value=dens_vals,
        isomin=2000,
        isomax=2900,
        surface_count=5,
        colorscale=custom_colorscale,
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.5
    ))

    # 5) Overlay 2 geocores
    geocore_locs = [(100, 100), (400, 200)]
    for i, (gx, gy) in enumerate(geocore_locs):
        profile = sample_geocore(density_3D, x_vals, y_vals, z_vals, gx, gy)
        # Build scatter lines
        z_line = [p[0] for p in profile]
        d_line = [p[1] for p in profile]  # not used in coords, but optional
        # x_line => all gx, y_line => all gy
        x_line = [gx]*len(profile)
        y_line = [gy]*len(profile)

        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode='lines+markers',
            line=dict(width=6, color='black'),
            marker=dict(size=3, color='black'),
            name=f'Geocore {i+1} at x={gx}, y={gy}'
        ))

    # 6) Layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Depth (m)',
            aspectmode='cube'
        ),
        title="Layered Regions with 4 Distinct Densities & Geocores"
    )

    fig.show()

if __name__ == "__main__":
    main()
