import numpy as np
import plotly.graph_objects as go
import csv
import random

def random_boundary_surface(x_vals, y_vals, seed=0, offset_z=20.0, amplitude=8.0):
    # Creates a 2D array (Nx,Ny) that represents a wavy boundary
    np.random.seed(seed)
    Nx = len(x_vals)
    Ny = len(y_vals)

    boundary_2d = np.zeros((Nx, Ny), dtype=float)

    for ix in range(Nx):
        for iy in range(Ny):
            rx = np.sin(0.01*x_vals[ix]) + 0.5*np.cos(0.02*y_vals[iy])
            rand_small = np.random.uniform(-1,1)*0.5  # small random
            base = offset_z + amplitude*rx + rand_small
            boundary_2d[ix, iy] = base

    return boundary_2d

def generate_crossing_density_map(Nx=70, Ny=70, Nz=60):
    """
    We define 3 surfaces that can cross or intersect each other in the domain.
    Instead of sorting them (which yields top/middle/bottom),
    we directly count how many boundaries lie above each voxel's z.
    That way, the layers genuinely intersect/lump together.
    """
    # Setup domain
    x_vals = np.linspace(0, 500, Nx)
    y_vals = np.linspace(0, 500, Ny)
    z_vals = np.linspace(0, 60, Nz)

    # Generate 3 random boundary surfaces, distinct seeds or offsets => they can cross each other
    boundary_1 = random_boundary_surface(x_vals, y_vals, seed=0, offset_z=15.0, amplitude=4.0)
    boundary_2 = random_boundary_surface(x_vals, y_vals, seed=1, offset_z=23.0, amplitude=30.0)
    boundary_3 = random_boundary_surface(x_vals, y_vals, seed=2, offset_z=45.0, amplitude=24.0)

    # Build 3D density array by counting how many boundaries are above z
    density_3D = np.zeros((Nx, Ny, Nz), dtype=float)
    for ix in range(Nx):
        for iy in range(Ny):
            for iz, zv in enumerate(z_vals):
                count_above = 0
                if boundary_1[ix, iy] > zv:
                    count_above += 1
                if boundary_2[ix, iy] > zv:
                    count_above += 1
                if boundary_3[ix, iy] > zv:
                    count_above += 1

                # Map count of boundaries-above to 4 density values
                if count_above == 3:
                    density_3D[ix, iy, iz] = 2000.0
                elif count_above == 2:
                    density_3D[ix, iy, iz] = 2300.0
                elif count_above == 1:
                    density_3D[ix, iy, iz] = 2600.0
                else:
                    density_3D[ix, iy, iz] = 2900.0

    return x_vals, y_vals, z_vals, density_3D

def sample_geocore(density_3D, x_vals, y_vals, z_vals, x_coord, y_coord):
    ix = np.argmin(np.abs(x_vals - x_coord))
    iy = np.argmin(np.abs(y_vals - y_coord))
    profile = []
    for iz, zv in enumerate(z_vals):
        dens = density_3D[ix, iy, iz]
        profile.append((zv, dens))
    return profile

def main():
    Nx, Ny, Nz = 70, 70, 60
    x_vals, y_vals, z_vals, density_3D = generate_crossing_density_map(Nx, Ny, Nz)

    # Save to CSV
    with open("density_map.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x","y","z","density"])
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    writer.writerow([
                        x_vals[ix],
                        y_vals[iy],
                        z_vals[iz],
                        density_3D[ix, iy, iz]
                    ])
    print("Saved density_map.csv")

    # Random geocores
    geocores = []
    for _ in range(3):
        gx = random.uniform(0, 500)
        gy = random.uniform(0, 500)
        geocores.append((gx, gy))

    # Print geocore data
    for i, (gx, gy) in enumerate(geocores):
        prof = sample_geocore(density_3D, x_vals, y_vals, z_vals, gx, gy)
        print(f"\nGeocore {i+1} at x={gx:.1f}, y={gy:.1f}")
        for (zv, dens) in prof:
            print(f"   z={zv:.1f}, dens={dens}")

    # Flatten arrays for Plotly
    X = []
    Y = []
    Z = []
    D = []
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                X.append(x_vals[ix])
                Y.append(y_vals[iy])
                Z.append(z_vals[iz])
                D.append(density_3D[ix, iy, iz])

    # Custom discrete color scale: 2000 => red, 2300 => green, 2600 => blue, 2900 => orange
    min_d, max_d = 2000.0, 2900.0
    custom_colorscale = [
        [0.00, "red"],    [0.11, "red"],
        [0.33, "green"],  [0.44, "green"],
        [0.66, "blue"],   [0.77, "blue"],
        [1.00, "orange"]
    ]

    fig = go.Figure(data=go.Isosurface(
        x=X, y=Y, z=Z,
        value=D,
        isomin=min_d,
        isomax=max_d,
        surface_count=5,  # 4 densities => ~3 boundaries
        colorscale=custom_colorscale,
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6
    ))

    # Plot random geocores
    for i, (gx, gy) in enumerate(geocores):
        profile = sample_geocore(density_3D, x_vals, y_vals, z_vals, gx, gy)
        z_line = [p[0] for p in profile]
        x_line = [gx]*len(profile)
        y_line = [gy]*len(profile)
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode='lines+markers',
            marker=dict(size=3, color='black'),
            line=dict(color='black', width=5),
            name=f"Geocore {i+1}"
        ))

    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Depth'),
        title="Intersecting Boundaries => 4 Layers + Random Geocores"
    )
    fig.show()

if __name__ == "__main__":
    main()
