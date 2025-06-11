import numpy as np
import plotly.graph_objects as go
import csv
import random

def random_boundary_surface(x_vals, y_vals, seed=0, offset_z=20.0, amplitude=8.0):
    """
    Creates a 2D array (Nx,Ny) that represents a wavy boundary,
    but with partial randomness. The result can intersect other surfaces
    if you define them with different seeds or offsets/amplitudes.

    offset_z: approximate 'mean' depth
    amplitude: how large the wave can go
    """
    np.random.seed(seed)
    Nx = len(x_vals)
    Ny = len(y_vals)

    # We'll build an array boundary_2d with shape (Nx, Ny).
    boundary_2d = np.zeros((Nx, Ny), dtype=float)

    # For each (ix, iy), define the boundary z value as:
    #  offset + amplitude*(some wave function) + random factor
    # We'll do a small random per (ix,iy)
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
    We'll store them in boundary1, boundary2, boundary3, but they won't be sorted
    the same way for all (x,y). So you do get physically "crossing" or "swapping" layers.

    Then we STILL do a final local sort to figure out which one is 'lowest' vs. 'middle' vs. 'highest'
    to map them into 4 layers with densities 2000, 2300, 2600, 2900.
    """
    # 1) Setup domain
    x_vals = np.linspace(0, 500, Nx)
    y_vals = np.linspace(0, 500, Ny)
    z_vals = np.linspace(0, 60, Nz)

    # 2) Generate 3 random boundary surfaces
    # Distinct seeds or offsets => they can cross each other
    boundary_1 = random_boundary_surface(x_vals, y_vals, seed=0, offset_z=15.0, amplitude=6.0)
    boundary_2 = random_boundary_surface(x_vals, y_vals, seed=1, offset_z=30.0, amplitude=8.0)
    boundary_3 = random_boundary_surface(x_vals, y_vals, seed=2, offset_z=45.0, amplitude=5.0)

    # 3) For each (ix,iy), we might get something like b1 > b2 or b3 < b2, etc.
    # We'll stack them and sort so we identify the "lowest", "middle", "top" boundary
    # at that cell
    stacked = np.stack([boundary_1, boundary_2, boundary_3], axis=-1)  # shape (Nx,Ny,3)
    sorted_boundaries = np.sort(stacked, axis=-1)  # sorts along the last dimension
    b1_2d = sorted_boundaries[:,:,0]  # lowest boundary
    b2_2d = sorted_boundaries[:,:,1]  # middle
    b3_2d = sorted_boundaries[:,:,2]  # highest

    # 4) Build 3D density array
    density_3D = np.zeros((Nx, Ny, Nz), dtype=float)
    for ix in range(Nx):
        for iy in range(Ny):
            z_low = b1_2d[ix, iy]
            z_mid = b2_2d[ix, iy]
            z_top = b3_2d[ix, iy]
            for iz, zv in enumerate(z_vals):
                if zv < z_low:
                    density_3D[ix, iy, iz] = 2000.0
                elif zv < z_mid:
                    density_3D[ix, iy, iz] = 2300.0
                elif zv < z_top:
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
        import csv
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
            print(f"   z={zv}, dens={dens}")

    # Flatten arrays for plotly
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

    # Custom discrete color scale
    # We'll map 2000 => red, 2300 => green, 2600 => blue, 2900 => orange
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
        title="Crossing Boundaries => 4 Layers + Random Geocores"
    )
    fig.show()

if __name__ == "__main__":
    main()
