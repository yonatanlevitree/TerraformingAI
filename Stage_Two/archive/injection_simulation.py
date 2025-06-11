import numpy as np
import csv
import heapq
import plotly.graph_objects as go
import random

##############################################################################
# 1) Generate 5 geocores (same logic as in geocore_to_interpolation).
#    We'll set a fixed random seed => same 5 geocores every run.
##############################################################################

def create_random_layered_density(z_max, z_points, layer_count=4):
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
            frac = (z_vals_uniform[idx] - z_min) / (z_max_layer - z_min)
            density_vals[idx] = d_min + frac * (d_max - d_min)
        else:
            density_vals[idx] = d_min
    return z_vals_uniform, density_vals

def generate_random_geocores(num_cores=5, z_max=60, z_points=50, layer_count=4):
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

##############################################################################
# 2) Utility Functions: load coarse-grid pressure, BFS injection w/ horizontal cost,
#    update pressure
##############################################################################

def load_pressure_from_csv(csv_file, Nx, Ny, Nz, x_vals, y_vals, z_vals):
    """
    Reads CSV columns [x,y,z,pressure], finds the nearest (ix,iy,iz) for each row,
    stores pressure in pressure_3D.
    """
    pressure_3D = np.zeros((Nx, Ny, Nz), dtype=float)

    def find_index(arr, val):
        idx = np.searchsorted(arr, val)
        if idx >= len(arr):
            idx = len(arr) - 1
        if idx > 0 and idx < len(arr):
            if abs(arr[idx - 1] - val) < abs(arr[idx] - val):
                idx = idx - 1
        return idx

    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_f = float(row['x'])
            y_f = float(row['y'])
            z_f = float(row['z'])
            p_f = float(row['pressure'])
            ix = find_index(x_vals, x_f)
            iy = find_index(y_vals, y_f)
            iz = find_index(z_vals, z_f)
            pressure_3D[ix, iy, iz] = p_f

    return pressure_3D

def neighbors_6(nx, ny, nz, ix, iy, iz):
    """6-connected neighbors within domain bounds."""
    for (dx, dy, dz) in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        x2 = ix + dx
        y2 = iy + dy
        z2 = iz + dz
        if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
            yield x2, y2, z2

def inject_slurry(slurry_3D, pressure_3D, inj_volume, inj_loc, cell_volume=1.0):
    """
    BFS/min-heap approach with "horizontal moves" cost = p_nbr * 1/(z+1)^2,
    vertical moves cost = p_nbr.
    """
    nx, ny, nz = slurry_3D.shape
    ix0, iy0, iz0 = inj_loc
    if not (0 <= ix0 < nx and 0 <= iy0 < ny and 0 <= iz0 < nz):
        print("Injection location out of domain!")
        return slurry_3D

    visited = np.zeros_like(slurry_3D, dtype=bool)
    heap = []

    p_start = pressure_3D[ix0, iy0, iz0]
    heapq.heappush(heap, (p_start, (ix0, iy0, iz0)))

    leftover = inj_volume
    while leftover > 1e-9 and heap:
        cost, (cx, cy, cz) = heapq.heappop(heap)
        if visited[cx, cy, cz]:
            continue
        visited[cx, cy, cz] = True

        if slurry_3D[cx, cy, cz] < 1.0:
            can_fill = 1.0 - slurry_3D[cx, cy, cz]
            capacity = can_fill * cell_volume
            if capacity <= leftover:
                slurry_3D[cx, cy, cz] = 1.0
                leftover -= capacity
            else:
                frac = leftover / cell_volume
                slurry_3D[cx, cy, cz] += frac
                leftover = 0.0

        for (nx_, ny_, nz_) in neighbors_6(nx, ny, nz, cx, cy, cz):
            if not visited[nx_, ny_, nz_] and slurry_3D[nx_, ny_, nz_] < 1.0:
                p_nbr = pressure_3D[nx_, ny_, nz_]
                if nz_ == cz:
                    # horizontal => cost = p_nbr*(1/(z+1)^2)
                    depth_factor = (nz_ + 1)**2
                    new_cost = p_nbr * (1.0 / depth_factor)
                else:
                    # vertical => cost = p_nbr
                    new_cost = p_nbr

                heapq.heappush(heap, (new_cost, (nx_, ny_, nz_)))

    return slurry_3D

def update_pressure(old_pressure, old_slurry, new_slurry, alpha=5e4):
    """
    Bump pressure in neighbors of newly filled cells.
    """
    nx, ny, nz = old_pressure.shape
    new_p = old_pressure.copy()
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                delta = new_slurry[ix, iy, iz] - old_slurry[ix, iy, iz]
                if delta > 1e-9:
                    for (x2, y2, z2) in neighbors_6(nx, ny, nz, ix, iy, iz):
                        new_p[x2, y2, z2] += alpha * delta
    return new_p

##############################################################################
# 3) Building an Animated Figure with Frames
##############################################################################

def flatten_isosurface_data(pressure_3D, x_vals, y_vals, z_vals,
                           min_p=0, max_p=17000):
    """
    Flatten the Nx*Ny*Nz array into x,y,z coords plus clamped pressures.
    We'll pass these to go.Isosurface(...).
    """
    Nx, Ny, Nz = pressure_3D.shape
    X, Y, Z, P = [], [], [], []
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                X.append(x_vals[ix])
                Y.append(y_vals[iy])
                Z.append(z_vals[iz])
                val = pressure_3D[ix, iy, iz]
                # clamp
                if val < min_p:
                    val = min_p
                elif val > max_p:
                    val = max_p
                P.append(val)
    return X, Y, Z, P

def create_geocore_traces(geocores):
    """
    Returns a list of Scatter3d traces, one per geocore,
    so we can add them to frames or the base data.
    """
    traces = []
    for i, gc in enumerate(geocores):
        gx, gy = gc['x'], gc['y']
        z_line = gc['z_vals']
        trace = go.Scatter3d(
            x=[gx]*len(z_line),
            y=[gy]*len(z_line),
            z=z_line,
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(size=3, color='black'),
            name=f"Geocore {i+1}"
        )
        traces.append(trace)
    return traces

##############################################################################
# 4) Main => Injection steps => build frames => single animated figure
##############################################################################

def main():
    # 1) Fixed seed => same 5 geocores
    random.seed(9)
    geocores = generate_random_geocores(num_cores=5, z_max=60, z_points=50, layer_count=4)

    # 2) Domain from geocore interpolation
    Nx, Ny, Nz = 70, 70, 60
    x_vals = np.linspace(0, 500, Nx)
    y_vals = np.linspace(0, 500, Ny)
    z_vals = np.linspace(0, 60, Nz)

    # 3) Load the coarse-grid pressure
    pressure_3D = load_pressure_from_csv("pressure_map.csv", Nx, Ny, Nz, x_vals, y_vals, z_vals)
    
    # 4) We'll store snapshots of pressure after each step
    pressure_snapshots = []

    # Also store the final slurry_3D after each iteration if you like, but not strictly needed for the isosurface.
    slurry_3D = np.zeros((Nx, Ny, Nz), dtype=float)

    # 5) Injection parameters
    total_vol = 5000.0
    steps = 6
    vol_per_step = total_vol / steps
    injection_cell = (Nx//2, Ny//2, 30)
    cell_vol = 1.0

    # 6) For iteration=0, store the initial pressure
    pressure_snapshots.append(pressure_3D.copy())

    # 7) BFS injection loop
    for step in range(1, steps+1):
        old_slurry = slurry_3D.copy()
        old_pressure = pressure_3D.copy()

        # BFS injection
        slurry_3D = inject_slurry(slurry_3D, pressure_3D, vol_per_step, injection_cell,
                                  cell_volume=cell_vol)

        # Update pressure
        pressure_3D = update_pressure(old_pressure, old_slurry, slurry_3D, alpha=5e4)

        # Store snapshot
        pressure_snapshots.append(pressure_3D.copy())

    # Now we have 'steps+1' snapshots (including initial at index 0).

    # 8) Build an animated figure with frames
    #    - Each frame => one isosurface trace + geocore lines

    min_p, max_p = 0, 17000
    surface_count = 6
    colorscale = 'Viridis'

    frames = []
    # We can build geocoreTraces once: they do not change
    geocoreTraces = create_geocore_traces(geocores)

    for frame_idx, P_3D in enumerate(pressure_snapshots):
        # Flatten for isosurface
        X, Y, Z, Pvals = flatten_isosurface_data(P_3D, x_vals, y_vals, z_vals,
                                                 min_p=min_p, max_p=max_p)
        # Build isosurface trace
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
            name=f"Pressure Step {frame_idx}"
        )
        # Combine isosurface + geocores
        data_traces = [iso_trace] + geocoreTraces
        frames.append(go.Frame(data=data_traces, name=f"frame{frame_idx}"))

    # The initial figure data => frames[0].data
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    # Add layout
    fig.update_layout(
    title="Animated 3D Pressure Isosurface Over Injection Steps",
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Depth (m)',
        zaxis=dict(autorange='reversed')
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            # Move the button downward, near the slider
            x=0.1,  # horizontal position in figure coords (0..1)
            y=0.05, # vertical position in figure coords (0..1)
            xanchor="left",
            yanchor="bottom",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=1000, redraw=True),
                            transition=dict(duration=500),
                            fromcurrent=True,
                            mode='immediate'
                        )
                    ]
                )
            ]
        )
    ],
    sliders=[
        dict(
            active=0,
            currentvalue={"prefix": "Iteration: "},
            pad={"t": 50},
            steps=[
                dict(
                    label=str(k),
                    method="animate",
                    args=[
                        [f"frame{k}"],
                        dict(
                            mode="immediate",
                            frame=dict(duration=300, redraw=True),
                            transition=dict(duration=300)
                        )
                    ]
                )
                for k in range(len(frames))
            ]
        )
    ]
)


    fig.show()

    print("Done. This single figure shows all injection steps as an animation.")

if __name__ == "__main__":
    main()
