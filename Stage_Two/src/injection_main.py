#!/usr/bin/env python
"""
injection_main.py

- Loads pressure_map.csv (coarse 3D pressure).
- Runs BFS injection simulation with cost favoring horizontal moves at depth.
- Updates pressure near newly filled cells.
- Builds a single Plotly animation with frames for each iteration.
- Also plots the same 5 geocores on top of the isosurface.

The main script that uses modules geocore_utils.py, injection_utils.py,
and plotting_utils.py to run BFS injection, update pressure, and produce an animated figure.
"""

import numpy as np
import random
import plotly.graph_objects as go
import time

from geocore_utils import generate_random_geocores
from injection_utils import load_pressure_from_csv, inject_slurry, update_pressure
from plotting_utils import flatten_isosurface_data, create_geocore_traces

def main():
    # 1) Generate the same 5 geocores
    seed_for_geocores = 9
    geocores = generate_random_geocores(
        num_cores=5, z_max=60, z_points=50, layer_count=4, seed=seed_for_geocores
    )

    # 2) Domain used in geocore_main
    Nx, Ny, Nz = 70, 70, 60
    x_vals = np.linspace(0, 500, Nx)
    y_vals = np.linspace(0, 500, Ny)
    z_vals = np.linspace(0, 60, Nz)

    # 3) Load coarse pressure from CSV
    csv_pressure = "pressure_map.csv"
    pressure_3D = load_pressure_from_csv(csv_pressure, Nx, Ny, Nz, x_vals, y_vals, z_vals)
    # start a timer here
    start_time = time.perf_counter()  # Get current time

    # 4) BFS injection loop
    # We'll store snapshots after each iteration
    slurry_3D = np.zeros((Nx, Ny, Nz), dtype=float)
    pressure_snaps = []
    inj_loc = (35, 35, 40)

    # Different injection volume and step configurations:
    # Current settings:
    #   - 7500.0 total volume spread over 60 steps
    #   - 600 steps is a good balance between a smooth injection and a reasonable number of steps
    #   - Results in smaller injections of 125 volume per step
    #   - Step 4 should be at 500 volume, check consistency with vol = 500 steps = 1,
    #                                                            vol = 500 steps = 20
    #
    # Alternative configurations (consistency check):
    #   - 7500.0 volume over 20 steps (375.0 per step)
    #   - 375.0 volume in a single step
    #
    # Larger step counts result in more gradual injections
    # while smaller counts create more abrupt changes
    total_vol = 7500.0
    steps = 10
    # total_vol = 7500.0
    # steps = 20
    # total_vol = 375.0
    # steps = 1
    vol_per_step = total_vol / steps
    total_leftover = total_vol

    for step in range(1, steps+1):
        # On each iteration, we attempt to inject up to min(total_leftover, vol_per_step)
        amt_this_step = min(total_leftover, vol_per_step)

        old_slurry = slurry_3D.copy()
        old_pressure = pressure_3D.copy()

        # BFS tries to place 'amt_this_step'
        placed = inject_slurry(
            slurry_3D,
            pressure_3D,
            amt_this_step,
            inj_loc,
            cell_volume=1.0
        )

        # Now we subtract from total_leftover only what BFS actually placed
        total_leftover -= placed
        if total_leftover < 1e-9:
            total_leftover = 0.0

        # Then update pressure
        pressure_3D = update_pressure(old_pressure, old_slurry, slurry_3D)

        # Save snapshot
        pressure_snaps.append(pressure_3D.copy())

        print(f"Iteration {step}: tried {amt_this_step}, placed {placed}, leftover => {total_leftover}")

    # end the timer here
    end_time = time.perf_counter()  # Get time after algorithm code block
    elapsed_time = end_time - start_time
    print(f"Algorithm completion time taken: {elapsed_time} seconds")

    # 5) Build frames for animation
    frames = []
    geocore_traces = create_geocore_traces(geocores)
    min_p, max_p = 0, 17000
    surface_count = 6
    colorscale = 'Viridis'

    for idx, press_3D in enumerate(pressure_snaps):
        X, Y, Z, Pvals = flatten_isosurface_data(press_3D, x_vals, y_vals, z_vals,
                                                 min_p=min_p, max_p=max_p)
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
        frame_data = [iso_trace] + geocore_traces
        frames.append(go.Frame(data=frame_data, name=f"frame{idx}"))

    # 6) Figure w/ frames
    fig = go.Figure(data=frames[0].data, frames=frames)
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
                x=0.1,
                y=0.05,
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
                currentvalue={"prefix":"Iteration: "},
                pad={"t":50},
                steps=[
                    dict(
                        label=str(i),
                        method="animate",
                        args=[
                            [f"frame{i}"],
                            dict(
                                mode='immediate',
                                frame=dict(duration=300, redraw=True),
                                transition=dict(duration=300)
                            )
                        ]
                    )
                    for i in range(len(frames))
                ]
            )
        ]
    )

    fig.show()
    print("Done. Single animated figure for BFS injection steps.")
    end_time2 = time.perf_counter()  # Get time after code block
    elapsed_time2 = end_time2 - start_time
    print(f"Visulization time taken: {elapsed_time2} seconds")

if __name__=="__main__":
    main()
