#!/usr/bin/env python
"""
geocore_main.py

A main script that:
  1) Generates random geocores (using geocore_utils).
  2) Builds 3D density & pressure on a coarse grid (using interpolation_utils).
  3) Saves to CSV.
  4) Visualizes them with Plotly isosurfaces + geocore lines.

The main script that uses modules geocore_utils.py, interpolation_utils.py, and 
plotting_utils.py to generate and visualize geocores, produce CSVs, etc.
"""

import numpy as np
import csv
import plotly.graph_objects as go
import random

from geocore_utils import generate_random_geocores
from interpolation_utils import build_3D_from_geocores, build_3D_pressure

def main():
    # 1) Generate geocores
    random_seed = 9
    geocores = generate_random_geocores(
        num_cores=5, z_max=60, z_points=50, layer_count=4, seed=random_seed
    )

    # 2) Coarse grid
    Nx, Ny, Nz = 70, 70, 60
    x_vals = np.linspace(0, 500, Nx)
    y_vals = np.linspace(0, 500, Ny)
    z_vals = np.linspace(0, 60, Nz)

    # 3) Build density & pressure
    density_3D = build_3D_from_geocores(geocores, x_vals, y_vals, z_vals, power=2.0)
    pressure_3D = build_3D_pressure(density_3D, z_vals)

    # 4) Save CSV
    with open("density_map.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["x","y","z","density"])
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    w.writerow([
                        x_vals[ix],
                        y_vals[iy],
                        z_vals[iz],
                        density_3D[ix, iy, iz]
                    ])
    with open("pressure_map.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["x","y","z","pressure"])
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    w.writerow([
                        x_vals[ix],
                        y_vals[iy],
                        z_vals[iz],
                        pressure_3D[ix, iy, iz]
                    ])
    print("Saved density_map.csv and pressure_map.csv.")

    # 5) Flatten arrays -> Plot isosurfaces
    Xs, Ys, Zs = [], [], []
    Ddens, Dpres = [], []
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                Xs.append(x_vals[ix])
                Ys.append(y_vals[iy])
                Zs.append(z_vals[iz])
                Ddens.append(density_3D[ix, iy, iz])
                Dpres.append(pressure_3D[ix, iy, iz])

    fig_dens = go.Figure(data=go.Isosurface(
        x=Xs, y=Ys, z=Zs,
        value=Ddens,
        isomin=20, # match the random density range
        isomax=30,
        surface_count=5,
        colorscale='Turbo',
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6
    ))
    fig_dens.update_layout(
        title="Coarse Grid - Density Isosurface",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Depth (m)',
            zaxis=dict(autorange='reversed')
        )
    )

    fig_press = go.Figure(data=go.Isosurface(
        x=Xs, y=Ys, z=Zs,
        value=Dpres,
        surface_count=6,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6
    ))
    fig_press.update_layout(
        title="Coarse Grid - Pressure Isosurface",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Depth (m)',
            zaxis=dict(autorange='reversed')
        )
    )

    # Add geocore lines
    for i, gc in enumerate(geocores):
        gx, gy = gc['x'], gc['y']
        z_line = gc['z_vals']
        fig_dens.add_trace(go.Scatter3d(
            x=[gx]*len(z_line),
            y=[gy]*len(z_line),
            z=z_line,
            mode='lines+markers',
            line=dict(color='black', width=4),
            marker=dict(size=3, color='black'),
            name=f"Geocore {i+1}"
        ))
        fig_press.add_trace(go.Scatter3d(
            x=[gx]*len(z_line),
            y=[gy]*len(z_line),
            z=z_line,
            mode='lines+markers',
            line=dict(color='black', width=4),
            marker=dict(size=3, color='black'),
            name=f"Geocore {i+1}"
        ))

    fig_dens.show()
    fig_press.show()

if __name__ == "__main__":
    main()
