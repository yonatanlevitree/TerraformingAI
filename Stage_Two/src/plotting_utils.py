#!/usr/bin/env python
"""
plotting_utils.py

Contains utility functions for building isosurfaces, flattening 3D arrays,
and creating animations (frames) in Plotly. 
Shared code for flattening data, building isosurfaces, creating frames/animations, etc.
"""

import numpy as np
import plotly.graph_objects as go

def flatten_isosurface_data(pressure_3D, x_vals, y_vals, z_vals,
                           min_p=0, max_p=1e5):
    """
    Flatten Nx*Ny*Nz into x,y,z + clamped pressure array for go.Isosurface(...).
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
                val = min(max_p, max(min_p, val))
                P.append(val)
    return X, Y, Z, P

def create_geocore_traces(geocores):
    """
    Creates a list of Scatter3d traces for each geocore, so they can be
    added to a Plotly figure. Each geocore is drawn as a vertical line of points.
    """
    traces = []
    for i, gc in enumerate(geocores):
        gx, gy = gc['x'], gc['y']
        z_line = gc['z_vals']
        sc3d = go.Scatter3d(
            x=[gx]*len(z_line),
            y=[gy]*len(z_line),
            z=z_line,
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(size=3, color='black'),
            name=f"Geocore {i+1}"
        )
        traces.append(sc3d)
    return traces
