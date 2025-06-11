#!/usr/bin/env python
"""
geocore_utils.py

Handles creation of random geocores with piecewise-layered density.
All logic related to generating geocores (piecewise layering, random seeds, etc.).
"""

import numpy as np
import random

def create_random_layered_density(z_max, z_points, layer_count=4):
    """
    Create a piecewise-linear density profile in [0, z_max].
    1) Random boundaries.
    2) Random densities [20..30].
    3) Linear interpolation for z_points.
    """
    internal_boundaries = np.sort(np.random.uniform(0, z_max, layer_count - 1))
    boundaries = np.concatenate(([0.0], internal_boundaries, [z_max]))
    densities = np.random.uniform(20, 30, layer_count + 1)

    z_vals_uniform = np.linspace(0, z_max, z_points)
    density_vals = np.zeros(z_points, dtype=float)

    for i in range(layer_count):
        z_min = boundaries[i]
        z_max_l = boundaries[i+1]
        d_min = densities[i]
        d_max = densities[i+1]
        idx = np.where((z_vals_uniform >= z_min) & (z_vals_uniform <= z_max_l))[0]
        if z_max_l>z_min:
            frac = (z_vals_uniform[idx] - z_min)/(z_max_l - z_min)
            density_vals[idx] = d_min + frac*(d_max - d_min)
        else:
            density_vals[idx] = d_min
    return z_vals_uniform, density_vals

def generate_random_geocores(num_cores=5, z_max=60, z_points=50, layer_count=4, seed=0):
    """
    Creates 'num_cores' random geocores in [0..500]x[0..500],
    each with piecewise-linear density from z=0..z_max.
    """
    random.seed(seed)  # ensure reproducibility if desired
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
