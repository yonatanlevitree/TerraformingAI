#!/usr/bin/env python
"""
interpolation_utils.py

IDW interpolation from geocores -> 3D density, plus
computing simple overburden pressure from a 3D density volume.
Methods for IDW interpolation, building 3D density from geocores, computing pressure.
"""

import numpy as np

def linear_interpolate_z(z_vals, dens_vals, z_query):
    """
    Performs linear interpolation on (z_vals, dens_vals) to get density at z_query.
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

def build_3D_from_geocores(geocores, x_vals, y_vals, z_vals, power=2.0):
    """
    Inverse Distance Weighting from multiple geocores -> 3D density array.
    Nx,Ny,Nz = len(x_vals), len(y_vals), len(z_vals).
    """
    Nx, Ny, Nz = len(x_vals), len(y_vals), len(z_vals)
    density_3D = np.zeros((Nx, Ny, Nz), dtype=float)
    eps = 1e-6

    core_coords = [(gc['x'], gc['y']) for gc in geocores]

    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            # horizontal distances -> IDW weights
            dists = []
            for (gx, gy) in core_coords:
                dist_xy = np.sqrt((x - gx)**2 + (y - gy)**2)
                dists.append(dist_xy)
            weights = [1.0/(dist**power + eps) for dist in dists]

            for iz, z in enumerate(z_vals):
                # For each geocore -> linear interpolate
                rho_list = []
                for gc, w in zip(geocores, weights):
                    dens_i = linear_interpolate_z(gc['z_vals'], gc['density_vals'], z)
                    rho_list.append(dens_i)
                # Weighted avg
                numerator = 0.0
                denominator = 0.0
                for w, dens_val in zip(weights, rho_list):
                    numerator += w*dens_val
                    denominator += w
                density_3D[ix, iy, iz] = numerator/denominator

    return density_3D

def build_3D_pressure(density_3D, z_vals):
    """
    Pressure calculation using a "pressure cone" model.
    Each cell applies pressure to the cell directly below it AND 
    to the 4 diagonally adjacent cells below at a cos(45°) factor.
    The total pressure distributed from each cell is conserved.
    """
    Nx, Ny, Nz = density_3D.shape
    pressure_3D = np.zeros((Nx, Ny, Nz), dtype=float)
    g = 9.81
    cos45 = np.cos(np.radians(45))  # cosine of 45 degrees ≈ 0.7071
    
    # Create an array to hold the force at each cell position
    force_3D = np.zeros((Nx, Ny, Nz), dtype=float)
    
    # First calculate the force (weight) of each cell based on density and height
    for iz in range(Nz):
        # Calculate dz for current layer
        if iz == 0 and Nz > 1:
            dz = z_vals[iz+1] - z_vals[iz]
        elif iz > 0:
            dz = z_vals[iz] - z_vals[iz-1]
        else:
            dz = 0
            
        for ix in range(Nx):
            for iy in range(Ny):
                # Current cell's own weight (force)
                force_3D[ix, iy, iz] = density_3D[ix, iy, iz] * g * dz
    
    # Now propagate pressure from top to bottom
    # The top layer pressure is just from its own weight
    pressure_3D[:, :, 0] = force_3D[:, :, 0]
    
    # Then for each layer, distribute pressure from above
    for iz in range(1, Nz):
        # For each cell in the current layer
        for ix in range(Nx):
            for iy in range(Ny):
                # Add the cell's own weight to its pressure
                pressure_3D[ix, iy, iz] += force_3D[ix, iy, iz]
                
        # Now for the layer above, distribute its pressure to this layer
        for ix in range(Nx):
            for iy in range(Ny):
                # Get the pressure from the cell above (previous layer)
                pressure_above = pressure_3D[ix, iy, iz-1]
                
                # Skip if there's no pressure to distribute
                if pressure_above <= 0:
                    continue
                    
                # Collect cells that will receive pressure (current layer)
                receiving_cells = []
                
                # Direct cell below (full pressure)
                receiving_cells.append((ix, iy, 1.0))
                
                # Four diagonal cells (at cos(45°) pressure)
                diagonals = [(ix+1, iy), (ix-1, iy), (ix, iy+1), (ix, iy-1)]
                for dx, dy in diagonals:
                    if 0 <= dx < Nx and 0 <= dy < Ny:
                        receiving_cells.append((dx, dy, cos45))
                
                # Calculate normalization factor to ensure conservation of force
                total_factor = 1.0 + sum(factor for _, _, factor in receiving_cells if factor == cos45)
                norm_factor = 1.0 / total_factor if total_factor > 0 else 0
                
                # Distribute the pressure to receiving cells
                for rx, ry, factor in receiving_cells:
                    pressure_3D[rx, ry, iz] += pressure_above * factor * norm_factor
    
    return pressure_3D
