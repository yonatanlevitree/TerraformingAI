#!/usr/bin/env python
"""
injection_utils.py

Contains functions for loading a 3D pressure map from CSV,
performing BFS injection with a custom cost for horizontal vs. vertical moves,
and updating pressure near newly filled cells.
Functions for loading pressure, performing BFS injection, updating pressure.
"""

import numpy as np
import csv
import heapq

def load_pressure_from_csv(csv_file, Nx, Ny, Nz, x_vals, y_vals, z_vals):
    """
    Reads columns [x,y,z,pressure] => fill pressure_3D with nearest-index approach.
    """
    pressure_3D = np.zeros((Nx, Ny, Nz), dtype=float)

    def find_index(arr, val):
        idx = np.searchsorted(arr, val)
        if idx>=len(arr):
            idx = len(arr)-1
        if idx>0 and idx<len(arr):
            if abs(arr[idx-1]-val)<abs(arr[idx]-val):
                idx=idx-1
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
    for (dx,dy,dz) in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        x2 = ix+dx
        y2 = iy+dy
        z2 = iz+dz
        if 0<=x2<nx and 0<=y2<ny and 0<=z2<nz:
            yield x2,y2,z2

def inject_slurry(slurry_3D, pressure_3D, inj_volume, inj_loc, cell_volume=1.0):
    """
    BFS/min-heap injection with cost logic:
      - horizontal => p_nbr / (z+1)^exponent
      - vertical   => p_nbr
    We pass in 'inj_volume' as the max volume BFS can place this iteration.
    
    Returns
    -------
    slurry_3D : ndarray
        The updated 3D slurry array
    placed_volume : float
        How much volume was actually placed in the domain.
    """
    # Make a copy of the input slurry to avoid modifying the original
    slurry_3D = slurry_3D.copy()
    
    # Get dimensions, with proper error handling
    if not isinstance(slurry_3D, np.ndarray) or slurry_3D.ndim != 3:
        print(f"Error: slurry_3D must be a 3D numpy array, got {type(slurry_3D)} with shape {getattr(slurry_3D, 'shape', 'unknown')}")
        # Return original array and 0 volume if input is invalid
        return slurry_3D, 0.0
        
    nx, ny, nz = slurry_3D.shape

    visited = np.zeros_like(slurry_3D, dtype=bool)
    heap = []

    # 1) Instead of pushing ONLY 'inj_loc', we push *all cells with slurry>0*
    #    so BFS can expand from the entire fluid region.
    for cx in range(nx):
        for cy in range(ny):
            for cz in range(nz):
                if slurry_3D[cx, cy, cz] > 1e-9:  # partial or full fluid
                    cost_val = pressure_3D[cx, cy, cz]
                    heapq.heappush(heap, (cost_val, (cx, cy, cz)))

    # If no slurry at all, push the injection cell
    if not heap:
        ix0, iy0, iz0 = inj_loc
        cost_inj = pressure_3D[ix0, iy0, iz0]
        heapq.heappush(heap, (cost_inj, (ix0, iy0, iz0)))

    leftover = inj_volume
    placed_volume = 0.0  # track how much BFS actually injected

    while leftover > 1e-9 and heap:
        cost, (cx, cy, cz) = heapq.heappop(heap)
        if visited[cx, cy, cz]:
            continue
        visited[cx, cy, cz] = True

        # fill fraction in the current cell if not fully filled
        if slurry_3D[cx, cy, cz] < 1.0:
            can_fill_fraction = 1.0 - slurry_3D[cx, cy, cz]
            cell_capacity = can_fill_fraction * cell_volume
            if cell_capacity <= leftover:
                # fill fully
                slurry_3D[cx, cy, cz] = 1.0
                leftover -= cell_capacity
                placed_volume += cell_capacity
            else:
                # partial fill
                frac = leftover / cell_volume
                slurry_3D[cx, cy, cz] += frac
                placed_volume += leftover
                leftover = 0.0  # we've placed all we can in this iteration

        # explore neighbors
        for (nx_, ny_, nz_) in neighbors_6(nx, ny, nz, cx, cy, cz):
            if not visited[nx_, ny_, nz_] and slurry_3D[nx_, ny_, nz_] < 1.0:
                p_nbr = pressure_3D[nx_, ny_, nz_]
                if nz_ == cz:
                    # horizontal => cost = p_nbr/(z+1)^exponent
                    exponent = 1.2
                    factor = (nz_ + 1)**exponent
                    new_cost = p_nbr / factor
                else:
                    # vertical => cost = p_nbr
                    new_cost = p_nbr
                heapq.heappush(heap, (new_cost, (nx_, ny_, nz_)))

    return slurry_3D, placed_volume


# def update_pressure(old_pressure, old_slurry, new_slurry, alpha=5e4):
#     """
#     For newly filled cells (delta_slurry>0), bump neighbor pressure by alpha*delta.
#     """
#     nx, ny, nz = old_pressure.shape
#     new_p = old_pressure.copy()
#     for ix in range(nx):
#         for iy in range(ny):
#             for iz in range(nz):
#                 delta = new_slurry[ix, iy, iz] - old_slurry[ix, iy, iz]
#                 if delta>1e-9:
#                     for (x2,y2,z2) in neighbors_6(nx,ny,nz,ix,iy,iz):
#                         new_p[x2,y2,z2]+= alpha*delta
#     return new_p

"""
Alternative option, depth-dependent pressure bump
Making the injected slurry less flat and more pocket-like
"""
def update_pressure(old_pressure, old_slurry, new_slurry):
    # Check if inputs are arrays with the expected dimensions
    if not isinstance(old_slurry, np.ndarray) or not isinstance(new_slurry, np.ndarray):
        print("Warning: slurry inputs to update_pressure are not arrays")
        return old_pressure.copy()
    
    if old_slurry.shape != old_pressure.shape or new_slurry.shape != old_pressure.shape:
        print(f"Warning: dimension mismatch in update_pressure. Shapes: old_pressure {old_pressure.shape}, old_slurry {getattr(old_slurry, 'shape', 'scalar')}, new_slurry {getattr(new_slurry, 'shape', 'scalar')}")
        return old_pressure.copy()
    
    nx, ny, nz = old_pressure.shape
    new_p = old_pressure.copy()
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # Access elements safely
                try:
                    delta = new_slurry[ix, iy, iz] - old_slurry[ix, iy, iz]
                except (IndexError, TypeError) as e:
                    print(f"Error accessing slurry at {ix},{iy},{iz}: {e}")
                    continue
                    
                if delta > 1e-9:
                    # Possibly scale alpha by z, so deeper zones get
                    # a stronger local pressure bump
                    alpha_local = 1e5 * (1 + iz*0.1)
                    for (x2,y2,z2) in neighbors_6(nx,ny,nz,ix,iy,iz):
                        new_p[x2,y2,z2] += alpha_local * delta
    return new_p

def compute_volume_used(new_slurry_3D, old_slurry_3D, cell_volume=1.0):
    """
    Compare old vs. new slurry fraction to see how many 'units' were added in BFS.
    If each cell can hold 1 volume unit at slurry=1.0, then
    volume_in_cell = fraction_in_cell * cell_volume.
    """
    delta_frac = new_slurry_3D - old_slurry_3D
    # sum all positive increases
    added_frac = np.sum(delta_frac[delta_frac>0])
    return added_frac * cell_volume

