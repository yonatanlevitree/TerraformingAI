next steps (1/23):
1. geocore first, then interpolation
    - given geocore, generate interpolation visualization from there (DONE)
    - have more "dots" on the geocore (DONE)
        - make the intervals of the dots randomized densities, not necessarily to have each dot being a random density. this way it is closer to reality
    - find out the fidelity parameter of IDW to determine where exactly materials transition
2. pressure matrix (density times volume for the mass and compute pressure from there)
    - should density of layers matter there?
    - only make vertical assumptions, don't worry about pressure cones for now (DONE)


3. pressure assumption changed from vertical to pressure cone
- fix the injection iteration number bug
- slope of the pressure cone assumption
    - F_mu cos(theta), theta is the angle between vertical and other grids, so should be 45 degrees.
    aperture (slurry) should curl up when going flat for a while because of physics
    along the direction it accumulates, otherwise it either doesn't change or just transfer
    make sure the 5 cells below the current one cell does not exceed the one cell itself

# Changes Made to the Pressure Calculation Model --- Pressure Cone Procedure

We've implemented a more physically realistic pressure distribution model with the following improvements:

* **Replaced vertical-only pressure propagation** with a "pressure cone" model where each cell distributes pressure both directly downward and diagonally.

* **Added diagonal pressure distribution** where cells apply pressure not only to the cell directly below them, but also to the 4 diagonal neighbors below at a reduced factor of cos(45°) ≈ 0.7071.

* **Implemented pressure conservation** by normalizing the distributed pressure to ensure the total force is conserved across the cone pattern.

* **Separated force calculation from pressure propagation** to correctly handle layer-by-layer pressure distribution from top to bottom.

* **Improved the physics realism** by modeling how pressure disperses in a cone-like pattern in subsurface formations rather than being strictly vertical.

These changes should produce more realistic pressure distributions and potentially affect how the injected slurry flows during the simulation.



4. optimization, initial --> goal terrain
    time and money constrains like stage one


Apply the logic of system.py and tai.ipynb, and how these two files work together, on them. Specifically:

1. Use your understanding, generate the initial terrain surface from underground geocore interpolation
2. Generate a goal terrain that we want it to be after slurry injection (if you want you can use goal_terrain.npy from Stage_One, or you can use your own method)
3. Use the Optimization Process logic you just learned to change initial terrain to become goal terrain through slurry injection, with time and money constrains like stage one
4. Visualize how initial terrain is changing to become goal terrain at every step


next: Inject underground, how surface changes accordingly to reach the goal terrain