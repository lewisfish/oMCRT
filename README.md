# oMCRT - Optix Monte Carlo Radiation Transfer.

Monte Carlo code in c++/CUDA that utilises Optix to enable GPU computation of nearest intersection.

Currently the code is setup with an isotropic source at the origin.
The photon packets then traverse a mesh of triangles and upon exit of the mesh are killed off.
The medium has an albedo of unity and scattering coefficient of $10cm^{-1}$.

Optix scaffolding code is based upon Ingo Wald's [Optix 7 course](https://github.com/ingowald/optix7course) in particular example's 4 and 7.
Actual MCRT code is contained in simulationPrograms.cu

Code also contains a basic renderer for viewing the meshes. Renderer shows the mesh with a solid colour and wireframe.

## Performance
  For a point source at the origin and uniform scattering with $\mu_s =10cm^{-1}$.
  * [Spot model](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/): ~155 MPhotons/s
  * [Nefertiti Bust](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/): ~ 80 Mphotons/s
  * [Standford Bunny](https://casual-effects.com/data/): ~95 MPhotons/s
  * [Happy Buddha](https://casual-effects.com/data/): ~70 MPhotons/s

## Screenshots
  #### Buddha model
  ![Budda](media/buddha.png)
  #### Spot model
  ![Spot](media/spot.png)
  #### Nefertiti bust
  ![Nefertiti bust](media/nefertiti.png)

## Dependencies
  * c++ compiler (17+)
  * Nvidia hpc SDK
  * Optix (version 7+)
  * tiny_obj_loader (included)
  * GDT (included)
  * CMake (3.17+)

## Todo
  - [x] Add absoprtion/photon weighting
  - [x] Add HG Phase function
  - [x] Add per mesh optical properties
  - [ ] Test current code.
  - [ ] Optimise
  - [ ] Add reflections/refractions
  - [ ] Add SDFs
  - [ ] Add wavelength variable optical properties
