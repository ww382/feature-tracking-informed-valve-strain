# A Geometric Feature Tracking Approach for Noninvasive Patient-Specific Estimation of Leaflet Strain from 3D Images of Heart Valves

This repository contains the data and code for the paper [W. Wu, M. Daemer, J. A. Weiss, A. M. Pouch, & M. A. Jolley. A Geometric Feature Tracking Approach for Noninvasive Patient-Specific Estimation of Leaflet Strain from 3D Images of Heart Valves. Annals of Biomedical Engineering, 2026.](https://arxiv.org/pdf/2510.06578).

## Data

All example data are in the folder [example](example).

- [example/raw_surface](example/raw_surface) contains the raw MD and MS leaflet surface meshes.
- [example/mean_shape](example/mean_shape) contains the reference mean-shape meshes.
- `example/remesh`, `example/center_mesh`, and `example/results` are generated when the pipeline runs.

The example dataset contains tricuspid valve leaflet meshes from a patient with hypoplastic left heart syndrome. File names indicate both the cardiac frame (`MD` or `MS`) and leaflet (`anterior`, `posterior`, or `septal`); for example, `MD anterior atrial surface.vtk` is the anterior leaflet surface mesh at the MD frame.

## Code

The runnable pipeline steps are in the folder [pipeline_steps](pipeline_steps). Shared surface mesh analysis code is in [surface_mesh_analysis](surface_mesh_analysis).

The code depends on `numpy`, `vtk`, `pyvista`, `pyacvd`, `pycpd`, and `matplotlib`.

`surface_mesh_analysis` is installable on its own, for use from other code:

```bash
pip install git+https://github.com/ww382/feature-tracking-informed-valve-strain
```

It needs only `numpy`, `vtk` and `pycpd`. The strain calculation has two entry points - one
that reads and writes files, and one that takes the meshes in memory:

```python
from surface_mesh_analysis.shell_strain import (
    calculate_large_deformation_strain,              # paths in, file out (optional)
    calculate_large_deformation_strain_from_polydata,  # vtkPolyData in, vtkPolyData out
)
```

The pipeline steps and the example data are not part of the installed package.

- [Resample raw surfaces](pipeline_steps/resample_points.py)
- [Center meshes](pipeline_steps/center_mesh.py)
- [Check mesh orientation](pipeline_steps/rotate_mesh.py)
- [Align mean shapes](pipeline_steps/align_mean_shape.py)
- [Register leaflets and compute strain](pipeline_steps/leaflet_registration.py)
- [Post-process results with ICP](pipeline_steps/results_postprocess_icp.py)

To run the full example pipeline:

```bash
python3 main.py
```

Pipeline order, case path, frame labels, and leaflet labels are configured in [main.py](main.py).

## Cite this work

If you use this data or code for academic research, please cite the associated paper. Page and DOI details will be added when available.

```
@article{wu2026featuretracking,
  author  = {Wensi Wu and Matthew Daemer and Jeffrey A. Weiss and Alison M. Pouch and Matthew A. Jolley},
  title   = {A Geometric Feature Tracking Approach for Noninvasive Patient-Specific Estimation of Leaflet Strain from 3D Images of Heart Valves}, 
  journal = {Annals of Biomedical Engineering},
  year    = {2026},
  note    = {In press}
}
```

## Questions

For questions about the data or code, open an issue in the GitHub "Issues" section.
