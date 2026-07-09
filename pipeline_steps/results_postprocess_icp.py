#!/usr/bin/env python3
"""Post-process transformed meshes with rigid ICP back-mapping."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_mesh_analysis.mesh_utils import apply_displacement, polydata_points_to_numpy, read_polydata, rigid_icp_transform, apply_transform


def align_icp(source_path, target_path):
    """Rigidly align a source mesh to a target mesh and return point displacement endpoints."""
    source_polydata = read_polydata(source_path)
    target_polydata = read_polydata(target_path)
    src_points = polydata_points_to_numpy(source_polydata)
    icp = rigid_icp_transform(source_polydata, target_polydata, max_iterations=100)
    final_points = polydata_points_to_numpy(apply_transform(source_polydata, icp))

    return src_points, final_points

def update_shape(filepath, displacement, output_filepath):
    """Apply a point displacement field and write the updated mesh."""
    apply_displacement(filepath, displacement, output_filepath)

if __name__ == '__main__':
    
    data_path = f'/file_path/'
    os.chdir(f'{data_path}')
    
    for leaflet in LEAFLETS:
        source_path = f"results/Transformed MD {leaflet} atrial surface.vtk"
        target_path = f"remesh/MD {leaflet} atrial surface.vtk"
        transformed_shape_path = f"results/Final strain MD {leaflet} atrial surface.vtk"
        src_points, final_points = align_icp(source_path, target_path)
        displacement = final_points - src_points
        update_shape(source_path, displacement, transformed_shape_path)

        source_path = f"results/Strain MS {leaflet} atrial surface.vtk"
        target_path = f"remesh/MS {leaflet} atrial surface.vtk"
        transformed_shape_path = f"results/Final strain MS {leaflet} atrial surface.vtk"

        src_points, final_points = align_icp(source_path, target_path)
        displacement = final_points - src_points
        update_shape(source_path, displacement, transformed_shape_path)
