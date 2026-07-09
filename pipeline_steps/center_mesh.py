"""Center remeshed surfaces and align their average normals to the Z-axis."""

import vtk
import os
import glob
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_mesh_analysis.mesh_utils import compute_average_cell_normal, center_polydata_at_origin, read_polydata, write_polydata


def align_mesh_to_z_axis(input_vtk, output_vtk):
    """Align a mesh normal to the nearest Z direction and center it at the origin."""
    polydata = read_polydata(input_vtk)

    avg_normal = compute_average_cell_normal(polydata)
    z_plus = np.array([0, 0, 1])
    z_minus = np.array([0, 0, -1])

    if np.dot(avg_normal, z_plus) >= np.dot(avg_normal, z_minus):
        target = z_plus
    else:
        target = z_minus

    axis = np.cross(avg_normal, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-6:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(avg_normal, target), -1.0, 1.0)) * 180.0 / np.pi
        transform = vtk.vtkTransform()
        transform.RotateWXYZ(angle, axis)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputData(polydata)
        tf_filter.SetTransform(transform)
        tf_filter.Update()
        polydata = tf_filter.GetOutput()

    centered_polydata = center_polydata_at_origin(polydata)

    write_polydata(centered_polydata, output_vtk)

if __name__ == '__main__':

    data_path = f'/file_path/'
    os.chdir(f'{data_path}')
    base_dir = os.getcwd()
 
    remesh_dir = os.path.join(base_dir, 'remesh')
    output_dir = os.path.join(base_dir, 'center_mesh')
    os.makedirs(output_dir, exist_ok=True)

    for vtk_file in glob.glob(os.path.join(remesh_dir, '*.vtk')):
        
        name = os.path.splitext(os.path.basename(vtk_file))[0]

        out_centered_vtk = os.path.join(output_dir, f"centered {name}.vtk")
        align_mesh_to_z_axis(vtk_file, out_centered_vtk)
