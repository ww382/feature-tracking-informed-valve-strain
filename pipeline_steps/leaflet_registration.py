#!/usr/bin/env python3
"""Register each leaflet mean shape to MD/MS frames and compute strain."""

import numpy as np
import matplotlib.pyplot as plt
from pycpd import RigidRegistration
import os
import sys
import vtk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_mesh_analysis.mesh_utils import apply_displacement
from surface_mesh_analysis.shell_strain import calculate_large_deformation_strain
from surface_mesh_analysis.dw_cpd_registration import DistanceWeightedDeformableCPD


def extract_vertices(vtk_filepath):
    """Extract point coordinates from a VTK polydata file."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_filepath)
    reader.Update()
        
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
        
    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    return vertices

def read_vtk_files(source_filepath, target_filepath):
    """Read source and target VTK files as point-coordinate arrays."""
    source_vertices = extract_vertices(source_filepath)
    target_vertices = extract_vertices(target_filepath)
    return source_vertices, target_vertices


def registration_callback(iteration, error, X, Y):
    """Plot source and target points during each registration iteration."""
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c='r', label='Target', marker='o')
    plt.scatter(Y[:, 0], Y[:, 1], c='b', label='Transformed Source', marker='x')
    plt.title(f"Iteration: {iteration}  Error: {error:.4f}")
    plt.legend()
    plt.draw()
    plt.pause(0.05)
  

def update_shape(filepath, displacement, output_filepath):
    """Apply a point displacement field and write the updated mesh."""
    apply_displacement(filepath, displacement, output_filepath)


def registration(source, target):
    """Register source points to target points with rigid and deformable CPD."""

    reg = RigidRegistration(X=target, Y=source, max_iterations=3)
    transformed_source, _ = reg.register(callback=registration_callback)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='r', label='Target')
    ax.scatter(transformed_source[:, 0], transformed_source[:, 1], transformed_source[:, 2], 
               c='b', marker='x', label='Transformed Source')
    ax.set_title("Final Registration Result")
    ax.legend()
    plt.show()

    # Lower decay factors give smoother deformation; higher values localize it.
    reg = DistanceWeightedDeformableCPD(X=target, Y=transformed_source, alpha=1, beta=5, max_iterations=30)
    transformed_source, _ = reg.register(callback=registration_callback)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='r', label='Target')
    ax.scatter(transformed_source[:, 0], transformed_source[:, 1], transformed_source[:, 2], 
               c='b', marker='x', label='Transformed Source')
    ax.set_title("Final Registration Result")
    ax.legend()
    plt.show()

    return transformed_source

if __name__ == '__main__':

    data_path = f'/file_path/'
    os.chdir(f'{data_path}')
    base_dir = os.getcwd()
    
    output_dir = os.path.join(base_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    for leaflet in LEAFLETS:
        reference_path = f"center_mesh/reference mean shape {leaflet} leaflet.vtk"
        transformed_paths = {}

        for frame in FRAMES:
            target_path = f"center_mesh/centered {frame} {leaflet} atrial surface.vtk"
            transformed_path = f"results/Transformed {frame} {leaflet} atrial surface.vtk"

            source_points, target_points = read_vtk_files(reference_path, target_path)
            transformed_points = registration(source_points, target_points)
            displacement = transformed_points - source_points
            update_shape(reference_path, displacement, transformed_path)
            transformed_paths[frame] = transformed_path

        strain_path = f"results/Strain MS {leaflet} atrial surface.vtk"
        calculate_large_deformation_strain(
            transformed_paths["MD"],
            transformed_paths["MS"],
            strain_path,
        )
