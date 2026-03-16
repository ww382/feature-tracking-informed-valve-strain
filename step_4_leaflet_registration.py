#!/usr/bin/env python3
"""
Register each leaflet mean shape to MD/MS frames and compute strain.
"""

import numpy as np
import matplotlib.pyplot as plt
from pycpd import RigidRegistration
import os
import sys
import vtk
from pipeline_utils import FRAMES, LEAFLETS, apply_displacement

from compute_shell_strain import calculate_large_deformation_strain
from DW_CPD_registration import DistanceWeightedDeformableCPD

def extract_vertices(vtk_filepath):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_filepath)
    reader.Update()
        
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
        
    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    return vertices

def read_vtk_files(source_filepath, target_filepath):

    """
    Reads two VTK files and extracts their vertices using vtk.

    Parameters:
        source_filepath (str): Path to the source VTK file.
        target_filepath (str): Path to the target VTK file.

    Returns:
        tuple: A tuple containing:
            - source_vertices (np.ndarray): Array of unique vertices from the source mesh.
            - target_vertices (np.ndarray): Array of unique vertices from the target mesh.
    """

    source_vertices = extract_vertices(source_filepath)
    target_vertices = extract_vertices(target_filepath)
    return source_vertices, target_vertices

def registration_callback(iteration, error, X, Y):
    """
    Callback function invoked at each iteration of the registration process.
    
    Parameters:
        iteration (int): The current iteration number.
        error (float): The current registration error.
        X (np.ndarray): The fixed (target) point set.
        Y (np.ndarray): The current state of the moving (source) point set.
    """
    plt.clf()
    # Plot target (fixed points)
    plt.scatter(X[:, 0], X[:, 1], c='r', label='Target', marker='o')
    # Plot current transformed source (moving points)
    plt.scatter(Y[:, 0], Y[:, 1], c='b', label='Transformed Source', marker='x')
    plt.title(f"Iteration: {iteration}  Error: {error:.4f}")
    plt.legend()
    plt.draw()
    plt.pause(0.05)
  
def update_shape(filepath, displacement, output_filepath):
    apply_displacement(filepath, displacement, output_filepath)

def registration(source, target):

    reg = RigidRegistration(X=target, Y=source, max_iterations=3)
    transformed_source, _ = reg.register(callback=registration_callback)

    # Final plot to display the registration result.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='r', label='Target')
    ax.scatter(transformed_source[:, 0], transformed_source[:, 1], transformed_source[:, 2], 
               c='b', marker='x', label='Transformed Source')
    ax.set_title("Final Registration Result")
    ax.legend()
    plt.show()

    # lower decay factor lead to smoother response, higher decay factor lead to more localized response
    reg = DistanceWeightedDeformableCPD(X=target, Y=transformed_source, alpha=1, beta=5, max_iterations=30)
    #reg = DeformableRegistration(X=target, Y=transformed_source, alpha=1, beta=5)
    transformed_source, _ = reg.register(callback=registration_callback)

    # Final plot to display the registration result.
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