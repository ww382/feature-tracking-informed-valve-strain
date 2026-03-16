#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Aug 05 2025

@author: Wensi Wu
@email: wensiwu@seas.upenn.edu
"""

import vtk
import numpy as np
import os

from vtk.util.numpy_support import vtk_to_numpy

def read_vtk(file_path):
    """
    Reads a VTK file (legacy .vtk, XML .vtp, or XML .vtu) and returns the
    surface as a vtkPolyData object. It automatically detects the file type
    and converts volumetric grids to surfaces if necessary.

    Args:
        file_path (str): The path to the VTK file.

    Returns:
        vtk.vtkPolyData: The VTK polydata object representing the surface.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    reader = None
    if file_extension == ".vtp":
        print(f"Reading XML PolyData file: {file_path}")
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file_path)
        reader.Update()
        return reader.GetOutput()
    elif file_extension == ".vtu":
        print(f"Reading XML UnstructuredGrid file: {file_path}")
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif file_extension == ".vtk":
        print(f"Reading legacy VTK file: {file_path}")
        # Use vtkDataSetReader which can handle any legacy format
        reader = vtk.vtkDataSetReader()
    else:
        raise ValueError(f"Unsupported file extension: '{file_extension}'. Please use .vtk, .vtp, or .vtu files.")

    reader.SetFileName(file_path)
    reader.Update()
    
    # The output might be UnstructuredGrid, so we need to ensure it's PolyData
    dataset = reader.GetOutput()
    if dataset.IsA("vtkPolyData"):
        return dataset
    else:
        # If it's not polydata (e.g., it's an unstructured grid),
        # extract the external surface.
        print("Input is not a surface. Applying vtkGeometryFilter to extract the surface.")
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputData(dataset)
        geometry_filter.Update()
        return geometry_filter.GetOutput()


def get_point_coordinates(polydata):
    """Return an (N, 3) float array of point coordinates from a vtkPolyData object."""
    return vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)

def get_shape_functions(xi, eta, num_nodes):
    """
    Returns shape functions N and their derivatives dN/d(xi,eta) for a given
    natural coordinate (xi, eta).
    """
    if num_nodes == 4: # Bilinear Quad
        N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
        dN_dxi_deta = 0.25 * np.array([
            [-(1-eta), (1-eta), (1+eta), -(1+eta)],
            [-(1-xi), -(1+xi), (1+xi), (1-xi)]
        ])
        return N, dN_dxi_deta
    elif num_nodes == 3: # Linear Triangle
        L1, L2, L3 = xi, eta, 1-xi-eta
        N = np.array([L1, L2, L3])
        dN_dxi_deta = np.array([
            [1, 0, -1],
            [0, 1, -1]
        ])
        return N, dN_dxi_deta
    else:
        raise ValueError(f"Shape functions for {num_nodes}-node elements not implemented.")

def get_gauss_points(num_nodes):
    """
    Returns Gauss integration points and weights for an element.
    """
    if num_nodes == 4: # 2x2 Gauss quadrature for Quads
        p = 1.0 / np.sqrt(3.0)
        points = [(-p, -p), (p, -p), (p, p), (-p, p)]
        weights = [1.0, 1.0, 1.0, 1.0]
        return points, weights
    elif num_nodes == 3: # 3-point quadrature for Triangles
        p = 1.0/6.0
        q = 2.0/3.0
        points = [(p, p), (q, p), (p, q)]
        weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        return points, weights
    else:
        raise ValueError(f"Gauss points for {num_nodes}-node elements not implemented.")

def calculate_large_deformation_strain(initial_mesh_file, deformed_mesh_file, output_file):
    """
    Calculates large deformation strain for the mid-surface of a shell
    using a formulation consistent with continuum-based shell theory.
    This involves multi-point Gaussian quadrature.
    """
    initial_mesh = read_vtk(initial_mesh_file)
    deformed_mesh = read_vtk(deformed_mesh_file)

    # Clear any existing data fields from the input deformed mesh
    deformed_mesh.GetPointData().Initialize()
    deformed_mesh.GetCellData().Initialize()

    initial_points = get_point_coordinates(initial_mesh)
    deformed_points = get_point_coordinates(deformed_mesh)

    num_points = initial_mesh.GetNumberOfPoints()
    num_cells = initial_mesh.GetNumberOfCells()

    if num_points != deformed_mesh.GetNumberOfPoints() or \
       num_cells != deformed_mesh.GetNumberOfCells():
        raise ValueError("Meshes must have the same number of points and cells.")

    # Data structures for storing intermediate results
    nodal_strain_accumulator = [[] for _ in range(num_points)]
    nodal_areal_strain_accumulator = [[] for _ in range(num_points)]

    # Calculate weighted average strain for each cell and distribute to nodes
    for i in range(num_cells):
        cell = initial_mesh.GetCell(i)
        point_ids = cell.GetPointIds()
        num_cell_points = cell.GetNumberOfPoints()
        
        initial_cell_points = np.array([initial_points[point_ids.GetId(j)] for j in range(num_cell_points)])
        deformed_cell_points = np.array([deformed_points[point_ids.GetId(j)] for j in range(num_cell_points)])

        gauss_points, gauss_weights = get_gauss_points(num_cell_points)
        total_weight = sum(gauss_weights)

        E_total = np.zeros((3, 3))
        J_total = 0.0
        
        for gp_idx, (xi, eta) in enumerate(gauss_points):
            weight = gauss_weights[gp_idx]
            _, dN_dxi_deta = get_shape_functions(xi, eta, num_cell_points)
            
            J_ref = np.dot(dN_dxi_deta, initial_cell_points)
            t1_ref, t2_ref = J_ref[0, :], J_ref[1, :]
            n_ref = np.cross(t1_ref, t2_ref); n_ref /= np.linalg.norm(n_ref)
            B_ref = np.vstack([t1_ref, t2_ref, n_ref]).T # basis
            
            J_def = np.dot(dN_dxi_deta, deformed_cell_points)
            t1_def, t2_def = J_def[0, :], J_def[1, :]
            n_def = np.cross(t1_def, t2_def); n_def /= np.linalg.norm(n_def)
            B_def = np.vstack([t1_def, t2_def, n_def]).T # basis

            try:
                F = np.dot(B_def, np.linalg.inv(B_ref))
            except np.linalg.LinAlgError:
                F = np.identity(3)
            
            C = np.dot(F.T, F)
            I = np.identity(3)
            E_gp = 0.5 * (C - I)
            
            # Accumulate the weighted strain and Jacobian determinant
            E_total += E_gp * weight
            J_total += np.linalg.det(F) * weight

        # Weighted-average strain and Jacobian determinant for the element
        E_element_avg = E_total / total_weight
        J_avg = J_total / total_weight
        areal_strain = J_avg - 1
        
        # Distribute this single averaged strain to the nodes of the element
        for j in range(num_cell_points):
            global_node_id = point_ids.GetId(j)
            nodal_strain_accumulator[global_node_id].append(E_element_avg)
            nodal_areal_strain_accumulator[global_node_id].append(areal_strain)

    # Allocate output arrays
    strain_tensor_mid_array = vtk.vtkDoubleArray()
    strain_tensor_mid_array.SetName("Lagrange_strain")
    strain_tensor_mid_array.SetNumberOfComponents(9)
    strain_tensor_mid_array.SetNumberOfTuples(num_points)

    principal_strains_mid_array = vtk.vtkDoubleArray()
    principal_strains_mid_array.SetName("Principal_Strains")
    principal_strains_mid_array.SetNumberOfComponents(3)
    principal_strains_mid_array.SetNumberOfTuples(num_points)

    area_strain_array = vtk.vtkDoubleArray()
    area_strain_array.SetName("ArealStrain")
    area_strain_array.SetNumberOfTuples(num_points)

    for i in range(num_points):
        if not nodal_strain_accumulator[i]:
            E_mid = np.zeros((3, 3))
            avg_areal_strain = 0.0
        else:
            E_mid = np.mean(nodal_strain_accumulator[i], axis=0)
            avg_areal_strain = np.mean(nodal_areal_strain_accumulator[i])

        principal_strains_sorted = np.sort(np.linalg.eigvalsh(E_mid))[::-1]

        strain_tensor_mid_array.SetTuple(i, E_mid.flatten())
        principal_strains_mid_array.SetTuple(i, principal_strains_sorted)
        area_strain_array.SetTuple1(i, avg_areal_strain)

    # Add output arrays to the deformed mesh
    point_data = deformed_mesh.GetPointData()
    point_data.AddArray(strain_tensor_mid_array)
    point_data.AddArray(principal_strains_mid_array)
    point_data.AddArray(area_strain_array)

    # Write the final mesh with POINT data
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(deformed_mesh)
    writer.Write()

    print(f"Strain analysis complete. Output written to: {output_file}")
    
if __name__ == '__main__':

    ref_file = "mitral_initial.vtk"
    def_file = "mitral_final.vtk"
    output_file = "shell_strain_analysis.vtk"

    calculate_large_deformation_strain(ref_file, def_file, output_file)

