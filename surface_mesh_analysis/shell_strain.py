"""Large-deformation shell strain calculations for registered valve meshes."""

import vtk
import numpy as np
import os

from vtk.util.numpy_support import vtk_to_numpy

DEGENERATE_TOLERANCE = 1e-12


def read_vtk(file_path):
    """Read VTK, VTP, or VTU input as surface polydata."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
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
        reader = vtk.vtkDataSetReader()
    else:
        raise ValueError(f"Unsupported file extension: '{file_extension}'. Please use .vtk, .vtp, or .vtu files.")

    reader.SetFileName(file_path)
    reader.Update()
    
    dataset = reader.GetOutput()
    if dataset.IsA("vtkPolyData"):
        return dataset

    print("Input is not a surface. Applying vtkGeometryFilter to extract the surface.")
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(dataset)
    geometry_filter.Update()
    return geometry_filter.GetOutput()


def get_point_coordinates(polydata):
    """Return an (N, 3) float array of point coordinates from a vtkPolyData object."""
    return vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)


def get_cell_point_ids(cell):
    """Return the point IDs that define a VTK cell."""
    point_ids = cell.GetPointIds()
    return tuple(point_ids.GetId(i) for i in range(point_ids.GetNumberOfIds()))


def validate_matching_topology(initial_mesh, deformed_mesh):
    """Raise if two meshes do not have identical cell types and point connectivity."""
    if initial_mesh.GetNumberOfPoints() != deformed_mesh.GetNumberOfPoints():
        raise ValueError("Meshes must have the same number of points.")

    if initial_mesh.GetNumberOfCells() != deformed_mesh.GetNumberOfCells():
        raise ValueError("Meshes must have the same number of cells.")

    for cell_idx in range(initial_mesh.GetNumberOfCells()):
        initial_cell = initial_mesh.GetCell(cell_idx)
        deformed_cell = deformed_mesh.GetCell(cell_idx)

        if initial_cell.GetCellType() != deformed_cell.GetCellType():
            raise ValueError(f"Cell {cell_idx} has different cell types between meshes.")

        if get_cell_point_ids(initial_cell) != get_cell_point_ids(deformed_cell):
            raise ValueError(f"Cell {cell_idx} has different point connectivity between meshes.")


def surface_normal_and_area_density(tangent_1, tangent_2, cell_idx, mesh_label):
    """Return the unit normal and local surface area density for two tangents."""
    normal = np.cross(tangent_1, tangent_2)
    area_density = np.linalg.norm(normal)

    if area_density <= DEGENERATE_TOLERANCE:
        raise ValueError(f"Degenerate {mesh_label} cell {cell_idx}: local surface area is near zero.")

    return normal / area_density, area_density


def get_shape_functions(xi, eta, num_nodes):
    """Return shape functions and natural-coordinate derivatives."""
    if num_nodes == 4:
        N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
        dN_dxi_deta = 0.25 * np.array([
            [-(1-eta), (1-eta), (1+eta), -(1+eta)],
            [-(1-xi), -(1+xi), (1+xi), (1-xi)]
        ])
        return N, dN_dxi_deta
    elif num_nodes == 3:
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
    """Return Gauss integration points and weights for an element."""
    if num_nodes == 4:
        p = 1.0 / np.sqrt(3.0)
        points = [(-p, -p), (p, -p), (p, p), (-p, p)]
        weights = [1.0, 1.0, 1.0, 1.0]
        return points, weights
    elif num_nodes == 3:
        p = 1.0/6.0
        q = 2.0/3.0
        points = [(p, p), (q, p), (p, q)]
        weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        return points, weights
    else:
        raise ValueError(f"Gauss points for {num_nodes}-node elements not implemented.")

def calculate_large_deformation_strain_from_polydata(initial_mesh, deformed_mesh):
    """Calculate mid-surface shell strain with Gaussian quadrature.

    Takes the two meshes in memory and returns the deformed mesh with the strain arrays
    added to its point data. calculate_large_deformation_strain() is the same computation
    with files on either side.
    """
    deformed_mesh.GetPointData().Initialize()
    deformed_mesh.GetCellData().Initialize()

    initial_points = get_point_coordinates(initial_mesh)
    deformed_points = get_point_coordinates(deformed_mesh)

    validate_matching_topology(initial_mesh, deformed_mesh)

    num_points = initial_mesh.GetNumberOfPoints()
    num_cells = initial_mesh.GetNumberOfCells()

    nodal_strain_accumulator = [[] for _ in range(num_points)]
    nodal_areal_strain_accumulator = [[] for _ in range(num_points)]

    for i in range(num_cells):
        cell = initial_mesh.GetCell(i)
        point_ids = cell.GetPointIds()
        num_cell_points = cell.GetNumberOfPoints()
        
        initial_cell_points = np.array([initial_points[point_ids.GetId(j)] for j in range(num_cell_points)])
        deformed_cell_points = np.array([deformed_points[point_ids.GetId(j)] for j in range(num_cell_points)])

        gauss_points, gauss_weights = get_gauss_points(num_cell_points)
        total_weight = sum(gauss_weights)

        strain_total = np.zeros((3, 3))
        ref_area_total = 0.0
        def_area_total = 0.0
        
        for gp_idx, (xi, eta) in enumerate(gauss_points):
            weight = gauss_weights[gp_idx]
            _, dN_dxi_deta = get_shape_functions(xi, eta, num_cell_points)
            
            J_ref = np.dot(dN_dxi_deta, initial_cell_points)
            t1_ref, t2_ref = J_ref[0, :], J_ref[1, :]
            n_ref, ref_area_density = surface_normal_and_area_density(t1_ref, t2_ref, i, "initial")
            B_ref = np.vstack([t1_ref, t2_ref, n_ref]).T
            
            J_def = np.dot(dN_dxi_deta, deformed_cell_points)
            t1_def, t2_def = J_def[0, :], J_def[1, :]
            n_def, def_area_density = surface_normal_and_area_density(t1_def, t2_def, i, "deformed")
            B_def = np.vstack([t1_def, t2_def, n_def]).T

            try:
                F = np.dot(B_def, np.linalg.inv(B_ref))
            except np.linalg.LinAlgError:
                raise ValueError(f"Cell {i} has a singular reference basis.")
            
            C = np.dot(F.T, F)
            I = np.identity(3)
            E_gp = 0.5 * (C - I)
            
            strain_total += E_gp * weight
            ref_area_total += ref_area_density * weight
            def_area_total += def_area_density * weight

        E_element_avg = strain_total / total_weight
        areal_strain = (def_area_total / ref_area_total) - 1
        
        for j in range(num_cell_points):
            global_node_id = point_ids.GetId(j)
            nodal_strain_accumulator[global_node_id].append(E_element_avg)
            nodal_areal_strain_accumulator[global_node_id].append(areal_strain)

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

    point_data = deformed_mesh.GetPointData()
    point_data.AddArray(strain_tensor_mid_array)
    point_data.AddArray(principal_strains_mid_array)
    point_data.AddArray(area_strain_array)

    return deformed_mesh


def calculate_large_deformation_strain(initial_mesh_file, deformed_mesh_file, output_file=None):
    """Read two meshes, calculate the strain, and write the result when a path is given."""
    initial_mesh = read_vtk(initial_mesh_file)
    deformed_mesh = read_vtk(deformed_mesh_file)

    deformed_mesh = calculate_large_deformation_strain_from_polydata(initial_mesh, deformed_mesh)

    if output_file is None:
        return deformed_mesh

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(deformed_mesh)
    writer.Write()

    print(f"Strain analysis complete. Output written to: {output_file}")
    return deformed_mesh


if __name__ == '__main__':

    ref_file = "mitral_initial.vtk"
    def_file = "mitral_final.vtk"
    output_file = "shell_strain_analysis.vtk"

    calculate_large_deformation_strain(ref_file, def_file, output_file)
