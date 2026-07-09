"""Shared helpers for the ShapeAnalysis processing pipeline."""

import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def set_workdir(data_path, subdir=""):
    """Change to `data_path/subdir` and return the absolute directory path."""
    target = os.path.join(data_path, subdir) if subdir else data_path
    abs_target = os.path.abspath(target)
    os.chdir(abs_target)
    return abs_target


def read_polydata(filename):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def write_polydata(polydata, filename):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def polydata_points_to_numpy(polydata):
    return vtk_to_numpy(polydata.GetPoints().GetData())


def apply_transform(polydata, transform):
    tf_filter = vtk.vtkTransformPolyDataFilter()
    tf_filter.SetInputData(polydata)
    tf_filter.SetTransform(transform)
    tf_filter.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(tf_filter.GetOutput())
    return output


def rigid_icp_transform(source_polydata, target_polydata, max_iterations=100):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_polydata)
    icp.SetTarget(target_polydata)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(max_iterations)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    return icp


def compute_average_cell_normal(polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputeCellNormalsOn()
    normals.Update()

    normal_data = normals.GetOutput().GetCellData().GetNormals()
    total = np.zeros(3)
    num_cells = polydata.GetNumberOfCells()
    for idx in range(num_cells):
        total += np.array(normal_data.GetTuple(idx))

    magnitude = np.linalg.norm(total)
    if magnitude > 0:
        return total / magnitude
    return np.array([0.0, 0.0, 1.0])


def center_polydata_at_origin(polydata):
    center_of_mass = vtk.vtkCenterOfMass()
    center_of_mass.SetInputData(polydata)
    center_of_mass.SetUseScalarsAsWeights(False)
    center_of_mass.Update()
    center = center_of_mass.GetCenter()

    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])
    return apply_transform(polydata, transform)


def apply_displacement(input_path, displacement, output_path):
    polydata = read_polydata(input_path)
    vertices = polydata_points_to_numpy(polydata)
    deformed_vertices = vertices + displacement

    new_points = vtk.vtkPoints()
    for point in deformed_vertices:
        new_points.InsertNextPoint(point)
    polydata.SetPoints(new_points)

    write_polydata(polydata, output_path)
