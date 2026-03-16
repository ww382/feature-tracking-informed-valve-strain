import vtk
import numpy as np
import os
from pathlib import Path
from pipeline_utils import LEAFLETS, read_polydata, write_polydata

def plot_meshes(mesh1, mesh2, color1=(1,0,0), color2=(0,1,0), opacity1=0.5, opacity2=0.5):
    # Create mappers and actors
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(mesh1)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetColor(color1)
    actor1.GetProperty().SetOpacity(opacity1)

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(mesh2)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetColor(color2)
    actor2.GetProperty().SetOpacity(opacity2)

    # Axes actor
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(20, 20, 20)  # Adjust length as needed
    axes.AxisLabelsOn()
    axes.SetShaftTypeToCylinder()

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)
    renderer.AddActor(axes)

    # Legend for mesh colors
    legend = vtk.vtkLegendBoxActor()
    legend.SetNumberOfEntries(2)
    legend.SetEntry(0, actor1.GetMapper().GetInput(), "Aligned", color1)
    legend.SetEntry(1, actor2.GetMapper().GetInput(), "Target", color2)
    legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    legend.GetPositionCoordinate().SetValue(0.7, 0.05)  # bottom right
    legend.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
    legend.GetPosition2Coordinate().SetValue(0.98, 0.18)
    legend.UseBackgroundOn()
    legend.SetBackgroundColor(1, 1, 1)
    legend.SetBackgroundOpacity(0.7)
    renderer.AddActor(legend)
    renderer.SetBackground(1,1,1)

    # Render window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(2000, 2000)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderWindow.Render()
    renderWindowInteractor.Start()

    # Clean up after closing the window
    renderWindowInteractor.TerminateApp()
    renderWindow.Finalize()
    del renderWindowInteractor
    del renderWindow

def rotate_polydata(polydata, angle_x=0, angle_y=0, angle_z=0):
    """
    Rotate the polydata around the x, y, and z axes by the specified angles in degrees.
    """
    transform = vtk.vtkTransform()
    transform.RotateX(angle_x)
    transform.RotateY(angle_y)
    transform.RotateZ(angle_z)
    rot_filter = vtk.vtkTransformPolyDataFilter()
    rot_filter.SetInputData(polydata)
    rot_filter.SetTransform(transform)
    rot_filter.Update()

    rotated_polydata = vtk.vtkPolyData()
    rotated_polydata.DeepCopy(rot_filter.GetOutput())
    return rotated_polydata

def align(input_file, target_file, output_file, angle_x=0, angle_y=0, angle_z=0):
    
    input_polydata = read_polydata(input_file)
    target_polydata = read_polydata(target_file)

    rotated_polydata = rotate_polydata(input_polydata, angle_x, angle_y, angle_z)

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(rotated_polydata)
    icp.SetTarget(target_polydata)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    tfFilter = vtk.vtkTransformPolyDataFilter()
    tfFilter.SetInputData(rotated_polydata)
    tfFilter.SetTransform(icp)
    tfFilter.Update()

    aligned_polydata = vtk.vtkPolyData()
    aligned_polydata.DeepCopy(tfFilter.GetOutput())

    write_polydata(aligned_polydata, output_file)

    plot_meshes(aligned_polydata, target_polydata)

if __name__ == "__main__":

    data_path = f'/file_path/'
    os.chdir(f'{data_path}')

    print(os.getcwd())

    for leaflet in LEAFLETS:
        source_path = f"{Path(data_path).parent}/mean_shape/{leaflet}_mean_shape.vtk"
        target_path = f"center_mesh/centered MS {leaflet} atrial surface.vtk"

        output_path = f"center_mesh/reference mean shape {leaflet} leaflet.vtk"

        angle_x, angle_y, angle_z = 0, 0, 0
        while True:
            print(f"\nAligning {leaflet} leaflet with rotation X={angle_x}, Y={angle_y}, Z={angle_z} degrees...")
            align(source_path, target_path, output_path, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
            user_input = input("Enter rotation angles as 'x y z' (or press Enter/0 to continue): ")
            if user_input.strip() == "" or user_input.strip() == "0":
                break
            try:
                parts = user_input.strip().split()
                if len(parts) == 3:
                    angle_x, angle_y, angle_z = map(float, parts)
                else:
                    print("Please enter three numbers separated by spaces (e.g., 0 0 90).")
            except ValueError:
                print("Invalid input. Please enter three numbers or press Enter.")