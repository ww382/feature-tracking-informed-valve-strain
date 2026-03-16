import vtk
import numpy as np
import os
from pipeline_utils import LEAFLETS, center_polydata_at_origin, compute_average_cell_normal, read_polydata, write_polydata

def rotate_around_y_axis(polydata, angle_degrees):
    """Rotate polydata around the Y-axis by `angle_degrees`."""
    transform = vtk.vtkTransform()
    transform.RotateY(angle_degrees)

    rot_filter = vtk.vtkTransformPolyDataFilter()
    rot_filter.SetInputData(polydata)
    rot_filter.SetTransform(transform)
    rot_filter.Update()

    output = vtk.vtkPolyData()
    output.DeepCopy(rot_filter.GetOutput())
    return output

def compute_centroid(polydata):
    """Compute centroid of all mesh points."""
    num_points = polydata.GetNumberOfPoints()
    centroid = np.zeros(3)
    for i in range(num_points):
        point = np.array(polydata.GetPoint(i))
        centroid += point
    return centroid / num_points

def plot_meshes(mesh1, mesh2, color1=(1,0,0), color2=(0,1,0)):
    """Visualize target and source meshes with centroid normal arrows."""
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(mesh1)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetColor(color1)

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(mesh2)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetColor(color2)
    actor2.GetProperty().SetOpacity(0.5)

    # Compute centroids and normals
    centroid1 = compute_centroid(mesh1)
    centroid2 = compute_centroid(mesh2)
    normal1 = compute_average_cell_normal(mesh1)
    normal2 = compute_average_cell_normal(mesh2)

    # Helper to create an arrow actor
    def create_arrow_actor(centroid, normal, color):
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(0.3)
        arrow_source.SetTipRadius(0.05)
        arrow_source.SetShaftRadius(0.02)

        # Create transform to orient the arrow
        norm = np.linalg.norm(normal)
        if norm == 0:
            normal = np.array([0, 0, 1])
        else:
            normal = normal / norm

        # Default arrow points along +X, so compute rotation to align with normal
        transform = vtk.vtkTransform()
        transform.Translate(centroid)
        # Find rotation axis and angle
        default_dir = np.array([1, 0, 0])
        axis = np.cross(default_dir, normal)
        angle = np.degrees(np.arccos(np.clip(np.dot(default_dir, normal), -1.0, 1.0)))
        if np.linalg.norm(axis) > 1e-6:
            transform.RotateWXYZ(angle, axis)
        # Scale arrow for visibility
        transform.Scale(10, 10, 10)

        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(arrow_source.GetOutputPort())
        tf.SetTransform(transform)
        tf.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        return actor

    arrow_actor1 = create_arrow_actor(centroid1, normal1, (0,0,1))
    arrow_actor2 = create_arrow_actor(centroid2, normal2, (1,0.5,0))

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)
    renderer.AddActor(arrow_actor1)
    renderer.AddActor(arrow_actor2)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(2000, 2000)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    render_window.Render()
    iren.Start()
    
def check_mesh(source_filepath, target_filepath, output_filepath):
    source_polydata = read_polydata(source_filepath)
    target_polydata = read_polydata(target_filepath)

    source_normal = compute_average_cell_normal(source_polydata)
    target_normal = compute_average_cell_normal(target_polydata)

    if np.dot(source_normal, target_normal) < 0:
        source_polydata = rotate_around_y_axis(source_polydata, 180)

    centered_mesh = center_polydata_at_origin(source_polydata)
    write_polydata(centered_mesh, output_filepath)

    print(f"Saved updated mesh to: {output_filepath}")

    plot_meshes(target_polydata, centered_mesh)

if __name__ == "__main__":

    data_path = f'/file_path/'
    os.chdir(f'{data_path}')

    for leaflet in LEAFLETS:
        source_path = f"center_mesh/centered MD {leaflet} atrial surface.vtk"
        target_path = f"center_mesh/centered MS {leaflet} atrial surface.vtk"

        check_mesh(source_path, target_path, source_path)
