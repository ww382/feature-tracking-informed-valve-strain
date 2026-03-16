import glob
import os
import pyvista as pv
import pyacvd

def uniform_remesh(mesh, n_points):
    """Uniformly remesh a surface to approximately `n_points` vertices."""
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    smooth_w_taubin = mesh.smooth_taubin(n_iter=10, pass_band=0.05)
    clus = pyacvd.Clustering(smooth_w_taubin)

    # Subdivision improves point distribution before clustering.
    clus.subdivide(3)
    clus.cluster(n_points)
    return clus.create_mesh()


data_path = f'/file_path/'
os.chdir(f'{data_path}')
base_dir = os.getcwd()

output_dir = "remesh"
os.makedirs(output_dir, exist_ok=True)

print(f'{base_dir}original mesh')
files = glob.glob(f'{base_dir}/original mesh/*.vtk')
print(f"Found files: {files}")

for source_path in files:
    mesh = pv.PolyData(source_path)

    mesh = mesh.clean(point_merging=True)

    remeshed = uniform_remesh(mesh, n_points=1000)

    print(f"Number of points in remeshed mesh: {remeshed.n_points}")
    outfile = source_path.split("/")[-1]
    out_file = os.path.join(output_dir, outfile)
    remeshed.save(out_file)
    print(f"Saved remeshed file to: {out_file}")