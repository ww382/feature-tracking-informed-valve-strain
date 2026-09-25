"""Run the valve strain processing pipeline for a selected case."""

import ast
import os

code_list = [
    'pipeline_steps/resample_points.py',
    'pipeline_steps/center_mesh.py',
    'pipeline_steps/rotate_mesh.py',
    'pipeline_steps/align_mean_shape.py',
    'pipeline_steps/leaflet_registration.py',
    'pipeline_steps/results_postprocess_icp.py',
]

base_folder = "."
case = "example/"
new_file_path = f"{base_folder}/{case}".replace("//", "/")
FRAMES = ("MD", "MS")
LEAFLETS = ("anterior", "posterior", "septal")

class DataPathUpdater(ast.NodeTransformer):
    """Rewrite `data_path = ...` assignments to use the selected case path."""

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'data_path':
                node.value = ast.Constant(value=new_file_path)
        return self.generic_visit(node)


def update_and_exec_script(script_path):
    script_path = os.path.abspath(script_path)
    working_directory = os.getcwd()
    with open(script_path, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source, filename=script_path)
    modified_tree = DataPathUpdater().visit(tree)
    ast.fix_missing_locations(modified_tree)

    code = compile(modified_tree, filename=script_path, mode='exec')
    exec_namespace = {
        '__name__': '__main__',
        'FRAMES': FRAMES,
        '__file__': script_path,
        'LEAFLETS': LEAFLETS,
    }
    try:
        exec(code, exec_namespace)
    finally:
        # the steps change into the case folder; the next script path is relative to here
        os.chdir(working_directory)

workspace_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(workspace_dir)

for step in code_list:
    print(f'Running {step} with updated data_path...')
    update_and_exec_script(step)
    print(f'Completed: {step}\n')
