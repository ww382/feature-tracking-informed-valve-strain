import ast
import os

# --- Configuration ---
code_list = [
    'step_1_resample_points.py',
    'step_2_center_mesh.py',
    'step_3_rotate_mesh.py',
    'step_3p5_align_mean_shape.py',
    'step_4_leaflet_registration.py',
    'step_5_results_postprocess_icp.py',
]

base_folder = "."
case = "example/"
new_file_path = f"{base_folder}/{case}".replace("//", "/")

class DataPathUpdater(ast.NodeTransformer):
    """Rewrite `data_path = ...` assignments to use the selected case path."""

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'data_path':
                node.value = ast.Constant(value=new_file_path)
        return self.generic_visit(node)


def update_and_exec_script(script_path):
    script_path = os.path.abspath(script_path)
    with open(script_path, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source, filename=script_path)
    modified_tree = DataPathUpdater().visit(tree)
    ast.fix_missing_locations(modified_tree)

    code = compile(modified_tree, filename=script_path, mode='exec')
    exec_namespace = {'__name__': '__main__'}
    exec(code, exec_namespace)

workspace_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(workspace_dir)

for step in code_list:
    print(f'Running {step} with updated data_path...')
    update_and_exec_script(step)
    print(f'Completed: {step}\n')