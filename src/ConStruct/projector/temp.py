import os
import pyvista as pv
import networkx as nx


def load_edges_from_polydata(mesh):
    """
    Extract edges from the 'lines' attribute of a PyVista PolyData.
    Assumes each edge is stored in the VTK format as [2, i, j].

    Returns:
        edges: A list of tuples (i, j) representing edges.
    """
    lines = mesh.lines.copy()  # flat connectivity array
    edges = []
    idx = 0
    while idx < len(lines):
        n = int(lines[idx])
        if n != 2:
            raise ValueError(f"Expected each edge cell to have 2 points, but got {n}")
        i = int(lines[idx + 1])
        j = int(lines[idx + 2])
        edges.append((i, j))
        idx += n + 1
    return edges


def save_connected_components(mesh, output_folder, base_filename):
    """
    Given a PyVista mesh, compute its connected components and save each as a separate VTP file.

    Args:
        mesh: A PyVista PolyData representing a graph.
        output_folder: Folder to save the individual component files.
        base_filename: Base filename (e.g., 'graph.vtp'); each component file will have a suffix like _1, _2, etc.
    """
    points = mesh.points  # shape: (num_points, 3)
    edges = load_edges_from_polydata(mesh)

    # Build a NetworkX graph.
    G = nx.Graph()
    num_points = points.shape[0]
    G.add_nodes_from(range(num_points))
    for (i, j) in edges:
        G.add_edge(i, j)

    # Compute connected components.
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected components.")

    # Process each connected component.
    for comp_idx, comp in enumerate(components, start=1):
        comp = sorted(list(comp))
        # Create a mapping from old node indices to new indices (0, 1, ..., |comp|-1)
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(comp)}

        # Filter edges: only keep edges with both endpoints in this component.
        comp_edges = [(i, j) for (i, j) in edges if i in mapping and j in mapping]

        # Create new connectivity in VTK line format.
        new_lines = []
        for (i, j) in comp_edges:
            new_lines.extend([2, mapping[i], mapping[j]])

        # Get the new points array (only nodes in the component).
        new_points = points[comp, :]

        # Create new PolyData.
        new_mesh = pv.PolyData(new_points, lines=new_lines)

        # Construct output filename with a suffix indicating the component number.
        out_filename = os.path.splitext(base_filename)[0] + f"_{comp_idx}.vtp"
        out_filepath = os.path.join(output_folder, out_filename)
        new_mesh.save(out_filepath)
        print(f"Saved component {comp_idx} to {out_filepath}")


# --- Main execution ---
input_folder = "/home/chinmayp/workspace/ConStruct/outputs/2025-02-20/20-58-29/check_something/graph_5.vtp"  # Update with your folder path.
output_folder = "/home/chinmayp/workspace/ConStruct/outputs/2025-02-20/20-58-29/check_something/"  # Update with your desired output folder.
os.makedirs(output_folder, exist_ok=True)

# Optionally, process all VTP files in the input folder:
# vtp_files = glob.glob(os.path.join(input_folder, "*.vtp"))
vtp_files = ["/home/chinmayp/workspace/ConStruct/outputs/2025-02-20/20-58-29/check_something/graph_5.vtp"]
for f in vtp_files:
    print("Processing file:", f)
    try:
        mesh = pv.read(f)
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue
    base_filename = os.path.basename(f)
    save_connected_components(mesh, output_folder, base_filename)
