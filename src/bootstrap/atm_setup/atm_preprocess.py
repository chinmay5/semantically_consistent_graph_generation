import shutil

import pyvista as pv
import numpy as np
import glob
import os
import networkx as nx

from tqdm import tqdm

def oversample_points():
    # Load the .vtp file
    base_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_raw"
    dest_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled"
    print("Executing oversample_points()")
    if os.path.exists(dest_folder):
        print("Cleaning destination folder")
        shutil.rmtree(dest_folder)
    for split in ['train_data', 'test_data']:
        input_folder = f"{base_folder}/{split}"
        output_folder = f"{dest_folder}/{split}"
        os.makedirs(output_folder, exist_ok=True)
        for input_filename in glob.glob(f"{input_folder}/*.vtp"):
            graph = pv.read(input_filename)

            # Extract points and edges
            points = graph.points  # Node positions
            cells = graph.lines.reshape(-1, 3)[:, 1:]  # Edge connectivity (assuming lines)

            # Get edge weights from the "radius" attribute
            edge_radii = graph.cell_data["radius"]

            # Step 1: Filter edges with radius > 3
            valid_edges = np.where(edge_radii > 3)[0]

            if len(valid_edges) == 0:
                raise ValueError("No edges found with radius > 3.")

            # Step 2: Find the longest edge among them
            longest_edge_index = None
            max_length = 0

            for edge_idx in valid_edges:
                p1, p2 = cells[edge_idx]  # Get point indices
                dist = np.linalg.norm(points[p1] - points[p2])  # Compute Euclidean distance

            if dist > max_length:
                max_length = dist
                longest_edge_index = edge_idx

            # Step 3: Generate 10 intermediate points along the longest edge
            p1, p2 = cells[longest_edge_index]
            start, end = points[p1], points[p2]
            intermediate_points = np.linspace(start, end, num=12)[1:-1]  # Excluding endpoints

            # Step 4: Add new points to the point cloud
            new_point_ids = list(range(len(points), len(points) + len(intermediate_points)))
            updated_points = np.vstack([points, intermediate_points])

            # Step 5: Create new edges connecting intermediate points
            new_edges = []
            prev_id = p1  # Start from the first node

            for new_id in new_point_ids:
                new_edges.append([prev_id, new_id])
                prev_id = new_id

            # Connect last intermediate point to the original second endpoint
            new_edges.append([prev_id, p2])

            # Convert to VTK format
            new_edges = np.array(new_edges).flatten()
            new_cells = np.hstack([np.full((len(new_edges) // 2, 1), 2), new_edges.reshape(-1, 2)]).flatten()

            # Step 6: Update radius values correctly
            new_radius_values = np.full(len(new_edges) // 2,
                                        edge_radii[longest_edge_index])  # Use the original longest edge's radius

            # Step 7: Save updated graph to .vtp
            updated_graph = pv.PolyData()
            updated_graph.points = updated_points
            updated_graph.lines = np.hstack([np.full((len(cells), 1), 2), cells]).flatten()  # Keep old edges
            updated_graph.lines = np.hstack([updated_graph.lines, new_cells])  # Append new edges

            # Copy original attributes and extend with new values
            updated_radius_values = np.append(edge_radii, new_radius_values)
            updated_graph.cell_data["radius"] = updated_radius_values

            # Save the new .vtp file
            updated_graph.save(f"{output_folder}/{os.path.basename(input_filename)}")

##############################################################################################
# Minimum Spanning Tree Code
##############################################################################################

def load_edges_from_polydata(mesh):
    """
    Extract edges from the 'lines' attribute of a PyVista PolyData.
    Assumes each cell (edge) is stored as: [2, i, j] (i.e. a 2-point line).

    Returns:
        edges: A list of tuples (i, j) representing each edge.
    """
    lines = mesh.lines.copy()  # flat array of connectivity data
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


def build_networkx_graph(mesh, edges, radii):
    """
    Build a NetworkX graph from the PyVista mesh and its list of edges.

    Each node is added with its coordinate (from mesh.points).
    Each edge is weighted by the Euclidean distance between its endpoints,
    and the original radius attribute is stored in the edge data.

    Args:
        mesh: PyVista mesh.
        edges: List of tuples (i, j) representing edge connectivity.
        radii: A numpy array (or list) of radius values for each edge.

    Returns:
        G: A NetworkX Graph with edge attributes "weight" and "radius".
    """
    G = nx.Graph()
    points = mesh.points
    num_points = points.shape[0]
    # Add nodes with positions
    for i in range(num_points):
        G.add_node(i, pos=points[i])
    # Add edges with weights computed as Euclidean distances and store the radius.
    for edge_idx, (i, j) in enumerate(edges):
        weight = np.linalg.norm(points[i] - points[j])
        edge_radius = radii[edge_idx]  # get the corresponding radius value
        G.add_edge(i, j, weight=weight, radius=edge_radius)
    return G


def convert_mst_to_polydata(mst, points):
    """
    Convert a minimum spanning tree (NetworkX Graph) into a PyVista PolyData object.

    Each edge in the MST will be converted to a line cell in VTK format.
    Args:
        mst: The MST as a NetworkX Graph.
        points: A numpy array of node coordinates (shape [num_points, 3]).

    Returns:
        new_mesh: A PyVista PolyData with the same points and new connectivity.
    """
    lines = []
    edge_radii = []
    for (i, j) in mst.edges():
        lines.extend([2, i, j])
        # Get the radius from the MST edge attribute.
        edge_radii.append(mst.edges[i, j]['radius'])
    new_mesh = pv.PolyData(points, lines=lines)
    new_mesh.cell_data["radius"] = np.array(edge_radii)
    return new_mesh


def convert_to_mst():
    """
    Process all VTP files in the input folder:
      - Load each VTP file.
      - Extract its points and edge connectivity.
      - Build a NetworkX graph and compute its minimum spanning tree.
      - Convert the MST back to a PolyData.
      - Save the new PolyData in the output folder.
    """
    input_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled"
    counter = total = 0
    for split in ['train_data', 'test_data']:
        output_folder = f"/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst/{split}"
        os.makedirs(output_folder, exist_ok=True)
        for f in tqdm(glob.glob(os.path.join(input_folder, split, "*.vtp"))):
            try:
                # Load the graph from the VTP file.
                mesh = pv.read(f)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue

            try:
                # Extract edges from the mesh.
                edges = load_edges_from_polydata(mesh)
            except Exception as e:
                print(f"Error extracting edges from {f}: {e}")
                continue

            # Get the original radii (make sure it's a numpy array).
            orig_radii = np.array(mesh.cell_data["radius"])

            # Build a networkx graph using node positions, edge weights, and radius attribute.
            G = build_networkx_graph(mesh, edges, orig_radii)

            # Compute the Minimum Spanning Tree (MST) of the graph.
            mst = nx.minimum_spanning_tree(G, weight='weight')

            # Check if the original graph already is acyclic:
            if mst.number_of_edges() == G.number_of_edges():
                # print("Graph is acyclic (no cycles found). Using original connectivity.")
                new_mesh = mesh  # Use original mesh (or mesh.copy() if needed)
            else:
                print(f" {os.path.basename(f)} contains cycles. Replacing connectivity with MST.")
                new_mesh = convert_mst_to_polydata(mst, mesh.points)
                counter += 1

            # Save the new graph as a VTP file.
            filename = os.path.basename(f)
            new_filepath = os.path.join(output_folder, filename)
            new_mesh.save(new_filepath)
            total += 1
    print(f"{counter} out of {total} graphs had cycles and were replaced with MSTs.")



##############################################################################################
# Discretization Code
##############################################################################################
def discretize_radii(all_radius_arr):
    def discrete_me(x):
        if x > 5:
            return 4
        elif 3 < x <= 5:
            return 3
        elif 1.5 < x <= 3:
            return 2
        else:
            return 1
    return np.array([discrete_me(x) for x in np.array(all_radius_arr)])


def discretize_all_graphs():
    print("Executing discretize_all_graphs()")
    print("Please execute this script after running the oversample_points() and MST functions.")
    discrete_radii_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete"
    mst_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst"
    if os.path.exists(discrete_radii_folder):
        print("Cleaning final folder")
        shutil.rmtree(discrete_radii_folder)
    for split in ['train_data', 'test_data']:
        cont_graph_folder = f"{mst_folder}/{split}"
        discrete_graph_folder = f"{discrete_radii_folder}/{split}"
        os.makedirs(discrete_graph_folder, exist_ok=True)
        print("Processing", cont_graph_folder)
        for input_filename in tqdm(glob.glob(f"{cont_graph_folder}/*.vtp")):
            graph = pv.read(input_filename)
            edge_radii = graph.cell_data["radius"]
            # discretize radii
            discrete_radius = discretize_radii(edge_radii)
            graph.cell_data["radius"] = discrete_radius
            graph.save(f"{discrete_graph_folder}/{os.path.basename(input_filename)}")

##############################################################################################
# Code to fix broken validity criterion
##############################################################################################
def build_node_to_edges(edges, num_nodes):
    """
    Build a dictionary mapping each node index to the list of edge indices incident on it.
    Args:
        edges: list of tuples (i, j)
        num_nodes: total number of nodes
    Returns:
        node_to_edges: dict mapping node index -> list of edge indices.
    """
    node_to_edges = {i: [] for i in range(num_nodes)}
    for e_idx, (i, j) in enumerate(edges):
        node_to_edges[i].append(e_idx)
        node_to_edges[j].append(e_idx)
    return node_to_edges


def update_edge_labels(mesh, threshold=1.0):
    """
    Given a PyVista mesh with edge connectivity stored in mesh.lines and an edge attribute
    (in cell_data["radius"]) that currently holds categorical labels (say, 1, 2, or 3),
    update the labels according to the following rules:

      1. If an edge labeled 2 is incident on both endpoints to at least one edge with label 3,
         then reassign that edge's label to 3.
      2. If an edge labeled 1 is incident on any endpoint to an edge with label 3,
         then reassign that edge's label to 2.

    These rules are intended to “fix” inconsistent labeling in the graph.

    Args:
        mesh: A PyVista PolyData object.
        threshold: Not used in this simple version (but could be used if differences are numeric).

    Returns:
        updated_labels: The updated labels as a numpy array.
    """
    # Extract edges from mesh.
    edges = load_edges_from_polydata(mesh)
    num_nodes = mesh.points.shape[0]

    # Get original labels (assumed to be stored in cell_data["radius"])
    # Here we assume they are categorical (e.g. integers 1, 2, 3).
    labels = np.array(mesh.cell_data["radius"])

    # Build node -> edge mapping.
    node_to_edges = build_node_to_edges(edges, num_nodes)

    # Make a copy of labels for update.
    new_labels = labels.copy()

    # Rule 1: For each edge with label 2, if at both endpoints there is at least one other edge
    # with label 3, update it to 3.
    for e_idx, (i, j) in enumerate(edges):
        if labels[e_idx] == 2:
            # For node i, check if any incident edge (other than e_idx) has label 3.
            has_label3_i = any(new_labels[ne] == 3 for ne in node_to_edges[i] if ne != e_idx)
            # For node j, check similarly.
            has_label3_j = any(new_labels[ne] == 3 for ne in node_to_edges[j] if ne != e_idx)
            if has_label3_i and has_label3_j:
                new_labels[e_idx] = 3

    # Rule 2: For each edge with label 1, if at either endpoint there is any incident edge with label 3,
    # then update its label to 2.
    for e_idx, (i, j) in enumerate(edges):
        if labels[e_idx] == 1:
            has_label3 = any(new_labels[ne] == 3 for ne in node_to_edges[i] if ne != e_idx) or \
                         any(new_labels[ne] == 3 for ne in node_to_edges[j] if ne != e_idx)
            if has_label3:
                new_labels[e_idx] = 2
    # Rule 3: For each edge with label 4, if at BOTH endpoint there is any incident edge with label 2 or 1,
    # then update its label to 2.
    for e_idx, (i, j) in enumerate(edges):
        if labels[e_idx] == 4:
            has_label_2_or_1 = any(new_labels[ne] == 2 or new_labels[ne] == 1 for ne in node_to_edges[i] if ne != e_idx) and \
                         any(new_labels[ne] == 2 or new_labels[ne] == 1 for ne in node_to_edges[j] if ne != e_idx)
            if has_label_2_or_1:
                new_labels[e_idx] = 2
    # Rule 4: For each edge with label 1, if at both end points there is an edge with label 2
    # then update its label to 2.
    for e_idx, (i, j) in enumerate(edges):
        if labels[e_idx] == 1:
            has_both_label2 = any(new_labels[ne] == 2 for ne in node_to_edges[i] if ne != e_idx) and \
                         any(new_labels[ne] == 2 for ne in node_to_edges[j] if ne != e_idx)
            if has_both_label2:
                new_labels[e_idx] = 2
    # Rule 5: For each edge with label 2, if one side is label 4 and other is label 1 or 2
    # then update its label to 3.
    for e_idx, (i, j) in enumerate(edges):
        if labels[e_idx] == 2:
            has_any_label4 = any(new_labels[ne] == 4 for ne in node_to_edges[i] if ne != e_idx) or \
                              any(new_labels[ne] == 4 for ne in node_to_edges[j] if ne != e_idx)
            if has_any_label4:
                new_labels[e_idx] = 3
    # Rule 6: For each edge with label 1, if one side is label 3 and other is label 1 or 2
    # then update its label to 2.
    for e_idx, (i, j) in enumerate(edges):
        if labels[e_idx] == 1:
            has_any_label3 = any(new_labels[ne] == 3 for ne in node_to_edges[i] if ne != e_idx) or \
                             any(new_labels[ne] == 3 for ne in node_to_edges[j] if ne != e_idx)
            if has_any_label3:
                new_labels[e_idx] = 2
    return new_labels


def fix_categorical_discrepancy_for_file(input_filepath, output_folder):
    """
    Process a single VTP file:
      - Load the graph,
      - Update edge labels according to our rules,
      - Save the updated graph to output_filepath.
    """
    mesh = pv.read(input_filepath)
    # Check that the edge attribute "radius" exists.
    if "radius" not in mesh.cell_data:
        print(f"File {input_filepath} does not have 'radius' attribute.")
        return

    # Update edge labels.
    updated_labels = update_edge_labels(mesh)

    # Assign the updated labels back to the mesh.
    mesh.cell_data["radius"] = updated_labels
    mesh.save(os.path.join(output_folder, os.path.basename(input_filepath)))

##############################################################################################
# Code to check how many times a validity criterion is broken
##############################################################################################


# Finally, a utility function to check the quality of our discretization
def extract_edges(mesh):
    """
    Extract edges from a PyVista mesh's lines array.
    Assumes each edge is represented as [2, i, j] (a 2-point line).

    Returns:
      edges: List of tuples (i, j) representing edges.
    """
    lines = mesh.lines.copy()  # a flat array
    edges = []
    idx = 0
    while idx < len(lines):
        n = int(lines[idx])
        if n != 2:
            raise ValueError("Expected each edge to have 2 points.")
        i, j = int(lines[idx + 1]), int(lines[idx + 2])
        edges.append((i, j))
        idx += n + 1
    return edges


def evaluate_discretization(mesh, threshold=1.0):
    """
    Evaluate the quality of discretization by checking for each node
    whether any two incident edges have a difference in the discretized
    radius value greater than the threshold.

    Args:
      mesh: A PyVista mesh with cell_data["radius"] containing the discretized radius.
      threshold: The maximum allowed difference (e.g., 1).

    Returns:
      discrepancies: A dictionary mapping node index to a list of pairs of edge indices
                     where the difference is greater than threshold.
      total_discrepancies: Total count of such discrepancies.
    """
    # Extract edges and corresponding discretized radii.
    edges = extract_edges(mesh)
    disc_radii = mesh.cell_data["radius"]

    # Build a dictionary mapping node -> list of (edge_index, radius)
    node_to_edges = {}
    for edge_idx, (i, j) in enumerate(edges):
        for node in (i, j):
            if node not in node_to_edges:
                node_to_edges[node] = []
            node_to_edges[node].append((edge_idx, disc_radii[edge_idx]))

    discrepancies = {}
    total_discrepancies = 0
    # For each node, check all pairs of incident edges.
    for node, incident in node_to_edges.items():
        if len(incident) < 2:
            continue  # nothing to compare if only one edge incident
        # Compare each pair of edges.
        for a in range(len(incident)):
            for b in range(a + 1, len(incident)):
                edge_idx_a, radius_a = incident[a]
                edge_idx_b, radius_b = incident[b]
                diff = abs(radius_a - radius_b)
                if diff > threshold:
                    if node not in discrepancies:
                        discrepancies[node] = []
                    discrepancies[node].append((edge_idx_a, edge_idx_b, diff))
                    total_discrepancies += 1
    return discrepancies, total_discrepancies

def check_quality_and_fix():
    discrete_radii_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete"
    discrete_radii_final_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified"
    counter = total_num = 0
    for split in ['train_data', 'test_data']:
        discrete_graph_folder = f"{discrete_radii_folder}/{split}"
        final_folder = f"{discrete_radii_final_folder}/{split}"
        os.makedirs(final_folder, exist_ok=True)
        print("Processing", discrete_graph_folder)
        for input_filename in tqdm(glob.glob(f"{discrete_graph_folder}/*.vtp")):
            graph = pv.read(input_filename)
            discrepancies, total = evaluate_discretization(graph, threshold=1.0)
            if total > 0:
                print(f"Found {total} discrepancies in {input_filename}")
                # Overwrite in the original folder
                fix_categorical_discrepancy_for_file(input_filename, final_folder)
                total_num += 1
            else:
                # We simply copy the file to the final folder
                shutil.copy2(input_filename, final_folder)
            counter += 1
    print(f"Found {total_num} graphs with discrepancies out of {counter} total graphs.")


##############################################################################################
# Just check violations and no fixing
##############################################################################################

def check_quality():
    print("-----------------------------------------------------")
    print("Running this after all the fixes")
    discrete_radii_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified"
    counter = total_num = 0
    for split in ['train_data', 'test_data']:
        discrete_graph_folder = f"{discrete_radii_folder}/{split}"
        print("Processing", discrete_graph_folder)
        for input_filename in tqdm(glob.glob(f"{discrete_graph_folder}/*.vtp")):
            graph = pv.read(input_filename)
            discrepancies, total = evaluate_discretization(graph, threshold=1.0)
            if total > 0:
                print(f"Found {total} discrepancies in {input_filename}")
                total_num += 1
            counter += 1
    print(f"Found {total_num} graphs with discrepancies out of {counter} total graphs.")


def remove_bad_samples():
    """
    These samples are recognized by eye-balling the graphs and checking if they are valid.
    We simply delete these really bad ones
    :return:
    """
    bad_list = [
        '007', '012', '013', '053', '067', '080', '087', '121',
        '160', '177', '179', '186', '187', '193', '194', '198',
        '220', '254', '646', '189', '203'
    ]
    discrete_radii_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified"
    for split in ['train_data', 'test_data']:
        discrete_graph_folder = f"{discrete_radii_folder}/{split}"
        for input_filename in tqdm(glob.glob(f"{discrete_graph_folder}/*.vtp")):
            if any(bad in input_filename for bad in bad_list):
                print(f"Removing {input_filename}")
                os.remove(input_filename)


if __name__ == "__main__":
    oversample_points()
    convert_to_mst()
    discretize_all_graphs()
    check_quality_and_fix()
    remove_bad_samples()
    check_quality()
