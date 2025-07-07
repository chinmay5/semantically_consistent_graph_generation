import glob
import os.path

import numpy as np
import pyvista
import networkx as nx
from scipy.spatial import cKDTree

link_pred_dataset_folder = '/mnt/elephant/chinmay/ATM22/atm_link_pred'
original_dataset_folder = '/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified/test_data'
predicted_samples = '/home/chinmayp/workspace/ConStruct/outputs/2025-02-22/10-46-42/check_something'

# Now, we check how many valid edges are proposed

def load_vtp_as_nx_grph(filename):
    vtk_data = pyvista.read(filename)
    graph = nx.Graph()

    edges = vtk_data.lines.reshape(-1, 3)
    nodes = np.asarray(vtk_data.points)
    nodes = nodes - np.mean(nodes, axis=0, keepdims=True)
    # Scale the point cloud
    max_distance = np.max(np.linalg.norm(nodes, axis=1))
    nodes = nodes / max_distance
    for edge in edges:
        graph.add_edge(edge[1], edge[2])
    # graph.add_nodes_from(nodes.tolist())
    for i, pos in enumerate(nodes):
        graph.add_node(i, pos=pos)
    return graph

def pos_to_node_idx(G):
    G_nodes = list(G.nodes())
    coords_G = np.array([G.nodes[node]['pos'] for node in G_nodes])
    return cKDTree(coords_G), coords_G

for filename in glob.glob(f'{original_dataset_folder}/*'):
    # Load all the networkx graphs.
    nx_graph_original = load_vtp_as_nx_grph(filename)
    nx_predicted = load_vtp_as_nx_grph(f'{predicted_samples}/{os.path.basename(filename)}')
    nx_incomplete = load_vtp_as_nx_grph(f'{link_pred_dataset_folder}/{os.path.basename(filename)}')
    # Now, computing the mapping between positions to node index
    orig_kd_tree, orig_nodes = pos_to_node_idx(nx_graph_original)
    # Now, we check how many of these graphs are present in the noisy version
    total = 0
    tolerance = 1e-1
    for u, v in nx_incomplete.edges():
        # Get the 3D coordinates for the two nodes.
        pos_u = tuple(nx_incomplete.nodes[u]['pos'])
        pos_v = tuple(nx_incomplete.nodes[v]['pos'])

        # Query the KDTree for the nearest node in G2 within the tolerance for both endpoints.
        d_u, idx_u = orig_kd_tree.query(pos_u, distance_upper_bound=tolerance)
        d_v, idx_v = orig_kd_tree.query(pos_v, distance_upper_bound=tolerance)

        # If no neighbor is found within tolerance, tree.query returns np.inf and an index of len(coords_G2).
        if np.isinf(d_u) or np.isinf(d_v) or idx_u >= len(nx_graph_original.edges) or idx_v >= len(nx_graph_original.edges):
            continue  # Skip this edge if one of the nodes isn't matched.

        # Map the indices from the KDTree back to G2's node IDs.
        node_u_original = orig_nodes[idx_u]
        node_v_original = orig_nodes[idx_v]

        # Check if G2 has an edge connecting these nodes.
        if nx_graph_original.has_edge(idx_u, idx_v):
            total += 1

    print(f"Found {total} edges")
    print(f"Downsampled has {len(nx_incomplete.edges)} edges")
    # Number of edges to predict
    num_edges_to_predict = len(nx_graph_original.edges) - len(nx_incomplete.edges)
    # Edges that are correctly predicted by the model
    all_matching_edges = len(set(nx_predicted.edges).intersection(set(nx_graph_original.edges)))
    matching_edges_predicted_by_model = all_matching_edges - len(nx_incomplete.edges)
    # Accuracy is correctly predicted edges by model / total edges to predict
    acc = matching_edges_predicted_by_model / num_edges_to_predict
    print(f"{filename} has {acc:.4f} accuracy")
