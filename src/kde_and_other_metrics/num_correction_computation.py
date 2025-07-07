import glob
import numpy as np
import os
import pyvista as pv

def edge_count(mesh: pv.PolyData) -> int:
    """
    Return the number of edges in a PolyData object.

    Fast path: if every cell is a 2-point line (`mesh.n_lines` available in
    PyVista ≥ 0.44).  Fallback walks the connectivity array and counts
    (n_pts − 1) per poly-line cell.
    """
    edges = np.asarray(mesh.lines.reshape(-1, 3))[:, 1:]
    edges_count = len(edges)
    return edges_count


def compute_num_edges(folder_path):
    # --- User parameters ---
    # Find all .vtp files in the folder
    vtp_files = glob.glob(os.path.join(folder_path, "*.vtp"))

    total_edges, per_file_edges = compute_total_num_edges(vtp_files)

    # Display per-file and aggregate counts
    print("Edges per file:")
    for filename, count in per_file_edges:
        print(f"{filename}: {count} edges")

    print(f"\nTotal edges across all VTP files: {total_edges}")
    return total_edges


def compute_total_num_edges(vtp_files):
    total_edges = 0
    per_file_edges = []
    for vtp_file in vtp_files:
        mesh = pv.read(vtp_file)
        # Get number of edges
        edges = np.asarray(mesh.lines.reshape(-1, 3))[:, 1:]
        edges_count = len(edges) // 2  # Since we generate only half of the edges and include the other half explicitly.
        per_file_edges.append((os.path.basename(vtp_file), edges_count))
        total_edges += edges_count
    return total_edges, per_file_edges


def compute_avg_number_of_edges(folder_path, dataset_name):
    valid_vtps = glob.glob(os.path.join(folder_path, "*.vtp"))
    if dataset_name is not None:
        valid_vtps = [f for f in valid_vtps if dataset_name in f]
    print(f"Number of valid VTPs: {len(valid_vtps)}")
    total_edges, per_file_edges = compute_total_num_edges(valid_vtps)
    avg_edges = total_edges / len(valid_vtps)
    print(f"Average number of edges: {avg_edges}")
    # Display per-file and aggregate counts
    print("Edges per file:")
    for filename, count in per_file_edges:
        print(f"{filename}: {count} edges")


def compute_avg_node_degree(folder_path, dataset_name):
    valid_vtps = glob.glob(os.path.join(folder_path, "*.vtp"))
    if dataset_name is not None:
        valid_vtps = [f for f in valid_vtps if dataset_name in f]
    print(f"Number of valid VTPs: {len(valid_vtps)}")
    avg_val = 0
    for file_path in valid_vtps:
        mesh = pv.read(file_path)
        E = edge_count(mesh)
        N = mesh.n_points  # number of vertices
        if N == 0:
            print(f"Skipping {file_path} due to zero vertices.")
            continue
        avg_val += 2.0 * E / N
    print(f"Average node degree: {avg_val/len(valid_vtps)}")

if __name__ == "__main__":
    folder_path = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified/train_data"
    compute_avg_number_of_edges(folder_path, None)
    compute_avg_node_degree(folder_path, None)