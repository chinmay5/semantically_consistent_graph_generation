import glob
import math
import os
import random
from collections import defaultdict

import numpy as np
import pyvista
import seaborn as sns
import torch
import torch_geometric.utils
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from src.ConStruct.analysis.topological_analysis import SimplicialComplex
from src.environment_setup import time_logging

FONTSIZE = 20


# Some utility functions that will be needed
def read_vtp_file(filename):
    vtk_data = pyvista.read(filename)
    nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    edges = edges.T
    if 'edge_type' in vtk_data.cell_data:
        radius = torch.tensor(np.asarray(vtk_data.cell_data['edge_type']), dtype=torch.float)
    elif 'labels' in vtk_data.cell_data:
        radius = torch.tensor(np.asarray(vtk_data.cell_data['labels']), dtype=torch.float)
    else:
        radius = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.float)
    # Let us normalize the node coordinates.
    # It is especially important for the ground truth data
    nodes = nodes - torch.mean(nodes, dim=0, keepdim=True)
    # Scale the point cloud
    max_distance = np.max(np.linalg.norm(nodes, axis=1))
    nodes = nodes / max_distance
    return nodes, edges, radius


def save_vtp_file(nodes, edges, radius, save_dir, filename):
    mesh_edge = np.concatenate((np.int32(2 * np.ones((1, edges.shape[0]))), edges.T), 0)
    mesh = pyvista.UnstructuredGrid(mesh_edge.T, np.array([4] * len(radius)), nodes.numpy())
    mesh.cell_data['radius'] = radius.numpy()
    mesh_structured = mesh.extract_surface()
    mesh_structured.save(os.path.join(save_dir, filename))


@time_logging
def plot_num_nodes_kde(folders):
    colors = ['r', 'g']
    labels = ['real', 'synth']
    plt.figure(figsize=(8, 6))
    num_nodes_list_all = []
    for idx, folder in enumerate(folders):
        kde, list_of_num_nodes = load_kde(folder=folder)
        num_nodes_list_all.append(list_of_num_nodes)
        # Generate a range of values for the PDF
        if idx == 0:
            x = np.linspace(min(list_of_num_nodes), max(list_of_num_nodes), 1000)

        plt.plot(x, kde(x), label=f"KDE_num_nodes_{labels[idx]}", color=colors[idx], alpha=max(0.5 ** idx, 0.5))
        plt.xlabel("Integer Values")
        plt.ylabel("Probability Density")
    plt.title("KDE for Number of nodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust subplots for better spacing
    plt.show()
    print(f"Kl divergence between nodes = {kl_divergence_between_lists(num_nodes_list_all[0], num_nodes_list_all[1])}")


@time_logging
def plot_num_edges_kde(folders, axes):
    colors = ['r', 'b', 'g']
    labels = ['real', 'baseline', 'ours']
    num_edges_list_all = []
    for idx, folder in enumerate(folders):
        kde, num_edges = load_edges_kde(folder=folder)
        num_edges_list_all.append(num_edges)
        # Generate a range of values for the PDF
        if idx == 0:
            x = np.linspace(min(num_edges), max(num_edges), 1000)

        axes[2, 1].plot(x, kde(x), label=f"KDE_num_edges_{labels[idx]}", color=colors[idx], alpha=max(0.5 ** idx, 0.5))
    axes[2, 1].grid(True)
    axes[2, 1].set_xlabel(r'|$\mathcal{E}$|', fontsize=FONTSIZE)
    axes[2, 1].set_ylabel('')
    print(
        f"KL divergence between num edges baseline {kl_divergence_between_lists(num_edges_list_all[0], num_edges_list_all[1])}")
    print(
        f"KL divergence between num edges ours {kl_divergence_between_lists(num_edges_list_all[0], num_edges_list_all[2])}")


def load_kde(folder):
    num_nodes_arr = _load_num_nodes(folder=folder)
    # Calculate the PMF
    unique_values, counts = np.unique(num_nodes_arr, return_counts=True)
    pmf = counts / len(num_nodes_arr)
    # Step 2: Create the PDF using KDE
    kde = gaussian_kde(num_nodes_arr)
    return kde, num_nodes_arr


def load_edges_kde(folder):
    num_edges_arr = _load_num_edges(folder=folder)
    kde = gaussian_kde(num_edges_arr)
    return kde, num_edges_arr


def _load_num_nodes(folder):
    num_nodes_arr = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = read_vtp_file(filename)
        num_nodes_arr.append(nodes.size(0))
    return num_nodes_arr


def _load_num_edges(folder):
    num_edges_arr = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = read_vtp_file(filename)
        if not torch_geometric.utils.is_undirected(edge_index=edges):
            edges = torch_geometric.utils.to_undirected(edge_index=edges)
        num_edges_arr.append(edges.size(1))
    return num_edges_arr


def generate_random_color(offset):
    return "#{:02x}{:02x}{:02x}".format(
        np.clip(random.randint(0, 128) + offset, a_min=0, a_max=255),  # R: 0-128
        np.clip(random.randint(128, 255) + offset, a_min=0, a_max=255),  # G: 128-255
        np.clip(random.randint(0, 255) + offset, a_min=0, a_max=255)  # B: 0-255
    )


def plot_num_nodes(folders):
    for idx, folder in enumerate(folders):
        num_nodes_real = _load_num_nodes(folder)
        plt.hist(num_nodes_real, bins=20, color=generate_random_color(offset=50 * idx), alpha=max(0.5 ** idx, 0.5))
    # Add labels and title
    plt.xlabel('Num nodes')
    plt.ylabel('Count')
    plt.title('Hist for node counts')
    # Show the plot
    plt.show()


def read_raw_coords(folder):
    all_coords = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        coords, _, _ = read_vtp_file(filename)
        coords = coords - torch.mean(coords, dim=0, keepdim=True)
        all_coords.append(coords)
    return torch.cat(all_coords)


def perform_coord_wise_kde(folders, axes, min_start, max_start):
    colors = ['r', 'b', 'g']
    labels = ['real', 'baseline', 'ours']
    list_x, list_y, list_z = [], [], []
    for idx, folder in enumerate(folders):
        coords = read_raw_coords(folder=folder)
        x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        x_coord_kde = gaussian_kde(x_coords.tolist())
        y_coord_kde = gaussian_kde(y_coords.tolist())
        z_coord_kde = gaussian_kde(z_coords.tolist())
        list_x.append(x_coords.tolist())
        list_y.append(y_coords.tolist())
        list_z.append(z_coords.tolist())
        pts = np.linspace(min_start, max_start, 2000)
        # pts = np.linspace(-400, 400, 2000)
        axes[0, 0].plot(pts, x_coord_kde(pts), label=f"KDE_x_coords_{labels[idx]}", color=colors[idx],
                        alpha=max(0.5 ** idx, 0.5))
        axes[0, 1].plot(pts, y_coord_kde(pts), label=f"KDE_y_coords_{labels[idx]}", color=colors[idx],
                        alpha=max(0.5 ** idx, 0.5))
        axes[0, 2].plot(pts, z_coord_kde(pts), label=f"KDE_z_coords_{labels[idx]}", color=colors[idx],
                        alpha=max(0.5 ** idx, 0.5))
    coord_kl_div = (kl_divergence_between_lists(list_x[0], list_x[1]) + kl_divergence_between_lists(list_y[0],
                                                                                                    list_y[1])
                    + kl_divergence_between_lists(list_z[0], list_z[1])) / 3
    print(f"Coord kl divergence with baseline = {coord_kl_div}")

    coord_kl_ours = (kl_divergence_between_lists(list_x[0], list_x[2]) + kl_divergence_between_lists(list_y[0],
                                                                                                     list_y[2])
                     + kl_divergence_between_lists(list_z[0], list_z[2])) / 3
    print(f"Coord kl divergence with ours = {coord_kl_ours}")
    # Turn on the grids
    axes[0, 0].grid(True)
    axes[0, 1].grid(True)
    axes[0, 2].grid(True)
    # Give the labels
    axes[0, 0].set_xlabel(r'$x$', fontsize=FONTSIZE)
    axes[0, 1].set_xlabel(r'$y$', fontsize=FONTSIZE)
    axes[0, 2].set_xlabel(r'$z$', fontsize=FONTSIZE)


def read_edge_angles(folder):
    edge_angles = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = read_vtp_file(filename)
        if not torch_geometric.utils.is_undirected(edges):
            edges, radius = torch_geometric.utils.to_undirected(edge_index=edges, edge_attr=radius)

        for idx in range(len(nodes)):
            p_a = nodes[idx]
            neighbors, _, _, _ = k_hop_subgraph(node_idx=idx, num_hops=1, edge_index=edges, relabel_nodes=False,
                                                flow='target_to_source')
            for i in neighbors:
                p_i = nodes[i]
                for j in neighbors:
                    if j == idx or i == j or i == idx:
                        continue
                    p_j = nodes[j]
                    v1 = p_i - p_a
                    v2 = p_j - p_a
                    assert not torch.isnan(v1).any()
                    assert not torch.isnan(v2).any()
                    prod = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-6)
                    if prod > 1:
                        print(f"Invalid angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                              f" {v2 / (torch.norm(v2) + 1e-6)}")
                    prod.clamp(min=0, max=1)
                    angle = torch.acos(prod)
                    if torch.isnan(angle).any():
                        print(f"Nan obtained in angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                              f" {v2 / (torch.norm(v2) + 1e-6)}")
                    else:
                        angle = angle * 180 / math.pi
                        edge_angles.append(angle)
    return edge_angles


def read_edge_lengths(folder):
    edge_lengths = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = read_vtp_file(filename)
        if torch_geometric.utils.is_undirected(edges):
            edges = torch_geometric.utils.to_undirected(edge_index=edges)
        nodes = nodes - torch.mean(nodes, dim=0, keepdim=True)
        edge_lengths.extend(torch.linalg.norm(nodes[edges[0]] - nodes[edges[1]], dim=1).tolist())
    return edge_lengths


def plot_edge_length_kde(folders, axes):
    colors = ['r', 'b', 'g']
    labels = ['real', 'baseline', 'ours']
    edge_length_arr = []
    for idx, folder in enumerate(folders):
        edge_length_array = read_edge_lengths(folder=folder)
        kde = gaussian_kde(edge_length_array)
        # Generate a range of values for the PDF
        edge_length_arr.append(edge_length_array)
        edge_max = max(edge_length_array)
        edge_min = min(edge_length_array)
        print(f"For {labels[idx]}: max edge length = {edge_max}")
        print(f"For {labels[idx]}: min edge length = {edge_min}")
        # Also compute % of edges less than 0.01 in thickness
        # if idx == 0:
        #     kde_plot_points = edge_max
        x = np.linspace(0, 1.25, 1000)
        axes[0, 0].plot(x, kde(x), color=colors[idx])
    axes[0, 0].grid(True)
    axes[0, 0].set_xlabel(r'$l_\mathcal{E}$', fontsize=FONTSIZE)
    print(
        f"kl divergence between edge lengths baseline {kl_divergence_between_lists(edge_length_arr[0], edge_length_arr[1])}")
    print(
        f"kl divergence between edge lengths ours {kl_divergence_between_lists(edge_length_arr[0], edge_length_arr[2])}")


def plot_edge_angles_kde(folders, axes):
    colors = ['r', 'b', 'g']
    labels = ['real', 'baseline', 'ours']
    edge_angle_max = 0
    kl_div_lists = []
    for idx, folder in enumerate(folders):
        edge_angles_for_graph = read_edge_angles(folder=folder)
        kl_div_lists.append(edge_angles_for_graph)
        kde = gaussian_kde(edge_angles_for_graph)
        # Generate a range of values for the PDF
        print(f"{labels[idx]} maximum angle = {max(edge_angles_for_graph)}")
        print(f"{labels[idx]} minimum angle = {min(edge_angles_for_graph)}")
        if idx == 0:
            edge_angle_max = max(edge_angles_for_graph)
        x = np.linspace(0, edge_angle_max, 1000)
        axes[0, 1].plot(x, kde(x), color=colors[idx], label=f"{labels[idx]}")
    axes[0, 1].grid(True)
    axes[0, 1].set_xlabel(r'$\mathcal{E}~\angle$', fontsize=FONTSIZE)
    print(f"Edge angles KL Div. baseline {kl_divergence_between_lists(kl_div_lists[0], kl_div_lists[1])}")
    print(f"Edge angles KL Div. Ours {kl_divergence_between_lists(kl_div_lists[0], kl_div_lists[2])}")


def read_edge_angles_along_axes(folder):
    angles_with_axes = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = read_vtp_file(filename)
        if not torch_geometric.utils.is_undirected(edges):
            edges, radius = torch_geometric.utils.to_undirected(edge_index=edges, edge_attr=radius)
        # Step 2: Identify pairs of nodes connected by each edge
        edge_pairs = edges.t()
        edge_vectors = nodes[edge_pairs[:, 1]] - nodes[edge_pairs[:, 0]]
        # Step 3: Compute vectors corresponding to each edge
        for vector in edge_vectors:
            for axis in torch.eye(3):
                dot_product = torch.dot(vector, axis)
                norm_vector = torch.norm(vector)
                norm_axis = torch.norm(axis)

                cosine_similarity = dot_product / (norm_vector * norm_axis)
                angle = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))

                # Convert angle to degrees
                angle_degrees = np.degrees(angle.item())
                angles_with_axes.append(angle_degrees)
    # Reshape the angles for each axis
    angles_with_axes = torch.tensor(angles_with_axes).reshape(-1, 3)
    return angles_with_axes


def percentile(t, qs):
    resulting_values = []
    for q in qs:
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        resulting_values.append(t.view(-1).kthvalue(k).values.item())
    return resulting_values


def _get_more_stats(value_list):
    if not isinstance(value_list, torch.Tensor):
        value_list = torch.as_tensor(value_list)
    # Calculate statistics
    mean_angle = torch.mean(value_list)
    median_angle = torch.median(value_list)
    std_dev_angle = torch.std(value_list)
    min_angle = torch.min(value_list)
    max_angle = torch.max(value_list)
    q25, q50, q75 = percentile(value_list, [25, 50, 75])
    return f"mean={mean_angle:.3f}; median={median_angle:.3f}; std_dev= {std_dev_angle:.3f};" \
           f" {q25=:.3f}; {q50=:.3f}; {q75=:.3f}; min={min_angle:.3f}; max={max_angle:.3f}"


def plot_edge_angles_along_axes_kde(folders, axes):
    """
    Computes edge angles along coordinate axes for only the graphs with degree 2
    :param folders: list[folder]
    :return: None
    """
    colors = ['r', 'b', 'g']
    labels = ['real', 'baseline', 'ours']
    max_angle = 0
    angle_lists1, angle_lists2, angle_lists3 = [], [], []
    for idx, folder in enumerate(folders):
        edge_angles_for_graph = read_edge_angles_along_axes(folder=folder)
        # We will get three kde plots here since we are working along the three axes
        x_axis_angles, y_axis_angles, z_axis_angles = edge_angles_for_graph[:, 0], edge_angles_for_graph[:,
                                                                                   1], edge_angles_for_graph[:, 2]
        x_axis_kde = gaussian_kde(x_axis_angles.tolist())
        y_axis_kde = gaussian_kde(y_axis_angles.tolist())
        z_axis_kde = gaussian_kde(z_axis_angles.tolist())
        angle_lists1.append(x_axis_angles)
        angle_lists2.append(y_axis_angles)
        angle_lists3.append(z_axis_angles)
        if idx == 0:
            max_angle = torch.max(edge_angles_for_graph).item()
        pts = np.linspace(0, max_angle, 1000)
        axes[1, 0].plot(pts, x_axis_kde(pts), color=colors[idx], alpha=max(0.5 ** idx, 0.5))
        axes[1, 1].plot(pts, y_axis_kde(pts), color=colors[idx], alpha=max(0.5 ** idx, 0.5))
        axes[1, 2].plot(pts, z_axis_kde(pts), color=colors[idx], alpha=max(0.5 ** idx, 0.5))
    # Turning on the grid
    axes[1, 0].grid(True)
    axes[1, 1].grid(True)
    axes[1, 2].grid(True)
    # We give the labels
    axes[1, 0].set_xlabel(r'$\theta$', fontsize=FONTSIZE)
    axes[1, 1].set_xlabel(r'$\phi$', fontsize=FONTSIZE)
    axes[1, 2].set_xlabel(r'$\psi$', fontsize=FONTSIZE)
    # Calling kl divergence on all the three lists
    angle_kl = ((kl_divergence_between_lists(angle_lists1[0], angle_lists1[1]) +
                 kl_divergence_between_lists(angle_lists2[0], angle_lists2[1])
                 + kl_divergence_between_lists(angle_lists3[0], angle_lists3[1]))) / 3
    print(f"kl divergence for edge angles along axes baseline {angle_kl}")
    angle_kl_ours = ((kl_divergence_between_lists(angle_lists1[0], angle_lists1[2]) +
                      kl_divergence_between_lists(angle_lists2[0], angle_lists2[2])
                      + kl_divergence_between_lists(angle_lists3[0], angle_lists3[2]))) / 3
    print(f"kl divergence for edge angles along axes ours {angle_kl_ours}")


def compute_histogram(data, num_bins, bin_min, bin_max):
    histogram = torch.histc(data, bins=num_bins, min=bin_min, max=bin_max)
    return histogram / histogram.sum()


def kl_divergence_between_lists(data1, data2, num_bins=10, bin_min=None, bin_max=None):
    """
    Compute the KL divergence between two lists of unequal lengths using PyTorch.
    """
    print("Make sure first list is the GT")
    # Determine bins based on the combined range of both lists
    data1_tensor = torch.as_tensor(data1, dtype=torch.float)
    data2_tensor = torch.as_tensor(data2, dtype=torch.float)

    if bin_min is None and bin_max is None:
        bin_min, bin_max = data1_tensor.min(), data1_tensor.max()
    # Compute histograms
    histogram1 = compute_histogram(data1_tensor, num_bins, bin_min, bin_max)
    histogram2 = compute_histogram(data2_tensor, num_bins, bin_min, bin_max)

    # Compute KL divergence
    # Second term is the ground truth and the first term is the prediction
    # Adding a very small constant to make sure we do not get nan
    kl_div = torch.nn.functional.kl_div(torch.log(histogram2 + 1e-15), histogram1)
    return kl_div.item()  # Convert to Python scalar


def compute_betti_vals(edge_index):
    edges = edge_index.T
    edge_list = edges.numpy().tolist()
    betti_val_info = SimplicialComplex(edge_list)
    bettis = defaultdict(list)
    for betti_number in [0, 1]:
        # A counter is internally a dictionary
        val = betti_val_info.betti_number(betti_number)
        bettis[betti_number].append(val)
    return bettis


def betti_0_and_1_plots(folders, axes):
    print("Computation made on undirected graphs")
    colors = ['tab:red', 'tab:blue', 'tab:green']
    labels = ['real', 'baseline', 'ours']
    for idx, folder in enumerate(folders):
        betti_0, betti_1 = load_betti_numbers(folder)
        # Very fragile code. We assume that the betti numbers are in the same order
        if idx == 0:
            betti_0_real = betti_0
            betti_1_real = betti_1
        elif idx == 1:
            betti_0_baseline = betti_0
            betti_1_baseline = betti_1
        elif idx == 2:
            betti_0_ours = betti_0
            betti_1_ours = betti_1
        else:
            raise ValueError("Invalid index")
        # Define the bin ranges
        bins_betti_0 = np.arange(0, 5) - 0.5
        bins_betti_1 = np.arange(0, 5) - 0.5
        sns.histplot(betti_0, bins=bins_betti_0, ax=axes[1, 1], stat='density', kde=False,
                     color=colors[idx], alpha=max(0.5 ** idx, 0.5), edgecolor='none', label=f"{labels[idx]}")
        sns.histplot(betti_1, bins=bins_betti_1, ax=axes[1, 2], stat='density', kde=False,
                     color=colors[idx], alpha=max(0.5 ** idx, 0.5), edgecolor='none', label=f"{labels[idx]}")
    axes[1, 1].grid(True)
    axes[1, 2].grid(True)
    axes[1, 1].set_xlabel(r'$\it{\beta_0}$', fontsize=FONTSIZE)
    axes[1, 2].set_xlabel(r'$\it{\beta_1}$', fontsize=FONTSIZE)
    axes[1, 1].set_ylabel('')
    axes[1, 2].set_ylabel('')

    print(f"Baseline Min: {min(betti_0_baseline)} and Max: {max(betti_0_baseline)}")
    print(f"Ours Min {min(betti_0_ours)} and Max: {max(betti_0_ours)}")
    print(f"GT Min: {min(betti_0_real)} and Max: {max(betti_0_real)}")

    print(
        f"Betti 0 KL divergence baseline {kl_divergence_between_lists(betti_0_real, betti_0_baseline, num_bins=5, bin_min=1, bin_max=10)}")
    print(
        f"Betti 1 KL divergence baseline {kl_divergence_between_lists(betti_1_real, betti_1_baseline, num_bins=3, bin_min=0, bin_max=3.5)}")

    # Doing the same for our method
    print(
        f"Betti 0 KL divergence ours {kl_divergence_between_lists(betti_0_real, betti_0_ours, num_bins=5, bin_min=1, bin_max=10)}")
    print(
        f"Betti 1 KL divergence ours {kl_divergence_between_lists(betti_1_real, betti_1_ours, num_bins=3, bin_min=0, bin_max=3.5)}")


def load_betti_numbers(folder):
    betti_0, betti_1 = [], []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = read_vtp_file(filename)
        edges = torch_geometric.utils.to_undirected(edge_index=edges)
        bettis = compute_betti_vals(edge_index=edges)
        betti_0.extend(bettis[0])
        x = bettis[1]
        betti_1.extend(x)
    return betti_0, betti_1


def _load_node_degrees(folder):
    all_degree_list = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = read_vtp_file(filename)
        if not torch_geometric.utils.is_undirected(edges):
            edges = torch_geometric.utils.to_undirected(edge_index=edges)
        dense_adj = torch_geometric.utils.to_dense_adj(edges, max_num_nodes=nodes.size(0)).squeeze(0)
        # degree = torch_geometric.utils.degree(index=edges[0], num_nodes=len(nodes))
        degree = dense_adj.sum(dim=1) + dense_adj.sum(dim=0)
        all_degree_list.extend(degree.tolist())
    return all_degree_list


def perform_node_degree_kde(folders, axes):
    print("Using dense adj for undirected graph degree computation")
    colors = ['r', 'b', 'g']
    labels = ['real', 'baseline', 'ours']
    degrees = []
    for idx, folder in enumerate(folders):
        degree_list = _load_node_degrees(folder=folder)
        kde = gaussian_kde(degree_list)
        degrees.append(degree_list)
        x = np.linspace(0, 10, 1000)
        axes[2, 0].plot(x, kde(x), label=f"KDE_{labels[idx]}", color=colors[idx], alpha=max(0.5 ** idx, 0.5))
    axes[2, 0].grid(True)
    axes[2, 0].set_ylabel('')
    axes[2, 0].set_xlabel(r'deg($\mathcal{V}$)', fontsize=FONTSIZE)
    print(f"kl divergence between node degrees baseline {kl_divergence_between_lists(degrees[0], degrees[1])}")
    print(f"kl divergence between node ours {kl_divergence_between_lists(degrees[0], degrees[2])}")


def plot_node_degree_staircase(folders, axes):
    print("Using dense adj for undirected graph degree computation")
    colors = ['tab:purple', 'tab:orange', 'tab:green']
    labels = ['real', 'synth']
    print("Reducing degree list by a factor of 2")
    list_of_dataset_degree = []
    for idx, folder in enumerate(folders):
        degree_list = _load_node_degrees(folder=folder)
        # Reducing the degree list
        degree_list = [x / 2 for x in degree_list]
        list_of_dataset_degree.append(degree_list)
        if idx == 0:
            alpha = 0.6
        elif idx == 1:
            alpha = 0.5
        else:
            alpha = 0.4
        # Define the bin range
        bins = np.arange(6) - 0.5
        sns.histplot(degree_list, bins=bins, ax=axes[1, 0], stat='density', kde=False, color=colors[idx],
                     alpha=alpha, edgecolor='none')
    axes[1, 0].grid(True)
    axes[1, 0].set_ylabel('')
    axes[1, 0].set_xlabel(r'deg($\mathcal{V}$)', fontsize=FONTSIZE)
    print(
        f"kl divergence between node degrees baseline {kl_divergence_between_lists(list_of_dataset_degree[0], list_of_dataset_degree[1])}")
    print(
        f"kl divergence between node ours {kl_divergence_between_lists(list_of_dataset_degree[0], list_of_dataset_degree[2])}")


def plot_num_edges_staircase(folders, axes):
    colors = ['tab:purple', 'tab:orange', 'tab:green']
    labels = ['real', 'synth']
    print("reducing num edge by a factor of 2")
    list_of_dataset_edges = []
    for idx, folder in enumerate(folders):
        _, num_edges = load_edges_kde(folder=folder)
        num_edges = [x / 2 for x in num_edges]
        list_of_dataset_edges.append(num_edges)
        # Generate a range of values for the PDF
        # h, edges = np.histogram(num_edges, bins=np.linspace(0, 20, 10))
        # axes[2, 1].stairs(h, edges, color=colors[idx])
        if idx == 0:
            alpha = 0.6
        elif idx == 1:
            alpha = 0.5
        else:
            alpha = 0.4
        bins = np.arange(35, 120)
        sns.histplot(num_edges, bins=bins, ax=axes[0, 2], stat='density', kde=False,
                     color=colors[idx], alpha=alpha, edgecolor='none')
    axes[0, 2].grid(True)
    axes[0, 2].set_xlabel(r'|$\mathcal{E}$|', fontsize=FONTSIZE)
    axes[0, 2].set_ylabel('')
    # Doing a reverse operation since the GT does not have any cycles at all
    # Doing the normal KL would give nan value.
    print(
        f"kl divergence between num edges baseline {kl_divergence_between_lists(list_of_dataset_edges[0], list_of_dataset_edges[1])}")
    print(
        f"kl divergence between num edges ours {kl_divergence_between_lists(list_of_dataset_edges[0], list_of_dataset_edges[2])}")


def atm_stats():
    print("Checking ATM stats")
    real_folder = '/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified/train_data'
    baseline = '/home/chinmayp/workspace/ConStruct/outputs/2025-02-23/atm_midi_baseline'
    ours = '/home/chinmayp/workspace/ConStruct/outputs/2025-02-23/atm_ours/generated_samples'
    # ours = '/home/chinmayp/workspace/ConStruct/outputs/atm_final_samples/multi_label_proj_ours'
    # We compare the results
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Continuous features we want to plot
    plot_edge_length_kde(folders=[real_folder, baseline, ours], axes=axes)
    plot_edge_angles_kde(folders=[real_folder, baseline, ours], axes=axes)

    # The discrete features
    plot_node_degree_staircase(folders=[real_folder, baseline, ours], axes=axes)
    plot_num_edges_staircase(folders=[real_folder, baseline, ours], axes=axes)
    betti_0_and_1_plots(folders=[real_folder, baseline, ours], axes=axes)

    # Get the handle and the final figure
    # Add legend
    handles, labels = axes[0, 1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1.05), loc='upper left', borderaxespad=0., fontsize=16)
    plt.tight_layout()
    plt.savefig('atm.jpg', format='jpg', dpi=250)
    plt.show()


if __name__ == '__main__':
    atm_stats()
