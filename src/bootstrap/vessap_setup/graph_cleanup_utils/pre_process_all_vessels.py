import glob
import io
import os

import numpy as np
import pyvista
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm


def load_vtp_graph(filename, string_buffer):
    vtk_data = pyvista.read(filename)
    nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    radius = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.float)
    edges = edges.T
    edges, radius = remove_self_loops(edge_index=edges, edge_attr=radius)
    degree = torch.bincount(edges[0, :], minlength=len(nodes)) + \
             torch.bincount(edges[1, :], minlength=len(nodes))
    graph_data = Data(x=nodes, edge_index=edges, edge_attr=radius)
    return graph_data, degree


def save_updated_vtp(pyG_data, filename, split_save_path):
    nodes, edge_index, edge_attr = pyG_data.x, pyG_data.edge_index, pyG_data.edge_attr
    mesh_edge = np.concatenate((np.int32(2 * np.ones((1, edge_index.shape[1]))), edge_index), 0)
    mesh = pyvista.UnstructuredGrid(mesh_edge.T, np.array([4] * len(edge_attr)), nodes.numpy())
    mesh.cell_data['radius'] = edge_attr.numpy()
    mesh_structured = mesh.extract_surface()
    base_filename = os.path.basename(filename)
    base_filename = base_filename.replace("sample", "processed")
    mesh_structured.save(os.path.join(split_save_path, base_filename))


def _remove_neg_radius_edges(pyGdata, filename, string_buffer):
    mask = pyGdata.edge_attr > 0
    edge_index_pruned = pyGdata.edge_index.T[mask]
    pyGdata.edge_index = edge_index_pruned.T
    pyGdata.edge_attr = pyGdata.edge_attr[mask]
    if mask.sum() != mask.shape[0]:
        # At least one edge has a negative radius
        string_buffer.write(f"{filename=} has negative edge. Removing these edges\n")


def _align_coordinartes(pyGdata):
    pos = pyGdata.x[:, :3]
    # Remove the COM
    pos = pos - torch.mean(pos, dim=0, keepdim=True)
    # We will also normalize the positions
    max_scale = torch.linalg.norm(pos, dim=1).max()
    pyGdata.x = pos / max_scale
    return pyGdata

def remove_neg_radius_edges_for_one_graph(filename, string_buffer, split_save_path):
    pyGdata, degree = load_vtp_graph(filename, string_buffer)
    # Remove the negative edges
    _align_coordinartes(pyGdata=pyGdata)
    save_updated_vtp(pyG_data=pyGdata, filename=filename, split_save_path=split_save_path)


def remove_neg_rad_edges_for_all_graphs(split_path, split_save_path, non_neg_rad_graph_loc, split):
    # Initialize an in-memory buffer
    buffer = io.StringIO()
    for filename in tqdm(glob.glob(f"{split_path}/*.vtp")):
        # We only remove the negative edges. Everything else remains.
        remove_neg_radius_edges_for_one_graph(filename, string_buffer=buffer, split_save_path=split_save_path)
    # Now let us save the information
    final_content = buffer.getvalue()
    # Close the buffer (not necessary for io.StringIO)
    buffer.close()
    # Write the final content to a text file
    with open(os.path.join(non_neg_rad_graph_loc, f"{split}_changelog.txt"), "w") as file:
        file.write(final_content)
