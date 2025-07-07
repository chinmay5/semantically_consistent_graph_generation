import os
import shutil
import sys
import warnings

import numpy as np
import pyvista
import torch
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, contains_isolated_nodes, contains_self_loops

warnings.filterwarnings("ignore")


def _save_vtp_file(nodes, edges, radius, filename, save_dir):
    mesh_edge = np.concatenate((np.int32(2 * np.ones((1, edges.shape[0]))), edges.T), 0)
    mesh = pyvista.UnstructuredGrid(mesh_edge.T, np.array([4] * len(radius)), nodes)
    mesh.cell_data['radius'] = radius
    mesh_structured = mesh.extract_surface()
    mesh_structured.save(os.path.join(save_dir, filename))


def read_vtp_file_as_pyg_data(filename, buffer, make_undirected):
    vtk_data = pyvista.read(filename)
    nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    # bugfix. Earlier I was loading the value as a long tensor and thus, it was already truncated.
    # Hence, the ceil function essentially did nothing to it.
    radius = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.float)
    edge_attr = radius
    # Now we will make the graph undirected
    edges = edges.T
    graph_data = Data(x=nodes, edge_index=edges, edge_attr=edge_attr, cont_radius=radius)
    if contains_isolated_nodes(graph_data.edge_index, num_nodes=graph_data.size(0)):
        buffer.write(f"contains isolated nodes\n")
        # Finally, we remove the self loops, in case there are any.
    if contains_self_loops(graph_data.edge_index):
        buffer.write(f"contains self loops. Removing them\n")
        graph_data.edge_index, graph_data.edge_attr = remove_self_loops(graph_data.edge_index, graph_data.edge_attr)

    if make_undirected and not torch_geometric.utils.is_undirected(graph_data.edge_index):
        # Making the graph undirected
        new_edges, new_edge_attr = torch_geometric.utils.to_undirected(edge_index=graph_data.edge_index,
                                                                       edge_attr=graph_data.edge_attr,
                                                                       num_nodes=graph_data.x.size(0),
                                                                       reduce='min')
    else:
        new_edges, new_edge_attr = graph_data.edge_index, graph_data.edge_attr
    # All the processing wrt edges is now done.
    graph = Data(x=nodes, edge_index=new_edges, edge_attr=new_edge_attr)
    degree = torch_geometric.utils.degree(index=graph.edge_index[1],
                                          num_nodes=graph.size(0), dtype=torch.long)
    if torch.any(new_edge_attr < 0.1):
        print(f"Something wrong with edge_attr for {filename}")
        sys.exit(0)
    return graph, degree


def assign_edge_type(radius):
    # Based on the radius values, we will assign the edge type
    discrete_radius = radius.clone()
    discrete_radius[discrete_radius < 2] = 1
    discrete_radius[(2 < discrete_radius) & (discrete_radius <= 3)] = 2
    discrete_radius[(3 < discrete_radius) & (discrete_radius <= 5)] = 3
    discrete_radius[discrete_radius > 5] = 4
    return discrete_radius.to(torch.long)


def convert_vtp_graph_to_pyg_and_save_as_pt(filename, idx, buffer, save_path, make_undirected):
    buffer.write(f"Processing {filename=}\n")
    graph_data, degree = read_vtp_file_as_pyg_data(filename=filename, buffer=buffer, make_undirected=make_undirected)
    num_nodes = graph_data.x.size(0)
    pos = graph_data.x[:, :3]
    if not torch.mean(pos, dim=0, keepdim=True).abs().max() < 1e-3:
        print(f"COM is not zero for {filename}. Value is {torch.mean(pos, dim=0, keepdim=True)}")
        print("We delete this sample so that it does not ruin our statistics")
        os.remove(filename)
    # We will also normalize the positions
    graph_data.pos = pos
    graph_data.x = torch.ones(num_nodes, 1, dtype=torch.float)
    # Now, we can go ahead and discretize the continuous valued radius to its discrete range
    # Define the interval for discretization
    cont_edge_radius = graph_data.edge_attr
    edge_attr_discrete = assign_edge_type(cont_edge_radius)
    graph_data.edge_attr = edge_attr_discrete
    graph_data.edge_radius = cont_edge_radius

    save_dir = os.path.join(save_path, "..", "vtk_sample_viz")
    filename = os.path.basename(filename)
    if idx <= 10:
        os.makedirs(save_dir, exist_ok=True)
        radius = graph_data.edge_attr.numpy()
        _save_vtp_file(graph_data.pos.numpy(), graph_data.edge_index.T.numpy(), radius,
                       f"{filename.replace('.vtp', '_checked.vtp')}", save_dir)
    filename_pt = filename.replace(".vtp", ".pt")
    torch.save(graph_data, os.path.join(save_path, filename_pt))


def convert_vtp_graph_to_pyg_and_save_as_pt_binary(filename, idx, buffer, save_path, make_undirected):
    buffer.write(f"Processing {filename=}\n")
    graph_data, degree = read_vtp_file_as_pyg_data(filename=filename, buffer=buffer, make_undirected=make_undirected)
    num_nodes = graph_data.x.size(0)
    pos = graph_data.x[:, :3]
    if not torch.mean(pos, dim=0, keepdim=True).abs().max() < 1e-3:
        print(f"COM is not zero for {filename}. Value is {torch.mean(pos, dim=0, keepdim=True)}")
        print("We delete this sample so that it does not ruin our statistics")
        os.remove(filename)
    # We will also normalize the positions
    graph_data.pos = pos
    graph_data.x = torch.ones(num_nodes, 1, dtype=torch.float)
    # Now, we can go ahead and discretize the continuous valued radius to its discrete range
    # Define the interval for discretization
    cont_edge_radius = graph_data.edge_attr
    edge_attr_binary = (cont_edge_radius > 0).to(torch.long)
    graph_data.edge_attr = edge_attr_binary
    graph_data.edge_radius = cont_edge_radius

    filename = os.path.basename(filename)
    filename_pt = filename.replace(".vtp", ".pt")
    torch.save(graph_data, os.path.join(save_path, filename_pt))