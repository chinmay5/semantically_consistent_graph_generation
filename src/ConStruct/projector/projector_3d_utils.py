import abc

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from src.ConStruct.utils import PlaceHolder

def get_adj_matrix(z_t):
    z_t_adj = torch.argmax(z_t.E, dim=3)
    z_t_adj[z_t_adj != 0] = 1  # not interested in the different edge types
    return z_t_adj


class AbstractProjector(abc.ABC):
    @abc.abstractmethod
    def valid_graph_fn(self, nx_graph):
        pass

    @property
    @abc.abstractmethod
    def can_block_edges(self):
        pass

    def __init__(self, z_t: PlaceHolder):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        if self.can_block_edges:
            # Which edges of the graph should be blocked.
            # These edges would not be considered a candidate for addition.
            # The process improves sampling speed since we need not check these invalid edges in subsequent steps.
            self.blocked_edges = {graph_idx: {} for graph_idx in range(self.batch_size)}

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_adj_matrix(z_t)
        assert (self.z_t_adj == 0).all()  # no edges in the planar limit dist

        # add data structure where planarity is checked
        for graph_idx in range(self.batch_size):
            # We iterate over each of the graphs and create an empty graph with node coordinate information.
            num_nodes = z_t.node_mask[graph_idx].sum()
            pos = z_t.pos[graph_idx, :num_nodes]
            nx_graph = nx.empty_graph(num_nodes)
            # Assign 3D coordinates to nodes
            node_positions =  {i: pos[i] for i in range(num_nodes)}
            nx.set_node_attributes(nx_graph, node_positions, name="pos")
            # An empty graph is always a valid graph
            assert self.valid_graph_fn(nx_graph)
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list
            if self.can_block_edges:
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][(node_1_idx, node_2_idx)] = False  # None of the edges are blocked.

    def project(self, z_s: PlaceHolder):
        # find added edges
        z_s_adj = get_adj_matrix(z_s)
        diff_adj = z_s_adj - self.z_t_adj
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        new_edges = diff_adj.nonzero(as_tuple=False)
        # add new edges and check planarity
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            old_nx_graph = nx_graph.copy()
            edges_to_add = (
                new_edges[
                    torch.logical_and(
                        new_edges[:, 0] == graph_idx,  # Select edges of the graph
                        new_edges[:, 1] < new_edges[:, 2],  # undirected graph
                    )
                ][:, 1:]
                .cpu()
                .numpy()
            )

            # TODO: add here counter for number of edges rejected

            # If we can block edges, we do it
            if self.can_block_edges:
                not_blocked_edges = []
                for edge in edges_to_add:
                    # Check if the edge has already been blocked.
                    if self.blocked_edges[graph_idx][tuple(edge)]:
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                    else:
                        # Only non blocked edges are the candidates.
                        not_blocked_edges.append(edge)
                edges_to_add = np.array(not_blocked_edges)

            # If no edges to add, we skip the graph
            if len(edges_to_add) == 0:
                continue

            # First try add all edges (we might be lucky)
            if len(edges_to_add) > 1:  # avoid repetition of steps
                nx_graph.add_edges_from(edges_to_add)

            # If it fails, we go one by one and delete the planarity breakers
            if not self.valid_graph_fn(nx_graph) or len(edges_to_add) == 1:
                # print(
                #     f"Planarity break - chain_idx: {s_int}, num edges:{len(edges_to_add)}"
                # )
                nx_graph = old_nx_graph
                # Try to add edges one by one (in random order)
                np.random.shuffle(edges_to_add)
                for edge in edges_to_add:
                    old_nx_graph = nx_graph.copy()
                    nx_graph.add_edge(edge[0], edge[1])
                    if not self.valid_graph_fn(nx_graph):
                        nx_graph = old_nx_graph
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # this edge breaks validity.
                        # Blocking it for future steps.
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][tuple(edge)] = True
                self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

            # Check that nx graphs is correctly stored
            assert (
                nx.to_numpy_array(nx_graph)
                == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all()

            # Check that adjacency matrices are the same in torch and nx
            num_nodes = self.nx_graphs_list[graph_idx].number_of_nodes()
            assert (
                get_adj_matrix(z_s)[graph_idx].cpu().numpy()[:num_nodes, :num_nodes]
                == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all()

        # store modified z_s
        # This is why we kept putting in-place updates for the z_s graphs.
        # The updated graph information is obtained from the final processing over the z_s tensor.
        self.z_t_adj = get_adj_matrix(z_s)


def has_no_cycles(nx_graph):
    # Tree have n-1 edges
    if nx_graph.number_of_edges() >= nx_graph.number_of_nodes():
        return False
    return nx.is_forest(nx_graph)



class TreeProjector(AbstractProjector):
    def valid_graph_fn(self, nx_graph):
        return has_no_cycles(nx_graph)

    @property
    def can_block_edges(self):
        return True


# Extra classes for vascular graph structure
class BaseValidProjector(AbstractProjector):


    def project(self, z_s: PlaceHolder):
        # find added edges
        z_s_adj = get_adj_matrix(z_s)
        diff_adj = z_s_adj - self.z_t_adj
        assert (diff_adj >= 0).all(), "Once unmasked, the edges cannot be removed."

    @property
    def can_block_edges(self):
        return True

    def valid_graph_fn(self, nx_graph) -> bool:
        return True


def get_multi_class_adj_matrix(z_t):
    z_t_adj = torch.argmax(z_t.E, dim=3)
    return z_t_adj

# Defining the Projector for CoW

class TopCoWLineGraphCheck():
    def __init__(self, device):
        super(TopCoWLineGraphCheck, self).__init__()
        #                                             1  2  3  4  5  6  7  8  9  10 11 12 13
        self.valid_adjacency = torch.as_tensor([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                                                     [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2
                                                     [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3
                                                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # 4
                                                     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 5
                                                     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],  # 6
                                                     [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # 7
                                                     [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # 8
                                                     [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],  # 9
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # 10
                                                     [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1],  # 11
                                                     [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],  # 12
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]], device=device)  # 13
        self.check_symmetric(self.valid_adjacency)
        self.counter = 0

    @staticmethod
    def check_symmetric(array: torch.Tensor) -> None:
        """
        Verifies if the given array is symmetric. Raises an assertion error if not symmetric
        :param array: Adjacency matrix B, N, N
        :return: None
        """
        assert torch.allclose(array, array.T), "Not symmetric"


class CoWProjector():

    def __init__(self, z_t: PlaceHolder, device: torch.device):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        self.top_cow_validity_check_and_fix_obj = TopCoWLineGraphCheckAndFixModule(device)
        if self.can_block_edges:
            # Which edges of the graph should be blocked.
            # These edges would not be considered a candidate for deletion.
            # The process improves sampling speed since we need not check these invalid edges in subsequent steps.
            self.blocked_edges = {graph_idx: {} for graph_idx in range(self.batch_size)}

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_multi_class_adj_matrix(z_t)

        assert (self.z_t_adj == 0).all()  # no edges in the planar limit dist
        # add data structure where validity is checked
        for graph_idx in range(self.batch_size):
            # We iterate over each of the graphs and create an empty graph with node coordinate information.
            num_nodes = z_t.node_mask[graph_idx].sum()
            pos = z_t.pos[graph_idx, :num_nodes]
            # We create a complete graph with the given number of nodes.
            nx_graph = nx.empty_graph(num_nodes)
            # Assign 3D coordinates to nodes
            node_positions =  {i: pos[i] for i in range(num_nodes)}
            nx.set_node_attributes(nx_graph, node_positions, name="pos")
            # An fully-connected graph is always a valid graph
            assert self.try_fix_and_validate_fn(nx_graph)
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list
            if self.can_block_edges:
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][(node_1_idx, node_2_idx)] = False  # None of the edges are blocked.

    def project(self, z_s: PlaceHolder):
        # find added edges
        num_classes = z_s.E.shape[-1]
        z_s_adj = get_multi_class_adj_matrix(z_s)  # B, N , N
        diff_adj = z_s_adj - self.z_t_adj
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        new_edges = diff_adj.nonzero(as_tuple=False)
        # add new edges and check validity
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            old_nx_graph = nx_graph.copy()
            edge_indices = (
                new_edges[
                    torch.logical_and(
                        new_edges[:, 0] == graph_idx,  # Select edges of the graph
                        new_edges[:, 1] < new_edges[:, 2],  # undirected graph
                    )
                ]
                .cpu()
                .numpy()
            )
            edges_to_add = edge_indices[:, 1:]
            if len(edges_to_add) > 0:
                edge_attrs = z_s_adj.cpu().numpy()[edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2]]
                candidate_attrs = z_s.extra_info.cpu().numpy()[edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2]]  # B, N, N, top_k

            # If we can block edges, we do it
            if self.can_block_edges and len(edges_to_add) > 0:
                not_blocked_edges, not_blocked_attrs, not_blocked_cands = [], [], []
                for edge, edge_attr, cands in zip(edges_to_add, edge_attrs, candidate_attrs):
                    # Check if the edge has already been blocked.
                    if self.blocked_edges[graph_idx][tuple(edge)]:
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                    else:
                        # Only non blocked edges are the candidates.
                        not_blocked_edges.append(edge)
                        not_blocked_attrs.append(edge_attr)
                        not_blocked_cands.append(cands)
                edges_to_add = np.array(not_blocked_edges)
                edge_attrs = np.array(not_blocked_attrs)
                candidate_attrs = np.array(not_blocked_cands)
            # If no edges to delete, we skip the graph
            if len(edges_to_add) == 0:
                continue

            # First try add all edges (we might be lucky)
            if len(edges_to_add) > 1:  # avoid repetition of steps
                # Add all the edges to the graph
                all_possible_new_edges_to_add = []
                for (u, v), label, cx in zip(edges_to_add, edge_attrs, candidate_attrs):
                    # nx_graph.add_edge(u, v, label=label, cands=cx)
                    # A tuple of three elements is added to the list.
                    all_possible_new_edges_to_add.append(
                        ((u, v), label, cx)
                    )
                nx_graph.graph['all_possible_new_edges_to_add'] = all_possible_new_edges_to_add
                # We check if all the edges are valid.
                if self.try_fix_and_validate_fn(nx_graph):
                    # The graph labels might have changed though.
                    # So, let us update the labels in the z_s tensor.
                    for (u, v) in edges_to_add:
                        new_label = nx_graph[u][v]['label']
                        z_s.E[graph_idx, u, v, :] = F.one_hot(torch.tensor(new_label), num_classes=num_classes)
                        z_s.E[graph_idx, v, u, :] = F.one_hot(torch.tensor(new_label), num_classes=num_classes)
            # If num_edges > 1, we do not have the `all_possible_new_edges_to_add` attribute.
            # However, we rely on the shot-circuit nature of the `or` operator.
            if len(edges_to_add) == 1 or not self.try_fix_and_validate_fn(nx_graph):
                nx_graph = old_nx_graph
                # Try to add edges one by one (in random order)
                np.random.shuffle(edges_to_add)
                for edge, label, cands in zip(edges_to_add, edge_attrs, candidate_attrs):
                    old_nx_graph = nx_graph.copy()
                    # nx_graph.add_edge(edge[0], edge[1], label=label, cands=cands)
                    nx_graph.graph['all_possible_new_edges_to_add'] = [
                        ((edge[0], edge[1]), label, cands)
                    ]  # A list of tuples
                    if not self.try_fix_and_validate_fn(nx_graph):
                        nx_graph = old_nx_graph
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # this edge breaks validity.
                        # Blocking it for future steps.
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][tuple(edge)] = True
                    else:
                        # The graph is valid. However, the label may have been updated
                        # So, we update the label in the z_s tensor.
                        updated_label = nx_graph[edge[0]][edge[1]]['label']
                        z_s.E[graph_idx, edge[0], edge[1], :] = F.one_hot(torch.tensor(updated_label), num_classes=num_classes)
                        z_s.E[graph_idx, edge[1], edge[0], :] = F.one_hot(torch.tensor(updated_label), num_classes=num_classes)
                # Need to update the graph in the list.
                # Needs to be done since the original graph is modified in place.
                # It turned out to be invalid, so we replace it.
                self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

            # Check that nx graphs is correctly stored
            assert (
                    nx.to_numpy_array(nx_graph)
                    == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all()

            # Check that adjacency matrices are the same in torch and nx
            num_nodes = self.nx_graphs_list[graph_idx].number_of_nodes()
            assert (
                    get_adj_matrix(z_s)[graph_idx].cpu().numpy()[:num_nodes, :num_nodes]
                    == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all()

        # store modified z_s
        # This is why we kept putting in-place updates for the z_s graphs.
        # The updated graph information is obtained from the final processing over the z_s tensor.
        self.z_t_adj = get_multi_class_adj_matrix(z_s)

    def try_fix_and_validate_fn(self, nx_graph) -> bool:
        # We add two condition. The first one is to check class validity.
        # The second to check number of cycles.
        return self.top_cow_validity_check_and_fix_obj.check_validity_and_fix_if_needed(nx_graph)


    @property
    def can_block_edges(self):
        return True


class TopCoWLineGraphCheckAndFixModule(TopCoWLineGraphCheck):
    def __init__(self, device):
        super(TopCoWLineGraphCheckAndFixModule, self).__init__(device=device)


    def check_validity_and_fix_if_needed(self, nx_graph: nx.Graph) -> bool:
        if nx_graph.graph.get('all_possible_new_edges_to_add', None) is None and len(nx_graph.edges) == 0:
            # An empty graph is always valid, and we have nothing new to add
            return True
        # We go through the new edges one by one and check whether adding them has any effect on the validity of the graph.
        is_valid_arr = []
        for (u, v), label, cx in nx_graph.graph['all_possible_new_edges_to_add']:
            # We add the edge to the graph
            # print(f"Adding edge {u} -> {v} with label {label}")
            nx_graph.add_edge(u, v, label=label)
            is_valid = True
            if not self.is_line_graph_valid(nx_graph):
                print(f"{u} -> {v} with label {label} is invalid")
                # The original graph is not valid.
                # We go though the candidates now to see if it can be fixed.
                is_valid = False
                self.counter -=1 # Since we repeat with
                for cand in cx:
                    self.counter += 1
                    if cand == 0:
                        print("Skipping proposal of type 0 since it implies edge deletion")
                        continue
                    print(f"trying now with {cand}")
                    # We try to add the edge with the candidate label
                    nx_graph.remove_edge(u, v)
                    nx_graph.add_edge(u, v, label=cand)
                    if self.is_line_graph_valid(nx_graph):
                        is_valid = True
                        print(f"Changing {u} -> {v} to {cand} made it valid")
                        break
                    else:
                        print("Still invalid")
            is_valid_arr.append(is_valid)
        return all(is_valid_arr)


    def is_line_graph_valid(self, nx_graph: nx.Graph) -> bool:
        line_graph = nx.line_graph(nx_graph)
        candidate_matrix = torch.zeros_like(self.valid_adjacency)
        # Transfer the 'label' from each edge in G to the corresponding node in L
        for u, v, data in nx_graph.edges(data=True):
            line_graph.nodes[u, v]['label'] = data.get('label')
            # line_graph.nodes[u, v]['cands'] = data.get('cands')
        # We start the processing
        valid = True
        for edge in line_graph.edges():
            # Each edge in the line graph connects two nodes, which correspond to edges in the original graph
            source_node, target_node = edge

            # Access the attributes of the source and target nodes in the line graph
            edge_class_a = line_graph.nodes[source_node]['label']
            edge_class_b = line_graph.nodes[target_node]['label']

            # Array indexing starts from 0.
            # Hence, we subtract 1 from the edge class to get the correct index in the adjacency matrix
            candidate_matrix[edge_class_a - 1, edge_class_b - 1] = 1
            candidate_matrix[edge_class_b - 1, edge_class_a - 1] = 1
        # Ensuring that the matrix is symmetric
        self.check_symmetric(candidate_matrix)
        # Delete the diagonal
        # This is for self-edges. It is needed since sometimes the vessel curvature introduces extra loops.
        # Hence, it is not wrong if a self connection appears
        candidate_matrix = (candidate_matrix - torch.eye(candidate_matrix.shape[0], device=candidate_matrix.device)).clip(min=0, max=1)
        possible_violations = candidate_matrix - self.valid_adjacency
        if torch.any(possible_violations > 0):
            valid = False
        return valid


class CoWVanillaProjector():

    def __init__(self, z_t: PlaceHolder, device: torch.device):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        self.top_cow_validity_check = TopCoWVanillaLineGraphCheck(device)
        if self.can_block_edges:
            # Which edges of the graph should be blocked.
            # These edges would not be considered a candidate for deletion.
            # The process improves sampling speed since we need not check these invalid edges in subsequent steps.
            self.blocked_edges = {graph_idx: {} for graph_idx in range(self.batch_size)}

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_multi_class_adj_matrix(z_t)

        assert (self.z_t_adj == 0).all()  # no edges in the planar limit dist
        # add data structure where validity is checked
        for graph_idx in range(self.batch_size):
            # We iterate over each of the graphs and create an empty graph with node coordinate information.
            num_nodes = z_t.node_mask[graph_idx].sum()
            pos = z_t.pos[graph_idx, :num_nodes]
            # We create a complete graph with the given number of nodes.
            nx_graph = nx.empty_graph(num_nodes)
            # Assign 3D coordinates to nodes
            node_positions =  {i: pos[i] for i in range(num_nodes)}
            nx.set_node_attributes(nx_graph, node_positions, name="pos")
            # An fully-connected graph is always a valid graph
            assert self.valid_graph_fn(nx_graph)
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list
            if self.can_block_edges:
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][(node_1_idx, node_2_idx)] = False  # None of the edges are blocked.

    def project(self, z_s: PlaceHolder):
        # find added edges
        z_s_adj = get_multi_class_adj_matrix(z_s)  # B, N , N
        diff_adj = z_s_adj - self.z_t_adj
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        new_edges = diff_adj.nonzero(as_tuple=False)
        # add new edges and check validity
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            old_nx_graph = nx_graph.copy()
            edge_indices = (
                new_edges[
                    torch.logical_and(
                        new_edges[:, 0] == graph_idx,  # Select edges of the graph
                        new_edges[:, 1] < new_edges[:, 2],  # undirected graph
                    )
                ]
                .cpu()
                .numpy()
            )
            edges_to_add = edge_indices[:, 1:]
            if len(edges_to_add) > 0:
                edge_attrs = z_s_adj.cpu().numpy()[edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2]]

            # If we can block edges, we do it
            if self.can_block_edges and len(edges_to_add) > 0:
                not_blocked_edges, not_blocked_attrs = [], []
                for edge, edge_attr in zip(edges_to_add, edge_attrs):
                    # Check if the edge has already been blocked.
                    if self.blocked_edges[graph_idx][tuple(edge)]:
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                    else:
                        # Only non blocked edges are the candidates.
                        not_blocked_edges.append(edge)
                        not_blocked_attrs.append(edge_attr)
                edges_to_add = np.array(not_blocked_edges)
                edge_attrs = np.array(not_blocked_attrs)
            # If no edges to delete, we skip the graph
            if len(edges_to_add) == 0:
                continue

            # First try add all edges (we might be lucky)
            if len(edges_to_add) > 1:  # avoid repetition of steps
                # Add all the edges to the graph
                for (u, v), label in zip(edges_to_add, edge_attrs):
                    nx_graph.add_edge(u, v, label=label)

            # If it fails, we go one by one and delete the violators
            if not self.valid_graph_fn(nx_graph) or len(edges_to_add) == 1:
                nx_graph = old_nx_graph
                # Try to add edges one by one (in random order)
                np.random.shuffle(edges_to_add)
                for edge, label in zip(edges_to_add, edge_attrs):
                    old_nx_graph = nx_graph.copy()
                    nx_graph.add_edge(edge[0], edge[1], label=label)
                    if not self.valid_graph_fn(nx_graph):
                        nx_graph = old_nx_graph
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # this edge breaks validity.
                        # Blocking it for future steps.
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][tuple(edge)] = True
                self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

            # Check that nx graphs is correctly stored
            assert (
                    nx.to_numpy_array(nx_graph)
                    == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all()

            # Check that adjacency matrices are the same in torch and nx
            num_nodes = self.nx_graphs_list[graph_idx].number_of_nodes()
            assert (
                    get_adj_matrix(z_s)[graph_idx].cpu().numpy()[:num_nodes, :num_nodes]
                    == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all()

        # store modified z_s
        # This is why we kept putting in-place updates for the z_s graphs.
        # The updated graph information is obtained from the final processing over the z_s tensor.
        self.z_t_adj = get_multi_class_adj_matrix(z_s)


    def valid_graph_fn(self, nx_graph) -> bool:
        return self.top_cow_validity_check.is_valid_graph(nx_graph)

    @property
    def can_block_edges(self):
        return True


class TopCoWVanillaLineGraphCheck(TopCoWLineGraphCheck):
    def __init__(self, device):
        super(TopCoWVanillaLineGraphCheck, self).__init__(device=device)

    def is_valid_graph(self, nx_graph: nx.Graph) -> bool:
        line_graph = nx.line_graph(nx_graph)
        candidate_matrix = torch.zeros_like(self.valid_adjacency)

        # Transfer the 'label' from each edge in G to the corresponding node in L
        for u, v, data in nx_graph.edges(data=True):
            line_graph.nodes[u, v]['label'] = data.get('label')

        valid = True
        for edge in line_graph.edges():
            # Each edge in the line graph connects two nodes, which correspond to edges in the original graph
            source_node, target_node = edge

            # Access the attributes of the source and target nodes in the line graph
            edge_class_a = line_graph.nodes[source_node]['label']
            edge_class_b = line_graph.nodes[target_node]['label']

            # Array indexing starts from 0.
            # Hence, we subtract 1 from the edge class to get the correct index in the adjacency matrix
            candidate_matrix[edge_class_a - 1, edge_class_b - 1] = 1
            candidate_matrix[edge_class_b - 1, edge_class_a - 1] = 1

        self.check_symmetric(candidate_matrix)
        # Delete the diagonal
        # This is for self-edges. It is needed since sometimes the vessel curvature introduces extra loops while extracting from voreen.
        # Hence, it is not wrong if a self connection appears
        candidate_matrix = (candidate_matrix - torch.eye(candidate_matrix.shape[0], device=candidate_matrix.device)).clip(min=0, max=1)
        possible_violations = candidate_matrix - self.valid_adjacency
        if torch.any(possible_violations > 0):
            valid = False
            # for e, l in nx_graph.edges.items():
            #     print(f"{e} -----------> {l}")
        return valid

# Now the formulation for ATM

class ATMLineGraphCheck(object):

    def __init__(self, device):
        #                                             1  2  3  4
        self.valid_adjacency = torch.as_tensor([[0, 1, 0, 0],  # 1
                                                     [1, 0, 1, 0],  # 2
                                                     [0, 1, 0, 1],  # 3
                                                     [0, 0, 1, 0]], device=device)  # 4
        self.check_symmetric(self.valid_adjacency)


    @staticmethod
    def check_symmetric(array: torch.Tensor) -> None:
        """
        Verifies if the given array is symmetric. Raises an assertion error if not symmetric
        :param array: Adjacency matrix B, N, N
        :return: None
        """
        assert torch.allclose(array, array.T), "Not symmetric"


class ATMTreeProjector():

    def __init__(self, z_t: PlaceHolder, device: torch.device):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        self.atm_validity_check_and_fix_obj = ATMLineGraphCheckAndFixModule(device)
        if self.can_block_edges:
            # Which edges of the graph should be blocked.
            # These edges would not be considered a candidate for deletion.
            # The process improves sampling speed since we need not check these invalid edges in subsequent steps.
            self.blocked_edges = {graph_idx: {} for graph_idx in range(self.batch_size)}

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_multi_class_adj_matrix(z_t)

        assert (self.z_t_adj == 0).all()  # no edges in the planar limit dist
        # add data structure where validity is checked
        for graph_idx in range(self.batch_size):
            # We iterate over each of the graphs and create an empty graph with node coordinate information.
            num_nodes = z_t.node_mask[graph_idx].sum()
            pos = z_t.pos[graph_idx, :num_nodes]
            # We create a complete graph with the given number of nodes.
            nx_graph = nx.empty_graph(num_nodes)
            # Assign 3D coordinates to nodes
            node_positions =  {i: pos[i] for i in range(num_nodes)}
            nx.set_node_attributes(nx_graph, node_positions, name="pos")
            # An fully-connected graph is always a valid graph
            assert self.try_fix_and_validate_fn(nx_graph)
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list
            if self.can_block_edges:
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][(node_1_idx, node_2_idx)] = False  # None of the edges are blocked.

    def project(self, z_s: PlaceHolder):
        # find added edges
        z_s_adj = get_multi_class_adj_matrix(z_s)  # B, N , N
        num_classes = z_s.E.shape[-1]
        diff_adj = z_s_adj - self.z_t_adj
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        new_edges = diff_adj.nonzero(as_tuple=False)
        # add new edges and check validity
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            old_nx_graph = nx_graph.copy()
            edge_indices = (
                new_edges[
                    torch.logical_and(
                        new_edges[:, 0] == graph_idx,  # Select edges of the graph
                        new_edges[:, 1] < new_edges[:, 2],  # undirected graph
                    )
                ]
                .cpu()
                .numpy()
            )
            edges_to_add = edge_indices[:, 1:]
            if len(edges_to_add) > 0:
                edge_attrs = z_s_adj.cpu().numpy()[edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2]]
                candidate_attrs = z_s.extra_info.cpu().numpy()[edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2]]  # B, N, N, top_k

            # If we can block edges, we do it
            if self.can_block_edges and len(edges_to_add) > 0:
                not_blocked_edges, not_blocked_attrs, not_blocked_cands = [], [], []
                for edge, edge_attr, cands in zip(edges_to_add, edge_attrs, candidate_attrs):
                    # Check if the edge has already been blocked.
                    if self.blocked_edges[graph_idx][tuple(edge)]:
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                    else:
                        # Only non blocked edges are the candidates.
                        not_blocked_edges.append(edge)
                        not_blocked_attrs.append(edge_attr)
                        not_blocked_cands.append(cands)
                edges_to_add = np.array(not_blocked_edges)
                edge_attrs = np.array(not_blocked_attrs)
                candidate_attrs = np.array(not_blocked_cands)
            # If no edges to delete, we skip the graph
            if len(edges_to_add) == 0:
                continue

            # First try add all edges (we might be lucky)
            if len(edges_to_add) > 1:  # avoid repetition of steps
                # Add all the edges to the graph
                all_possible_new_edges_to_add = []
                for (u, v), label, cx in zip(edges_to_add, edge_attrs, candidate_attrs):
                    # nx_graph.add_edge(u, v, label=label, cands=cx)
                    # A tuple of three elements is added to the list.
                    all_possible_new_edges_to_add.append(
                        ((u, v), label, cx)
                    )
                nx_graph.graph['all_possible_new_edges_to_add'] = all_possible_new_edges_to_add
                # We check if all the edges are valid and fixable together.
                if self.try_fix_and_validate_fn(nx_graph):
                    # The graph labels might have changed though.
                    # So, let us update the labels in the z_s tensor.
                    for (u, v) in edges_to_add:
                        new_label = nx_graph[u][v]['label']
                        z_s.E[graph_idx, u, v, :] = F.one_hot(torch.tensor(new_label), num_classes=num_classes)
                        z_s.E[graph_idx, v, u, :] = F.one_hot(torch.tensor(new_label), num_classes=num_classes)

            # If num_edges > 1, we do not have the `all_possible_new_edges_to_add` attribute.
            # However, we rely on the shot-circuit nature of the `or` operator.
            if len(edges_to_add) == 1 or not self.try_fix_and_validate_fn(nx_graph):
                nx_graph = old_nx_graph
                # Try to add edges one by one (in random order)
                np.random.shuffle(edges_to_add)
                for edge, label, cands in zip(edges_to_add, edge_attrs, candidate_attrs):
                    old_nx_graph = nx_graph.copy()
                    nx_graph.graph['all_possible_new_edges_to_add'] = [
                        ((edge[0], edge[1]), label, cands)
                    ]  # A list of tuples
                    if not self.try_fix_and_validate_fn(nx_graph):
                        nx_graph = old_nx_graph
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # this edge breaks validity.
                        # Blocking it for future steps.
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][tuple(edge)] = True
                    else:
                        # The graph is valid. However, the label may have been updated
                        # So, we update the label in the z_s tensor.
                        updated_label = nx_graph[edge[0]][edge[1]]['label']
                        z_s.E[graph_idx, edge[0], edge[1], :] = F.one_hot(torch.tensor(updated_label),
                                                                          num_classes=num_classes)
                        z_s.E[graph_idx, edge[1], edge[0], :] = F.one_hot(torch.tensor(updated_label),
                                                                          num_classes=num_classes)
                # Need to update the graph in the list.
                # Needs to be done since the original graph is modified in place.
                # It turned out to be invalid, so we replace it.
                self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

            # Check that nx graphs is correctly stored
            assert (
                    nx.to_numpy_array(nx_graph)
                    == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all()

            # Check that adjacency matrices are the same in torch and nx
            num_nodes = self.nx_graphs_list[graph_idx].number_of_nodes()
            # assert (
            #         get_adj_matrix(z_s)[graph_idx].cpu().numpy()[:num_nodes, :num_nodes]
            #         == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            # ).all()
            if not (
                    get_adj_matrix(z_s)[graph_idx].cpu().numpy()[:num_nodes, :num_nodes]
                    == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all():
                print("Something went wrong!!")

        # store modified z_s
        # This is why we kept putting in-place updates for the z_s graphs.
        # The updated graph information is obtained from the final processing over the z_s tensor.
        self.z_t_adj = get_multi_class_adj_matrix(z_s)


    def try_fix_and_validate_fn(self, nx_graph) -> bool:
        # We add two condition. The first one is to check class validity. Second checks no-cycles.
        return self.atm_validity_check_and_fix_obj.check_validity_and_fix_if_needed(nx_graph) and has_no_cycles(nx_graph)

    def valid_graph_fn_lite(self, nx_graph) -> bool:
        # We have an intermediate noised graph version.
        # We do not have any new edges to add. However, we want to check if the intermediate is valid or not.
        return (nx_graph.graph.get('all_possible_new_edges_to_add', None) is None
                and self.atm_validity_check_and_fix_obj.is_line_graph_valid(nx_graph))

    @property
    def can_block_edges(self):
        return True


class ATMLineGraphCheckAndFixModule(ATMLineGraphCheck):
    def __init__(self, device):
        super(ATMLineGraphCheckAndFixModule, self).__init__(device=device)

    def check_validity_and_fix_if_needed(self, nx_graph: nx.Graph) -> bool:

        if nx_graph.graph.get('all_possible_new_edges_to_add', None) is None and len(nx_graph.edges) == 0:
            # An empty graph is always valid, and we have nothing new to add
            return True
            # We go through the new edges one by one and check whether adding them has any effect on the validity of the graph.
        is_valid_arr = []
        for (u, v), label, cx in nx_graph.graph['all_possible_new_edges_to_add']:
            # We add the edge to the graph
            # print(f"Adding edge {u} -> {v} with label {label}")
            nx_graph.add_edge(u, v, label=label)
            is_valid = True
            if not self.is_line_graph_valid(nx_graph):
                print(f"{u} -> {v} with label {label} is invalid")
                # The original graph is not valid.
                # We go though the candidates now to see if it can be fixed.
                is_valid = False
                for cand in cx:
                    print(f"trying now with {cand}")
                    if cand == 0:
                        print("Skipping proposal of type 0 since it implies edge deletion")
                        continue
                    # We try to add the edge with the candidate label
                    nx_graph.remove_edge(u, v)
                    nx_graph.add_edge(u, v, label=cand)
                    if self.is_line_graph_valid(nx_graph):
                        is_valid = True
                        print(f"Changing {u} -> {v} to {cand} made it valid")
                        break
                    else:
                        print("Still invalid")
            is_valid_arr.append(is_valid)
        return all(is_valid_arr)

    def is_line_graph_valid(self, nx_graph: nx.Graph) -> bool:
        line_graph = nx.line_graph(nx_graph)
        candidate_matrix = torch.zeros_like(self.valid_adjacency)

        # Transfer the 'label' from each edge in G to the corresponding node in L
        for u, v, data in nx_graph.edges(data=True):
            line_graph.nodes[u, v]['label'] = data.get('label')

        valid = True
        for edge in line_graph.edges():
            # Each edge in the line graph connects two nodes, which correspond to edges in the original graph
            source_node, target_node = edge

            # Access the attributes of the source and target nodes in the line graph
            edge_class_a = line_graph.nodes[source_node]['label']
            edge_class_b = line_graph.nodes[target_node]['label']

            # Array indexing starts from 0.
            # Hence, we subtract 1 from the edge class to get the correct index in the adjacency matrix
            candidate_matrix[edge_class_a - 1, edge_class_b - 1] = 1
            candidate_matrix[edge_class_b - 1, edge_class_a - 1] = 1

        # Ensuring that the matrix is symmetric
        self.check_symmetric(candidate_matrix)
        # Delete the diagonal
        # This is for self-edges. It is needed since sometimes the vessel curvature introduces extra loops.
        # Hence, it is not wrong if a self connection appears
        candidate_matrix = (candidate_matrix - torch.eye(candidate_matrix.shape[0], device=candidate_matrix.device)).clip(min=0, max=1)
        possible_violations = candidate_matrix - self.valid_adjacency
        if torch.any(possible_violations > 0):
            valid = False
        return valid



# Formulation that does not try to fix the ATM labels.
class ATMVanillaTreeProjector():

    def __init__(self, z_t: PlaceHolder, device: torch.device):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        self.atm_validity_check = ATMVanillaLineGraphCheck(device)
        if self.can_block_edges:
            # Which edges of the graph should be blocked.
            # These edges would not be considered a candidate for deletion.
            # The process improves sampling speed since we need not check these invalid edges in subsequent steps.
            self.blocked_edges = {graph_idx: {} for graph_idx in range(self.batch_size)}

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_multi_class_adj_matrix(z_t)

        assert (self.z_t_adj == 0).all()  # no edges in the planar limit dist
        # add data structure where validity is checked
        for graph_idx in range(self.batch_size):
            # We iterate over each of the graphs and create an empty graph with node coordinate information.
            num_nodes = z_t.node_mask[graph_idx].sum()
            pos = z_t.pos[graph_idx, :num_nodes]
            # We create a complete graph with the given number of nodes.
            nx_graph = nx.empty_graph(num_nodes)
            # Assign 3D coordinates to nodes
            node_positions =  {i: pos[i] for i in range(num_nodes)}
            nx.set_node_attributes(nx_graph, node_positions, name="pos")
            # An fully-connected graph is always a valid graph
            assert self.valid_graph_fn(nx_graph)
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list
            if self.can_block_edges:
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][(node_1_idx, node_2_idx)] = False  # None of the edges are blocked.

    def project(self, z_s: PlaceHolder):
        # find added edges
        z_s_adj = get_multi_class_adj_matrix(z_s)  # B, N , N
        diff_adj = z_s_adj - self.z_t_adj
        assert (diff_adj >= 0).all()  # No edges can be removed in the reverse
        new_edges = diff_adj.nonzero(as_tuple=False)
        # add new edges and check validity
        for graph_idx, nx_graph in enumerate(self.nx_graphs_list):
            old_nx_graph = nx_graph.copy()
            edge_indices = (
                new_edges[
                    torch.logical_and(
                        new_edges[:, 0] == graph_idx,  # Select edges of the graph
                        new_edges[:, 1] < new_edges[:, 2],  # undirected graph
                    )
                ]
                .cpu()
                .numpy()
            )
            edges_to_add = edge_indices[:, 1:]
            if len(edges_to_add) > 0:
                edge_attrs = z_s_adj.cpu().numpy()[edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2]]

            # If we can block edges, we do it
            if self.can_block_edges and len(edges_to_add) > 0:
                not_blocked_edges, not_blocked_attrs = [], []
                for edge, edge_attr in zip(edges_to_add, edge_attrs):
                    # Check if the edge has already been blocked.
                    if self.blocked_edges[graph_idx][tuple(edge)]:
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                    else:
                        # Only non blocked edges are the candidates.
                        not_blocked_edges.append(edge)
                        not_blocked_attrs.append(edge_attr)
                edges_to_add = np.array(not_blocked_edges)
                edge_attrs = np.array(not_blocked_attrs)
            # If no edges to delete, we skip the graph
            if len(edges_to_add) == 0:
                continue

            # First try add all edges (we might be lucky)
            if len(edges_to_add) > 1:  # avoid repetition of steps
                # Add all the edges to the graph
                for (u, v), label in zip(edges_to_add, edge_attrs):
                    nx_graph.add_edge(u, v, label=label)

            # If it fails, we go one by one and delete the violators
            if not self.valid_graph_fn(nx_graph) or len(edges_to_add) == 1:
                nx_graph = old_nx_graph
                # Try to add edges one by one (in random order)
                np.random.shuffle(edges_to_add)
                for edge, label in zip(edges_to_add, edge_attrs):
                    old_nx_graph = nx_graph.copy()
                    nx_graph.add_edge(edge[0], edge[1], label=label)
                    if not self.valid_graph_fn(nx_graph):
                        nx_graph = old_nx_graph
                        # deleting edge from edges tensor (changes z_s in place)
                        z_s.E[graph_idx, edge[0], edge[1]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        z_s.E[graph_idx, edge[1], edge[0]] = F.one_hot(
                            torch.tensor(0), num_classes=z_s.E.shape[-1]
                        )
                        # this edge breaks validity.
                        # Blocking it for future steps.
                        if self.can_block_edges:
                            self.blocked_edges[graph_idx][tuple(edge)] = True
                self.nx_graphs_list[graph_idx] = nx_graph  # save new graph

            # Check that nx graphs is correctly stored
            assert (
                    nx.to_numpy_array(nx_graph)
                    == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all()

            # Check that adjacency matrices are the same in torch and nx
            num_nodes = self.nx_graphs_list[graph_idx].number_of_nodes()
            assert (
                    get_adj_matrix(z_s)[graph_idx].cpu().numpy()[:num_nodes, :num_nodes]
                    == nx.to_numpy_array(self.nx_graphs_list[graph_idx])
            ).all()

        # store modified z_s
        # This is why we kept putting in-place updates for the z_s graphs.
        # The updated graph information is obtained from the final processing over the z_s tensor.
        self.z_t_adj = get_multi_class_adj_matrix(z_s)


    def valid_graph_fn(self, nx_graph) -> bool:
        # We add two condition. The first one is to check class validity.
        return self.atm_validity_check.is_valid_graph(nx_graph) and has_no_cycles(nx_graph)

    @property
    def can_block_edges(self):
        return True



class ATMVanillaLineGraphCheck(ATMLineGraphCheck):
    def __init__(self, device):
        super(ATMVanillaLineGraphCheck, self).__init__(device=device)

    def is_valid_graph(self, nx_graph: nx.Graph) -> bool:
        line_graph = nx.line_graph(nx_graph)
        candidate_matrix = torch.zeros_like(self.valid_adjacency)

        # Transfer the 'label' from each edge in G to the corresponding node in L
        for u, v, data in nx_graph.edges(data=True):
            line_graph.nodes[u, v]['label'] = data.get('label')

        valid = True
        for edge in line_graph.edges():
            # Each edge in the line graph connects two nodes, which correspond to edges in the original graph
            source_node, target_node = edge

            # Access the attributes of the source and target nodes in the line graph
            edge_class_a = line_graph.nodes[source_node]['label']
            edge_class_b = line_graph.nodes[target_node]['label']

            # Array indexing starts from 0.
            # Hence, we subtract 1 from the edge class to get the correct index in the adjacency matrix
            candidate_matrix[edge_class_a - 1, edge_class_b - 1] = 1
            candidate_matrix[edge_class_b - 1, edge_class_a - 1] = 1

        self.check_symmetric(candidate_matrix)
        # Delete the diagonal
        # This is for self-edges. It is needed since sometimes the vessel curvature introduces extra loops.
        # Hence, it is not wrong if a self connection appears
        candidate_matrix = (candidate_matrix - torch.eye(candidate_matrix.shape[0], device=candidate_matrix.device)).clip(min=0, max=1)
        possible_violations = candidate_matrix - self.valid_adjacency
        if torch.any(possible_violations > 0):
            valid = False
            # for e, l in nx_graph.edges.items():
            #     print(f"{e} -----------> {l}")
        return valid
