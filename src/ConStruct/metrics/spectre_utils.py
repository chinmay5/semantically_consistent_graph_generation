###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
try:
    import graph_tool.all as gt
    from graph_tool import Graph as gt_graph
    from graph_tool.topology import isomorphism as gt_isomorphism
except (ModuleNotFoundError, ImportError) as e:
    print("Graph tool could not be loaded")
    print(f"Error: {e}")
# except ImportError as e:
#     print(f"Error importing graph_tool: {e}")
##Navigate to the ./util/orca directory and compile orca.cpp
# g++ -O2 -std=c++11 -o orca orca.cpp
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures

import pygsp as pg
import secrets
from string import ascii_uppercase, digits
from datetime import datetime
from scipy.linalg import eigvalsh
from scipy.stats import chi2
from src.ConStruct.analysis.dist_helper import (
        compute_mmd,
        gaussian_emd,
        gaussian,
        emd,
        gaussian_tv,
        disc,
)
from torch_geometric.utils import to_networkx
import wandb
import signal
from hydra.utils import get_original_cwd

PRINT_TIME = False
__all__ = [
    "degree_stats",
    "clustering_stats",
    "orbit_stats_all",
    "spectral_stats",
    "eval_acc_lobster_graph",
]


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)

    return mmd_dist


###############################################################################


def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1 : n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def get_spectral_pmf(eigs, max_eig):
    spectral_pmf, _ = np.histogram(
        np.clip(eigs, 0, max_eig), bins=200, range=(-1e-5, max_eig), density=False
    )
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def eigval_stats(
    eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_pmf,
                eig_ref_list,
                [max_eig for i in range(len(eig_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_pmf,
                eig_pred_list,
                [max_eig for i in range(len(eig_ref_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eig_ref_list)):
            spectral_temp = get_spectral_pmf(eig_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(eig_pred_list)):
            spectral_temp = get_spectral_pmf(eig_pred_list[i])
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing eig mmd: ", elapsed)
    return mmd_dist


def eigh_worker(G):
    L = nx.normalized_laplacian_matrix(G).todense()
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)


def compute_list_eigh(graph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list


def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop**2, axis=2)
    hist_range = [0, bound]
    hist = np.array(
        [np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt]
    )  # NOTE: change number of bins
    return hist.flatten()


def spectral_filter_stats(
    eigvec_ref_list,
    eigval_ref_list,
    eigvec_pred_list,
    eigval_pred_list,
    is_parallel=False,
    compute_emd=False,
):
    """Compute the distance between the eigvector sets.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    prev = datetime.now()

    class DMG(object):
        """Dummy Normalized Graph"""

        lmax = 2

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_ref_list,
                eigval_ref_list,
                [filters for i in range(len(eigval_ref_list))],
                [bound for i in range(len(eigval_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_pred_list,
                eigval_pred_list,
                [filters for i in range(len(eigval_pred_list))],
                [bound for i in range(len(eigval_pred_list))],
            ):
                # TODO: change line above to eigval_pred_list to see if it solves the problem
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_ref_list[i], eigval_ref_list[i], filters, bound
                )
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_pred_list[i], eigval_pred_list[i], filters, bound
                )
                sample_pred.append(spectral_temp)
            except:
                pass

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing spectral filter stats: ", elapsed)
    return mmd_dist


def spectral_stats(
    graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker,
                graph_pred_list_remove_empty,
                [n_eigvals for i in graph_pred_list_remove_empty],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True, compute_emd=False
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
        mmd_dist = compute_mmd(
            sample_ref,
            sample_pred,
            kernel=gaussian_emd,
            sigma=1.0 / 10,
            distance_scaling=bins,
        )
    else:
        mmd_dist = compute_mmd(
            sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10
        )

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
COUNT_START_STR = "orbit counts:"


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    # tmp_fname = f'analysis/orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = f'../analysis/orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    # print(tmp_fname, flush=True)
    f = open(tmp_fname, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()
    output = sp.check_output(
        [
            str(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "../analysis/orca/orca"
                )
            ),
            "node",
            "4",
            tmp_fname,
            "std",
        ]
    )
    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(
    graph_ref_list,
    graph_pred_list,
    motif_type="4cycle",
    ground_truth_match=None,
    bins=100,
    compute_emd=False,
):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    indices = motif_to_indices[motif_type]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    total_counts_ref = np.array(total_counts_ref)[:, None]
    total_counts_pred = np.array(total_counts_pred)[:, None]

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, is_hist=False)
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    return mmd_dist


def orbit_stats_all(graph_ref_list, graph_pred_list, compute_emd=False):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    # mmd_dist = compute_mmd(
    #     total_counts_ref,
    #     total_counts_pred,
    #     kernel=gaussian,
    #     is_hist=False,
    #     sigma=30.0)

    # mmd_dist = compute_mmd(
    #         total_counts_ref,
    #         total_counts_pred,
    #         kernel=gaussian_tv,
    #         is_hist=False,
    #         sigma=30.0)

    if compute_emd:
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, sigma=30.0)
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(
            total_counts_ref,
            total_counts_pred,
            kernel=gaussian,
            is_hist=False,
            sigma=30.0,
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref,
            total_counts_pred,
            kernel=gaussian_tv,
            is_hist=False,
            sigma=30.0,
        )
    return mmd_dist


def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]
    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_tree_graph(G_list):
    count = 0
    for gg in G_list:
        if nx.is_tree(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_grid_graph(G_list, grid_start=10, grid_end=20):
    count = 0
    for gg in G_list:
        if is_grid_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_sbm_graph(
    G_list,
    p_intra=0.3,
    p_inter=0.005,
    strict=True,
    refinement_steps=1000,
    is_parallel=True,
):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(
                is_sbm_graph,
                [gg for gg in G_list],
                [p_intra for i in range(len(G_list))],
                [p_inter for i in range(len(G_list))],
                [strict for i in range(len(G_list))],
                [refinement_steps for i in range(len(G_list))],
            ):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(
                gg,
                p_intra=p_intra,
                p_inter=p_inter,
                strict=strict,
                refinement_steps=refinement_steps,
            )
    return count / float(len(G_list))


def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))


def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def is_lobster_graph(G):
    """
    Check a given graph is a lobster graph or not

    Removing leaf nodes twice:

    lobster -> caterpillar -> path

    """
    ### Check if G is a tree
    if nx.is_tree(G):
        G = G.copy()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


def is_grid_graph(G):
    """
    Check if the graph is grid, by comparing with all the real grids with the same node count
    """
    all_grid_file = os.path.join(
        get_original_cwd(), "..", "data", "grid", "all_grids.pt"
    )
    if os.path.isfile(all_grid_file):
        all_grids = torch.load(all_grid_file)
    else:
        all_grids = {}
        for i in range(2, 20):
            for j in range(2, 20):
                G_grid = nx.grid_2d_graph(i, j)
                n_nodes = f"{len(G_grid.nodes())}"
                all_grids[n_nodes] = all_grids.get(n_nodes, []) + [G_grid]
        torch.save(all_grids, all_grid_file)

    n_nodes = f"{len(G.nodes())}"
    if n_nodes in all_grids:
        for G_grid in all_grids[n_nodes]:
            if nx.faster_could_be_isomorphic(G, G_grid):
                if test_isomorphism(G, G_grid):
                    return True
        return False
    else:
        return False


def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        if (
            (node_counts > 40).sum() > 0
            or (node_counts < 20).sum() > 0
            or n_blocks > 5
            or n_blocks < 2
        ):
            return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9  # p value < 10 %
    else:
        return p


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if test_isomorphism(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def eval_fraction_unique(fake_graphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if test_isomorphism(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs

    return frac_unique


def eval_fraction_unique_non_isomorphic_valid(
    fake_graphs, train_graphs, validity_func=(lambda x: True)
):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if test_isomorphism(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_idx, train_g in enumerate(train_graphs):
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if test_isomorphism(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (
        float(len(fake_graphs)) - count_non_unique - count_isomorphic
    ) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def test_isomorphism(G1, G2):
    # Added function to deal with cases where NetworkX is unable to conclude isomorphism within a reasonable time (it stalls...)
    try:
        # Set a timeout for the isomorphism check with NetworkX (10 seconds in this example)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # Set a 5-second timeout

        # isomorphic = nx.is_isomorphic(G1, G2)
        isomorphic = nx.could_be_isomorphic(G1, G2)

        signal.alarm(0)  # Cancel the alarm if the function completes within the timeout
        return isomorphic
    except (nx.NetworkXError, nx.exception.ExceededMaxIterations):
        signal.alarm(0)  # Cancel the alarm if an exception occurs
        pass  # Continue to graph-tool if NetworkX doesn't conclude within the timeout
    except TimeoutError as e:
        pass  # Function call timed out, continue to graph-tool

    print("NetworkX stalled in isomorphism check. Using graph-tool instead")
    # Convert NetworkX graphs to graph-tool graphs
    gt_g1 = gt_graph(directed=False)
    gt_g2 = gt_graph(directed=False)
    gt_g1.add_edge_list(G1.edges())
    gt_g2.add_edge_list(G2.edges())

    # Perform isomorphism check with graph-tool
    return gt_isomorphism(gt_g1, gt_g2)


class SpectreSamplingMetrics(nn.Module):
    def __init__(self, train_dataloader, val_dataloader, compute_emd, metrics_list):
        super().__init__()

        self.train_graphs = self.loader_to_nx(train_dataloader)
        self.val_graphs = self.loader_to_nx(val_dataloader)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list

        # Store for wavelet computaiton
        self.ref_eigvals, self.ref_eigvecs = compute_list_eigh(self.val_graphs)

    def loader_to_nx(self, loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            # TODO: this does not run with current loader
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                networkx_graphs.append(
                    to_networkx(
                        data,
                        node_attrs=None,
                        edge_attrs=None,
                        to_undirected=True,
                        remove_self_loops=True,
                    )
                )
        return networkx_graphs

    def forward(
        self,
        generated_graphs: list,
        current_epoch,
        local_rank,
    ):
        if local_rank == 0:
            print(
                f"Computing SPECTRE sampling metrics between {sum([placeholder.X.shape[0] for placeholder in generated_graphs])} generated graphs and {len(self.val_graphs)}"
                f" val graphs -- emd computation: {self.compute_emd}"
            )
        networkx_graphs = []
        adjacency_matrices = []
        if local_rank == 0:
            print("Building networkx graphs...")
        for batch in generated_graphs:
            graphs = batch.split()
            for graph in graphs:
                A = graph.E.cpu().numpy()
                adjacency_matrices.append(A)
                nx_graph = nx.from_numpy_array(A)
                networkx_graphs.append(nx_graph)

        np.savez("generated_adjs.npz", *adjacency_matrices)

        to_log = {}

        if "degree" in self.metrics_list:
            if local_rank == 0:
                print("Computing degree stats..")
            degree = degree_stats(
                self.val_graphs,
                networkx_graphs,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            to_log["degree"] = degree
            if wandb.run:
                wandb.run.summary["degree"] = degree

        # val_eigvals = [graph["eigval"][1:self.k + 1].cpu().detach().numpy() for graph in self.val]
        # train_eigvals = [graph["eigval"][1:self.k + 1].cpu().detach().numpy() for graph in self.train]

        # eigval_stats(eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False)
        # spectral_filter_stats(eigvec_ref_list, eigval_ref_list, eigvec_pred_list, eigval_pred_list, is_parallel=False,
        #                       compute_emd=False)          # This is the one called wavelet

        if "wavelet" in self.metrics_list:
            if local_rank == 0:
                print("Computing wavelet stats...")

            pred_graph_eigvals, pred_graph_eigvecs = compute_list_eigh(networkx_graphs)
            wavelet = spectral_filter_stats(
                eigvec_ref_list=self.ref_eigvecs,
                eigval_ref_list=self.ref_eigvals,
                eigvec_pred_list=pred_graph_eigvecs,
                eigval_pred_list=pred_graph_eigvals,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )
            to_log["wavelet"] = wavelet
            if wandb.run:
                wandb.run.summary["wavelet"] = wavelet

        if "spectre" in self.metrics_list:
            if local_rank == 0:
                print("Computing spectre stats...")
            spectre = spectral_stats(
                self.val_graphs,
                networkx_graphs,
                is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )

            to_log["spectre"] = spectre
            if wandb.run:
                wandb.run.summary["spectre"] = spectre

        if "clustering" in self.metrics_list:
            if local_rank == 0:
                print("Computing clustering stats...")
            clustering = clustering_stats(
                self.val_graphs,
                networkx_graphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            to_log["clustering"] = clustering
            if wandb.run:
                wandb.run.summary["clustering"] = clustering

        if "motif" in self.metrics_list:
            if local_rank == 0:
                print("Computing motif stats")
            motif = motif_stats(
                self.val_graphs,
                networkx_graphs,
                motif_type="4cycle",
                ground_truth_match=None,
                bins=100,
                compute_emd=self.compute_emd,
            )
            to_log["motif"] = motif
            if wandb.run:
                wandb.run.summary["motif"] = motif

        if "orbit" in self.metrics_list:
            if local_rank == 0:
                print("Computing orbit stats...")
            orbit = orbit_stats_all(
                self.val_graphs, networkx_graphs, compute_emd=self.compute_emd
            )
            to_log["orbit"] = orbit
            if wandb.run:
                wandb.run.summary["orbit"] = orbit

        if "sbm" in self.metrics_list:
            if local_rank == 0:
                print("Computing accuracy...")
            acc = eval_acc_sbm_graph(networkx_graphs, refinement_steps=100, strict=True)
            to_log["sbm_acc"] = acc
            if wandb.run:
                wandb.run.summary["sbmacc"] = acc

        if "planar" in self.metrics_list:
            if local_rank == 0:
                print("Computing planar accuracy...")
            planar_acc = eval_acc_planar_graph(networkx_graphs)
            to_log["planar_acc"] = planar_acc
            if wandb.run:
                wandb.run.summary["planar_acc"] = planar_acc

        if "grid" in self.metrics_list:
            if local_rank == 0:
                print("Computing grid accuracy...")
            grid_acc = eval_acc_grid_graph(networkx_graphs)
            to_log["grid_acc"] = grid_acc
            if wandb.run:
                wandb.run.summary["grid_acc"] = grid_acc

        if "tree" in self.metrics_list:
            if local_rank == 0:
                print("Computing tree accuracy...")
            tree_acc = eval_acc_tree_graph(networkx_graphs)
            to_log["tree_acc"] = tree_acc
            if wandb.run:
                wandb.run.summary["tree_acc"] = tree_acc

        if "lobster" in self.metrics_list:
            if local_rank == 0:
                print("Computing lobster accuracy...")
            lobster_acc = eval_acc_lobster_graph(networkx_graphs)
            to_log["lobster_acc"] = lobster_acc
            if wandb.run:
                wandb.run.summary["lobster_acc"] = lobster_acc

        if "sbm" or "planar" or "tree" or "lobster" or "grid" in self.metrics_list:
            if local_rank == 0:
                print("Computing all fractions...")
            if "sbm" in self.metrics_list:
                valid_fn = is_sbm_graph
            elif "tree" in self.metrics_list:
                valid_fn = nx.is_tree
            elif "lobster" in self.metrics_list:
                valid_fn = is_lobster_graph
            elif "grid" in self.metrics_list:
                valid_fn = is_grid_graph
            else:
                valid_fn = is_planar_graph
            (
                frac_unique,
                frac_unique_non_isomorphic,
                fraction_unique_non_isomorphic_valid,
            ) = eval_fraction_unique_non_isomorphic_valid(
                networkx_graphs,
                self.train_graphs,
                validity_func=valid_fn,
            )
            frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(
                networkx_graphs, self.train_graphs
            )
            to_log.update(
                {
                    "sampling/frac_unique": frac_unique,
                    "sampling/frac_unique_non_iso": frac_unique_non_isomorphic,
                    "sampling/frac_unic_non_iso_valid": fraction_unique_non_isomorphic_valid,
                    "sampling/frac_non_iso": frac_non_isomorphic,
                }
            )

        if local_rank == 0:
            print("Sampling statistics", to_log)
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log

    def reset(self):
        pass


class Comm20SamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=True,
            metrics_list=["degree", "clustering", "orbit"],
        )


class PlanarSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "planar",
                "wavelet",
            ],
        )


class GridSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "grid",
                "wavelet",
                "planar",  # Add planar to check if grid is planar
            ],
        )


class TreeSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "tree",
                "wavelet",
                "planar",  # Add planar to check if tree is planar
            ],
        )


class LobsterSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "lobster",
                "wavelet",
                "planar",  # Add planar to check if lobster is planar
            ],
        )


class SBMSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet", "sbm"],
        )


class EgoSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre"],
        )


class ProteinSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre"],
        )
