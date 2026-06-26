import wot
import json
import gc
import concurrent
from functools import partial
from tqdm import tqdm
import multiprocessing
import torch 
from torch.distributions import Normal
from collections import defaultdict
from gmvae.networks import GMVAENet
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from dataclasses import dataclass
import pandas as pd
import ot
import anndata as ad
from anndata import AnnData
import warnings
import matplotlib 
import matplotlib.pyplot as pl
from matplotlib import gridspec
import tables
import logging
from numpy.typing import NDArray
import scipy.sparse as sp_sparse
from pathlib import Path
from ot import sinkhorn_unbalanced
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior
import multiprocessing as mp
from functools import reduce

mp.set_start_method('spawn', force=True)

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PATH_TO_RAW_EXP_MATX = "/Users/sarahboufelja/waddington_ot/GSE122662_RAW"
PATH_TO_SERUM_CELLS = "/Users/sarahboufelja/waddington_ot/data/serum_cell_ids.txt"
PATH_TO_CELL_DAYS = "/Users/sarahboufelja/waddington_ot/data/cell_days.txt"
CACHE = None

@dataclass
class Population:
    prob_vec: torch.Tensor | None
    support: list
    day: float
    name: str


def filter_path_to_exp_matx(path2files: Path | str) -> list:
    """Filters the path to expression matrices containing Serum and Dox C1/C2 data.

    Args:
        path2files (Path | str): Path to all expression matrices.

    Returns:
        list: Path to all expression matrices in the serum exp.
    """
    path_to_exp_matx = []
    exp_identifiers = ["serum", "Dox_C1", "Dox_C2"]
    for file in Path(path2files).glob("*.h5"):
        if any([idx in str(file) for idx in exp_identifiers]):
            path_to_exp_matx.append(file)
    logger.info(f"Retrieved {len(path_to_exp_matx)} expression matrices.")
    return path_to_exp_matx

def get_matrix_from_h5(filenames: str | list) -> pd.DataFrame:
    """Reads the list of h5 files containing the expression matrices into one pandas DF.

    Args:
        filename (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    filenames = [filenames] if isinstance(filenames, str|Path) else filenames
    cell_dataframes = []
    for idx, file in enumerate(filenames):
        with tables.open_file(file, 'r') as f:
            mat_group = f.get_node(f.root, 'mm10')
            barcodes = f.get_node(mat_group, 'barcodes').read()
            data = getattr(mat_group, 'data').read()
            indices = getattr(mat_group, 'indices').read()
            indptr = getattr(mat_group, 'indptr').read()
            shape = getattr(mat_group, 'shape').read()
            gene_names = getattr(mat_group, 'gene_names').read()
            # genes = getattr(mat_group, 'genes').read()
            matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
            # countMat = CountMatrix(barcodes, matrix, genes, gene_names)
            filtered_feature_bc_df = pd.DataFrame(matrix.todense()).T
            filtered_feature_bc_df.index = [str(bar, encoding="utf-8") for bar in barcodes]
            filtered_feature_bc_df.columns = [str(gn, encoding="utf-8") for gn in gene_names]
            filtered_feature_bc_df["day"] = float(str(file).split("_D")[1].split("_", maxsplit=1)[0])
            cell_dataframes.append(filtered_feature_bc_df)

    cell_df = pd.concat(cell_dataframes, axis = 0)
    assert cell_df.shape[0] == np.sum(np.array([df.shape[0] for df in cell_dataframes])), print("Issue with the final gene expression matrix detected. Review before proceeding")
    return cell_df

def get_seq_timestamps(path_to_cell_days: Path | str) -> list:
    """_summary_

    Args:
        path_to_cell_days (Path | str): _description_

    Returns:
        set: _description_
    """
    days_dict = {}
    with open(path_to_cell_days) as fv:
        for idx, line in enumerate(fv):
            if idx == 0:
                continue
            (key, val) = line.split()
            days_dict[key] = float(val)
    timestamps = sorted(list(set(days_dict.values())))
    timestamps = [int(tmsp) if ".0" in str(tmsp) else float(tmsp) for tmsp in timestamps]
    return timestamps

def get_reduced_express_matrix(adata:pd.DataFrame, 
                            reducer: GMVAENet,
                            gumbel_temp: float = 0.5,
                            gumbel_hard: bool = True)->pd.DataFrame:
    """Produces the gene expression matrix on a given day, reduced using the trained GM-VAE model.

    Args:
        data (pd.DataFrame): the single-cell sequencing data produced on a given day.
        reducer (GMVAENet): the trained model.

    Returns:
        (pd.DataFrame): the gene expression matrix on a given day with Leiden clusters based on the KNN graph.
    """
    adata_reduced = adata.copy(deep=True)
    if "day" in adata_reduced:
        day = adata_reduced["day"]
        adata_reduced = adata_reduced.drop("day", axis=1)
    
    # Perform dim. reduction with GM-VAE
    index = adata_reduced.index
    data_reduced_tensor = torch.tensor(adata_reduced.values, dtype=torch.float32)
    adata_reduced = reducer(data_reduced_tensor, gumbel_temp, gumbel_hard)
    adata_reduced = pd.DataFrame(adata_reduced.latent_sample.detach().numpy())
    adata_reduced.index = index
    adata_reduced.columns = [f"col_{idx}" for idx in range(100)]
    adata_reduced["day"] = day
    logger.info(f"Expression matrix size after reduction: {adata_reduced.shape}")
    return adata_reduced

def compute_leiden_communities(dataset: pd.DataFrame):
    return dataset

def read_daily_dataset(day: float, reducer: GMVAENet | None,) -> pd.DataFrame:
    """Produces the expression matrix on a given day.
    If a reducer model is provided, the data is transformed using the pre-trained GM-VAE model. 

    Args:
        day (_type_): _description_
        reducer (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Read the path to expression matrices.
    path_2_exp_matx = filter_path_to_exp_matx(path2files=PATH_TO_RAW_EXP_MATX)
    # Filter the exps containing the day's sequencing info
    path_2_exp_matx_day = [pth for pth in path_2_exp_matx if f"_D{day}_" in str(pth)]

    # Filter cells belonging to the Serum path
    with open(PATH_TO_SERUM_CELLS, "r") as f:
        # The following line extracts cell names belonging to the Serum path sequenced on D day
        serum_cells = set([line.strip() for line in f if f"D{day}_" in line])

    expression_matrices = []

    for exp_matx in path_2_exp_matx_day:
        cell_prefix = Path(exp_matx).stem.rsplit("_", maxsplit=3)[0].split("_", maxsplit=1)[-1]
        processed_exp_mat = get_matrix_from_h5(exp_matx)

        # Update the index with the cell_prefix
        processed_exp_mat.index = processed_exp_mat.index.map(lambda x: cell_prefix + "_" + str(x))

        serum_cell_indices = processed_exp_mat.index.get_indexer_for(serum_cells)
        serum_cell_indices = serum_cell_indices[serum_cell_indices > -1]
        processed_exp_mat = processed_exp_mat.iloc[serum_cell_indices]

        # If a reducer model is provided, perform the dim. reduction
        if reducer:
            processed_exp_mat = get_reduced_express_matrix(processed_exp_mat, reducer, )
        
        expression_matrices.append(processed_exp_mat)

    expression_matrices = pd.concat(expression_matrices, axis=0)
    expression_matrices = expression_matrices[:300]
    logger.info(f"Shape of the final expression matrix: {expression_matrices.shape}")
    return expression_matrices

def population_from_ids(expression_matrix: pd.DataFrame,
                        mapping_cells_to_pops: list[list]) -> list:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        list_cells_per_pop (list[list]): _description_

    Returns:
        list: _description_
    """
    def get_population(population):
        # cells_names = [idx.split("_")[-1] for idx in population]
        # Assert the file naming convention is respected.
        # assert all([len(elem) == 18 for elem in population])
        cell_indices = expression_matrix.index.get_indexer_for(population)
        cell_indices = cell_indices[cell_indices > -1]
        if len(cell_indices) == 0:
            return None
        # The prob. distribution is supported on the cells in expression_matrix
        prob_vec = torch.zeros(len(expression_matrix))
        prob_vec[cell_indices] = 1.0
        prob_vec /= prob_vec.sum()
        return prob_vec

    result = [get_population(population) for population in mapping_cells_to_pops]
    return result

def population_from_cell_sets(exp_matrix: pd.DataFrame,
                            cell_sets: dict) -> list[Population]:
    """_summary_

    Args:
        exp_matrix (torch.Tensor): expression matrix
        cell_sets (dict): contains the mapping from population names to cell names.
        For example:
            {"IPS": ['D9_serum_C1_GCTTGAAAGCTGTCTA-1',
                    'D9.5_2i_C1_CTAGTGATCGAGCCCA-1',
                    'D9.5_2i_C2_CAGATCACAAATACAG-1',
                    'D9.5_serum_C2_ACGCCGATCGAATGCT-1']}
    Raises:
        ValueError: _description_

    Returns:
        list[Population]: list of Population objects.
    """
    population_names = list(cell_sets.keys())
    at_time = set(exp_matrix["day"].values)
    if len(at_time) > 1:
        raise ValueError("Found multiple days in the expression matrix")
    day = next(iter(at_time))
    day = int(day) if ".0" in str(day) else float(day)
    # Given an expression matrix and the list of cells pertaining to each population,
    # return the prob. distribution of each population:
    populations = population_from_ids(exp_matrix, [cell_sets[name] for name in population_names])
    populations = [Population(prob_vec = populations[i] if populations[i] is not None else None,
                              day=day, name=key, support=exp_matrix.index.values) for i, key in enumerate(population_names)]
    return populations

# def quantize_emp_dist(data: pd.DataFrame, population_assignment: NDArray | None = None)->dict:
#     """A function that produces a quantized version of a pmf given the expression matrix.
#     The quantization is simply done my means of the pre-computed Leiden clusters.

#     Args:
#         data (pd.DataFrame): gene expression matrix.
    
#     Returns:
#         Tuple: the support and the probability vector.
#     """
#     if population_assignment is None:
#         population_assignment = np.ones_like(len(data))
#     # Append weights to the DF
#     data["population_assignment"] = population_assignment
#     if "leiden" not in data.columns:
#         raise ValueError("No pre-computed Leiden clusters detected. Aborting.")
#     data_supp = data.groupby(["leiden"]).agg("mean")
#     concentration = data[data["population_assignment"]==1]["leiden"].value_counts().reindex(data_supp.index)
#     # Normalize the concentration
#     prob_weights = concentration / np.sum(concentration)
#     # Save the mapping from Leiden clusters to cells
#     cells_to_clusters = data["leiden"].to_dict()
#     clusters_to_cells_mapping = defaultdict(list)
#     for cell, cluster in cells_to_clusters.items():
#         clusters_to_cells_mapping[cluster].append(cell)
#     return {"support": np.array(data_supp),
#             "prob_vec": np.array(prob_weights),
#             "clusters_to_cells": clusters_to_cells_mapping}

def compute_cost_matrix(cloud1: np.ndarray | pd.DataFrame,
                        cloud2: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Compute the cost between two clouds of points (support of the pmfs).

    Args:
        cloud1 (np.ndarray): 2-D array, support of the first distribution
        cloud2 (np.ndarray): 2-D array, support of the second distribution
    """
    if isinstance(cloud1, pd.DataFrame) and all([ll in cloud1.columns for ll in ["day", "leiden"]]):
        cloud1.drop(columns=["day", "leiden"], axis=1, inplace=True)
    if isinstance(cloud2, pd.DataFrame) and all([ll in cloud2.columns for ll in ["day", "leiden"]]):
        cloud2.drop(columns=["day", "leiden"], axis=1, inplace=True)
    cost_matrix = pairwise_distances(cloud1, cloud2, metric="sqeuclidean", n_jobs=-1)
    cost_matrix /= np.median(cost_matrix)
    return cost_matrix

def compute_eot_plan(supp1: np.ndarray, 
                    weights1: np.ndarray,
                    supp2: np.ndarray,
                    weights2: np.ndarray, ) -> np.ndarray:
    """Computes the EOT plan between two distributions, defined, respectively, by their support and weights.

    Args:
        supp1 (np.array): the support of the first distribution 
        weights1 (np.array): the weights associated with cloud1. 
        supp2 (np.array): the support of the second distribution 
        weights2 (np.array): the weights associated with cloud2. 

    Returns:
        np.array: the EOT plan.
    """
    # Euclidean cost matrix between supports of the two marginals
    cost_matrix = compute_cost_matrix(supp1, supp2)
    # Compute the EOT plan
    if weights1 is None:
        p = np.ones(len(supp1)) / len(supp1)
    else:
        weights1 = weights1.astype('float64')
        p = weights1 / weights1.sum()

    if weights2 is None:
        q = np.ones(len(supp2)) / len(supp2)
    else:
        weights2 = weights2.astype('float64')
        q = weights2 / weights2.sum()
    gamma, logs = ot.bregman.sinkhorn(p, q, cost_matrix, reg=1e-2, method="sinkhorn_log", log=True, numItermax=20000)
    return gamma

def compute_wassertein_distance(supp1: np.ndarray, 
                                weights1: np.ndarray, 
                                supp2: np.ndarray, 
                                weights2: np.ndarray)->float:
    """_summary_

    Args:
        supp1 (np.array): _description_
        weights1 (np.array | None): _description_
        supp2 (np.array): _description_
        weights2 (np.array | None): _description_

    Returns:
        np.array: _description_
    """
    M = compute_cost_matrix(supp1, supp2)
    if weights1 is None:
        p = np.ones(len(supp1)) / len(supp1)
    else:
        weights1 = weights1.astype('float64')
        p = weights1 / weights1.sum()

    if weights2 is None:
        q = np.ones(len(supp2)) / len(supp2)
    else:
        weights2 = weights2.astype('float64')
        q = weights2 / weights2.sum()
    Wd_reg, _ = ot.sinkhorn2(a=p, b=q, M=M, reg=1e-2, method="sinkhorn_log") 
    return Wd_reg


def kl_div(p, q, eps=1e-8):
    return np.sum(p * np.log(p + eps / q + eps))


def compute_uncertainty_radius(adata: pd.DataFrame,
                              reducer: GMVAENet,
                              gumbel_temp: float,
                              gumbel_hard: bool) -> float:
    """Given a raw expression matrix, this function computes the uncertainty radius $\eta_{0}$ as follows:
       1 - Using the GM-VAE model -- the reducer -- we first map each cell to its latent Gaussian distribution.
       2 - We then compute the nominal distribution, which consists of the mixture of all latent Gaussian
       distributions learned in step 1.
       3 - Finally, the uncertainty radius is estimated as the expected KL divergence from the Mixture
       of Gaussians to each of the Gaussian distributions. 

    Args:
        data (pd.DataFrame): the original single cell data.
        reducer (GMVAENet): the trained GMVAE model.

    Returns:
        float: the uncertainty radius.
    """
    adata_copy = adata.copy(deep=True)
    if "day" in adata_copy:
        adata_copy = adata_copy.drop("day", axis=1)

    # Perform dim. reduction with the trained GM-VAE
    adata_tensor = torch.tensor(adata_copy.values, dtype=torch.float32)
    reduced_out = reducer(adata_tensor, gumbel_temp, gumbel_hard)
    # Extract the mean and covariance of each latent Gaussian
    mixture_gaussians_means = reduced_out.mean_inf
    mixture_gaussians_stds = torch.sqrt(reduced_out.var_inf)
    # Compute the KL divergence from the mixture of Gaussians to each of the latent Gaussian
    mixture_gaussians_means = reduced_out.mean_inf[:100]
    mixture_gaussians_stds = torch.sqrt(reduced_out.var_inf)[:100]

    logger.info(f"Starting the processing of kl_divs. Number of computations: {len(mixture_gaussians_means)}")
    kl_divs = np.array([kl_div_mixture_gaussians(mixture_gaussians_means, mixture_gaussians_stds, gaussian_mean, gaussian_std) 
               for (gaussian_mean, gaussian_std) in zip(mixture_gaussians_means, mixture_gaussians_stds)])
    return kl_divs.mean()

def kl_div_mixture_gaussians(mixture_gaussians_means: torch.Tensor,
                            mixture_gaussians_stds: torch.Tensor,
                            gaussian_mean: torch.Tensor,
                            gaussian_std: torch.Tensor,) -> float:
    """_summary_

    Args:
        mixture_gaussians_means (torch.Tensor): _description_
        mixture_gaussians_stds (torch.Tensor): _description_
        gaussian_mean (torch.Tensor): _description_
        gaussian_std (torch.Tensor): _description_

    Returns:
        float: _description_
    """
    num_of_components, size = mixture_gaussians_means.shape
    mixture_gaussians = Normal(mixture_gaussians_means, mixture_gaussians_stds)
    gaussian_dist = Normal(gaussian_mean, gaussian_std)
    # samples is a num_of_components x size tensor
    samples = mixture_gaussians.rsample()
    log_pi = torch.tensor([1.0 / num_of_components]).log()
    # log_q_w_components is a (num_of_components x num_of_components x size) tensor
    log_q_w_components = log_pi + mixture_gaussians.log_prob(samples.unsqueeze(1).expand(num_of_components, num_of_components, size))
    log_q_w_sum = torch.logsumexp(log_q_w_components, dim=1)
    log_p_w = gaussian_dist.log_prob(samples)
    f_w = log_q_w_sum - log_p_w
    kl_div = f_w.mean(dim=0)
    return kl_div.sum().item()

# def compute_uncertainty_radii(data:pd.DataFrame, 
#                               nominal_weights:NDArray, 
#                               num_boot:int=100,)->float:
#     """ Compute the uncertainty radii of the KL balls, centered on a nominal dist, using bootstrapping. 

#     Args:
#         data (DataFrame): the expression matrix observed on a given day, 
#         nominal_dist (array): the nominal distribution
#         num_bootst (int): number of bootstrapping rounds.
#         num_samples (int): number of samples per bootstrapping round
#     """
#     klds = []
#     num_samples = int(0.20 * data.shape[0])
#     for _ in range(num_boot):
#         data_temp = data.sample(n=num_samples, replace=True)
#         quant_dist = quantize_emp_dist(data_temp)
#         # data_support = quant_dist["support"]
#         concentration = quant_dist["prob_vec"]
#         kld = kl_div(p=concentration/np.sum(concentration), q=nominal_weights)
#         if kld == np.inf:
#             continue
#         klds.append(kld)
#     mean_kld = np.mean(klds)
#     var_kld  = np.var(klds)
#     radius = (mean_kld + 1.96 * np.sqrt(var_kld))
#     return radius 

def interpolate_distributions(dist1: torch.Tensor,
                     dist2: torch.Tensor,
                     gamma: torch.Tensor,
                     t_interpolate: float, 
                     size: int) -> torch.Tensor:
    """_summary_

    Args:
        dist1 (torch.Tensor): distribution at time 1
        dist2 (torch.Tensor): distribution at time 2
        gamma (torch.Tensor): transport plan between dist1 and dist2
        t_interpolate (float): interpolation rate
        size (int): size of the support of the interpolated distribution.

    Returns:
        torch.Tensor: interpolated distribution.
    """
    assert dist1.shape[1] == dist2.shape[1], print("Number of genes do not match in the input distributions")

    II = len(dist1)
    JJ = len(dist2)
    p = torch.flatten(gamma)
    p = p / p.sum()
    # sample indices in [1, II x JJ] according to the probability vector p.
    choices = np.random.choice(II * JJ, p=p, size=size)
    # TODO: Clarify the replacement issue
    interp_dist = torch.tensor([dist1[i // JJ] * (1 - t_interpolate) + dist2[i % JJ] * t_interpolate for i in choices])
    return interp_dist

def compute_markov_kernel(gamma: torch.Tensor) -> torch.Tensor:
    """Normalizes the transport plan to produce the associated Markov kernel.

    Args:
        gamma (torch.Tensor): the transport plan

    Returns:
        torch.Tensor: markov kernel
    """
    mkv_kern = (gamma.T / gamma.sum(dim=1)).T
    return mkv_kern

def push_forward(p: torch.Tensor, tplan: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        p (torch.Tensor): _description_
        tplan (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    tmap_kern = compute_markov_kernel(tplan)
    tmap_kern = tmap_kern.to(torch.float32)
    p = p @ tmap_kern
    # p = (p.T / torch.sum(p, dim=1)).T
    return p

def push_forward_uot_plan(populations: list[Population],
                reducer: GMVAENet, 
                cache: None,
                normalize: bool = True,) -> list[Population]:
    """Pushes the Populations to the next valid sequencing timestamp using the UOT plan. 
    The function computes the UOT plan between two consecutive timestamps, if not available in the cache.

    Args:
        populations (list[Population]): _description_
        tplan (torch.Tensor): _description_
        normalize (bool, optional): _description_. Defaults to True.
        as_list (bool, optional): _description_. Defaults to False.

    Returns:
        list[Population] | Population: _description_
    """
    keys = [p.name for p in populations]
    i = set([p.day for p in populations])
    t0 = next(iter(i))
    valid_timestamps = get_seq_timestamps(PATH_TO_CELL_DAYS)
    t1 = valid_timestamps[valid_timestamps.index(t0)+1]
    logger.info(f"Pushing populations from time {t0} to time {t1}")

    # Compute the transport plan
    tplan, source_cells_ids, target_cells_ids = get_unbalanced_ot_plan(t0, t1, reducer, cache)

    p = torch.vstack([pop.prob_vec for pop in populations if pop.prob_vec is not None])
    p = p.to(torch.float32)
    p = push_forward(p, tplan)

    result = [Population(prob_vec=p[k, :],
                        day=t1,
                        name=keys[k],
                        support=target_cells_ids) for k in range(p.shape[0])]
    return result

def push_forward_uot_multi_steps(populations: list[Population],
                                to_time: float,
                                reducer: GMVAENet,
                                cache: Path,
                                ignore_interm_steps: bool) -> tuple[list[Population], list[list[Population]]]:
    """Pushes the current population n steps ahead.
    Given the Markovian property of the MC, the Markov kernel over n steps is simply the product of the individual kernels.

    Args:
        populations (list[Population] | Population): _description_
        to_time (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        list[Population] | Population: _description_
    """
    valid_timestamps = get_seq_timestamps(PATH_TO_CELL_DAYS)
    if to_time not in valid_timestamps:
        raise ValueError("The target timestamp is not valid.")
    i = set([p.day for p in populations])
    t0 = next(iter(i))
    keys = [p.name for p in populations]

    if to_time <= t0:
        raise ValueError("End date should come after start date when computing descendants.")

    intermediate_pops = []
    if not ignore_interm_steps:
        # Keep track of the intermediate populations.
        while t0 < to_time:
            populations = push_forward_uot_plan(populations, reducer, None, False)
            intermediate_pops.append(populations)
            # Extract the new t0 from the new population
            i = set([p.day for p in populations])
            t0 = next(iter(i))
    else:
        # Compute the composite Markov kernel.
        tplans_seq = []
        while t0 < to_time:
            t1 = valid_timestamps[valid_timestamps.index(t0)+1]
            tplan, _, target_cells_ids = get_unbalanced_ot_plan(t0, t1, reducer, cache)
            tplans_seq.append(tplan)
            t0 = t1
        # The final kernel is the product of the intermediate kernels (Markov property)
        tplan_final = reduce(torch.matmul, tplans_seq)
        p = torch.vstack([pop.prob_vec for pop in populations if pop.prob_vec is not None])
        p = p.to(torch.float32)
        print(f"Final tmap: {tplan_final}")
        p = push_forward(p, tplan_final)
        populations = [Population(prob_vec=p[k, :],
                        day=t1,
                        name=keys[k],
                        support=target_cells_ids) for k in range(p.shape[0])]
    return populations, intermediate_pops

def pull_back_uot_multi_steps(populations: list[Population],
                            to_time: float,
                            reducer: GMVAENet,
                            cache: Path,
                            ignore_interm_steps: bool) -> tuple[list[Population], list[list[Population]]]:
    """_summary_

    Args:
        populations (list[Population] | Population): _description_
        to_time (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        list[Population] | Population: _description_
    """
    valid_timestamps = get_seq_timestamps(PATH_TO_CELL_DAYS)
    if to_time not in valid_timestamps:
        raise ValueError("The target timestamp is not valid.")
    i = set([p.day for p in populations])
    t0 = next(iter(i))
    if to_time >= t0:
        raise ValueError("The end date for pull back operation should come before the start date.")
    
    intermediate_pops = []

    while t0 > to_time:
        populations = pull_back_uot_plan(populations, reducer, None, False)
        if not ignore_interm_steps:
            intermediate_pops.append(populations)
        # Extract the new t0 from the new population
        i = set([p.day for p in populations])
        t0 = next(iter(i))
    return populations, intermediate_pops

def pull_back_uot_plan(populations: list[Population],
                      reducer: GMVAENet,
                      cache: Path,
                      ignore_interm_steps: bool) -> list[Population]:
    """_summary_

    Args:
        populations (list[Population] | Population): _description_
        to_time (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        list[Population] | Population: _description_
    """
    keys = [p.name for p in populations if p.prob_vec is not None]
    i = set([p.day for p in populations])
    t1 = next(iter(i))
    valid_timestamps = get_seq_timestamps(PATH_TO_CELL_DAYS)
    t0 = valid_timestamps[valid_timestamps.index(t1)-1]
    logger.info(f"Pulling back populations from time {t1} to time {t0}")

    # Compute and normalize the tmap to get its Markov kernel
    tplan, source_cells_ids, _ = get_unbalanced_ot_plan(t0, t1, reducer, cache)

    p = torch.vstack([pop.prob_vec for pop in populations if pop.prob_vec is not None])
    p = p.to(torch.float32)
    p = pull_back(p, tplan)

    result = [Population(prob_vec=p[k, :],
                         day=t0, name=keys[k],
                         support=source_cells_ids) for k in range(p.shape[0])]
    return result

def pull_back(p: torch.Tensor, tplan: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        p (torch.Tensor): _description_
        tplan (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    tmap_kern = compute_markov_kernel(tplan)
    tmap_kern = tmap_kern.to(torch.float32)
    p = p @ tmap_kern.T
    p = (p.T / torch.sum(p, dim=1)).T
    return p

def push_forward_multi_rand_tmaps(populations: list[Population],
                                num_rand_plans: int,
                                reducer: GMVAENet,
                                with_leiden_clusters: bool=False) -> tuple[list[list[Population]], list]:
    """Pushes forward the current populations to the next valid timestamp using a set of random transport plans.
    The set of random transport plans are computed between the current and next time steps using
    the get_rand_transport_plans function.

    Args:
        populations (list[Population]): _description_
        transport_maps (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    keys = [p.name for p in populations if p.prob_vec is not None]
    i = set([p.day for p in populations])
    t0 = next(iter(i))
    valid_timestamps = get_seq_timestamps(PATH_TO_CELL_DAYS)
    t1 = valid_timestamps[valid_timestamps.index(t0)+1]
    logger.info(f"Pushing populations from time {t0} to time {t1}")

    # Compute the random transport plans between t0 and t1
    rand_plans, source_cells_ids, target_cells_ids, mh_log_ratio = get_rand_transport_plans(t0, t1, reducer, num_rand_plans,
                                                                            with_leiden_clusters, CACHE)
    result = []
    p = torch.vstack([pop.prob_vec for pop in populations if pop.prob_vec is not None])
    p = p.to(torch.float32)
    with concurrent.futures.ProcessPoolExecutor(
        mp_context=multiprocessing.context.SpawnContext(), max_workers=10
    ) as executor:
        futures = [
            executor.submit(
                push_forward,
                p,
                tplan, 
            )
            for tplan in rand_plans
        ]
        with tqdm(total=len(futures)) as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                result.append(future.result())
                progress_bar.update()
    
    descendant_samples = [[Population(prob_vec=rand_sample[k, :],
                        day=t1,
                        name=keys[k],
                        support=target_cells_ids) for k in range(rand_sample.shape[0])] for rand_sample in result]
    return descendant_samples, mh_log_ratio

def push_forward_multi_steps_multi_tmaps(populations: list[Population],
                                        reducer: GMVAENet,
                                        to_time: float,
                                        num_rand_tmaps: int,
                                        pruning_size: int,
                                        ignore_interm_steps: bool) -> tuple[list[list[Population]], list[list[list[Population]]], list]:
    """Computes the push-forward distribution of a list of populations using `num_samples` random
    transport plans for a given number of timestamps. Prunes the number of particles at each
    iteration to maximum `pruning_size`.

    Args:
        populations (list[Population] | Population): _description_
        to_time (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        list[Population] | Population: _description_
    """
    sequencing_timestamps = get_seq_timestamps(PATH_TO_CELL_DAYS)
    i = set([p.day for p in populations])
    t0 = next(iter(i))
    if to_time <= t0:
        raise ValueError("End date should come after the start date.")

    intermediate_steps = []

    while t0 < to_time:

        print(f"Processing time: {t0}")
        if not isinstance(populations[0], list):
            list_of_lists_populations = [populations]
        else:
            list_of_lists_populations = populations

        new_populations = []
        mh_log_ratio_all = []
        for sample in list_of_lists_populations:
            # sample is a list of populations, i.e. a particle.
            propagated_samples, mh_log_ratio = push_forward_multi_rand_tmaps(sample, num_rand_tmaps, reducer, False)
            new_populations.append(propagated_samples)
            mh_log_ratio_all.append(mh_log_ratio)
        
        # Flatten the list[list[list[Population]]] to a list[list[Population]]
        flat_populations = [pop for sample in new_populations for pop in sample]
        # TODO: Here: compute the uncertainty propagation score
        
        # Before proceeding to the next timestamp, prune the number of particles
        new_populations = prune_particles(flat_populations, pruning_size)

        if not ignore_interm_steps:
            intermediate_steps.append(flat_populations)

        t0_idx = sequencing_timestamps.index(t0)
        t0 = sequencing_timestamps[t0_idx+1]
        populations = new_populations

    return populations, intermediate_steps, mh_log_ratio_all

def pull_back_multi_rand_tmaps(populations: list[Population],
                                num_rand_plans: int,
                                num_burnin: int,
                                reducer: GMVAENet,
                                with_leiden_clusters: bool=False) -> tuple[list[list[Population]], list]:
    """Pulls back the current populations to the previous valid timestamp using a set of random transport plans.
    The set of random transport plans are computed between the current and previous time steps using
    the get_rand_transport_plans function.

    Args:
        populations (list[Population]): _description_
        transport_maps (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    keys = [p.name for p in populations]
    i = set([p.day for p in populations])
    t1 = next(iter(i))
    valid_timestamps = get_seq_timestamps(PATH_TO_CELL_DAYS)
    t0 = valid_timestamps[valid_timestamps.index(t1)-1]
    logger.info(f"Pulling back populations from time {t1} to time {t0}")
    # Compute the random transport plans between t0 and t1
    rand_plans, source_cells_ids, target_cells_ids, mh_log_ratio= get_rand_transport_plans(t0, t1, reducer, num_rand_plans, num_burnin,
                                                                            with_leiden_clusters, CACHE)
    result = []
    p = torch.vstack([pop.prob_vec for pop in populations if pop.prob_vec is not None])
    p = p.to(torch.float32)
    with concurrent.futures.ProcessPoolExecutor(
        mp_context=multiprocessing.context.SpawnContext(), max_workers=10
    ) as executor:
        futures = [
            executor.submit(
                pull_back,
                p,
                tplan, 
            )
            for tplan in rand_plans
        ]
        with tqdm(total=len(futures)) as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                result.append(future.result())
                progress_bar.update()
    
    ancestors_samples = [[Population(prob_vec=rand_sample[k, :],
                        day=t0,
                        name=keys[k],
                        support=source_cells_ids) for k in range(rand_sample.shape[0])] for rand_sample in result]
    return ancestors_samples, mh_log_ratio

def pull_back_multi_steps_multi_tmaps(populations: list[Population],
                                        reducer: GMVAENet,
                                        to_time: float,
                                        num_samples: int,
                                        num_burnin: int,
                                        pruning_size: int,
                                        ignore_interm_steps: bool) -> tuple[list[list[Population]], list[list[list[Population]]], list]:
    """Computes the push-forward distribution of a list of populations using `num_samples` random
    transport plans for a given number of timestamps. Prunes the number of particles at each
    iteration to maximum `pruning_size`.

    Args:
        populations (list[Population] | Population): _description_
        to_time (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        list[Population] | Population: _description_
    """
    sequencing_timestamps = get_seq_timestamps(PATH_TO_CELL_DAYS)
    i = set([p.day for p in populations])
    t0 = next(iter(i))
    if to_time >= t0:
        raise ValueError("When pulling back, end date should come before the start date.")

    intermediate_steps = []

    while t0 > to_time:

        print(f"Processing time: {t0}")
        if not isinstance(populations[0], list):
            list_of_lists_populations = [populations]
        else:
            list_of_lists_populations = populations

        new_populations = []
        mh_log_ratio_all = []

        print(f"Num samples to be pulled back: {len(list_of_lists_populations)}, using {num_samples} tmaps")
        for sample in list_of_lists_populations:
            # sample is a list of populations, i.e. a particle.
            pulled_samples, mh_log_ratio = pull_back_multi_rand_tmaps(sample, num_samples, num_burnin, reducer, False)
            new_populations.append(pulled_samples)
            mh_log_ratio_all.append(mh_log_ratio)
        
        # Flatten the list[list[list[Population]]] to a list[list[Population]]
        flat_populations = [pop for sample in new_populations for pop in sample]

        # Before proceeding to the next timestamp, prune the number of particles
        new_populations = prune_particles(flat_populations, pruning_size)

        if not ignore_interm_steps:
            intermediate_steps.append(flat_populations)

        t0_idx = sequencing_timestamps.index(t0)
        t0 = sequencing_timestamps[t0_idx-1]
        populations = new_populations

    return populations, intermediate_steps, mh_log_ratio_all

def prune_particles(populations: list[list[Population]],
                    pruning_size: int)->list[list[Population]]:
    # flatten_list = [pop for sample in populations for pop in sample]
    pop_indices = np.random.choice(np.arange(len(populations)), pruning_size, replace = False)
    return [populations[idx] for idx in pop_indices]

def get_unbalanced_ot_plan(t1: float,
                           t2: float,
                           reducer: GMVAENet,
                           cache: Path | None = None) -> tuple[torch.Tensor, list, list]:
    """Computes the unbalanced entropy regularised OT plan between two timestamps.

    Args:
        t1 (float): _description_
        t2 (float): _description_

    Returns:
        torch.Tensor: _description_
    """
    if cache:
        with open(cache, "r") as ch:
            cached_uot = json.load(ch)
        if f"{t1}_{t2}" in cached_uot:
            return cached_uot[f"{t1}_{t2}"]

    start_dataset = read_daily_dataset(t1, reducer)
    source_cells_ids = list(start_dataset.index.values)
    p0_x = start_dataset.values
    del start_dataset
    gc.collect()

    end_dataset = read_daily_dataset(t2, reducer)
    target_cells_ids = list(end_dataset.index.values)
    p1_x = end_dataset.values
    del end_dataset
    gc.collect()

    uot, growth = compute_unbalanced_ot_plan(supp1=p0_x,
                                            supp2=p1_x,
                                            growth_iters=5,
                                            reg=1e-2,
                                            reg_m=[1, 50])
    if cache:
        with open(cache, "w+") as ch:
            json.dump({f"{t1}_{t2}": uot}, ch, indent=2)
    return torch.tensor(uot), source_cells_ids, target_cells_ids


def compute_unbalanced_ot_plan(supp1: np.ndarray,
                            supp2: np.ndarray,
                            growth_iters: int,
                            reg: float, 
                            reg_m: float | list,
                            weights1: np.ndarray | None = None,
                            weights2: np.ndarray | None = None, 
                            )->tuple:
    """Computes the UOT plan between two distributions, defined, respectively, by their support and weights.
    Allows for the estimation of the growth rate by solving iteratively UOT problems.

    Args:
        supp1 (np.array): the support of the first distribution 
        weights1 (np.array): the weights associated with cloud1. 
        supp2 (np.array): the support of the second distribution 
        weights2 (np.array): the weights associated with cloud2. 

    Returns:
        np.array: the Unbalanced EOT plan.
    """
    # Euclidean cost matrix between supports of the two marginals
    cost_matrix = compute_cost_matrix(supp1, supp2)
    if weights1 is None:
        p = np.ones(len(supp1)) / len(supp1)
    else:
        weights1 = weights1.astype('float32')
        dx = weights1 / weights1.sum()

    if weights2 is None:
        dy = np.ones(len(supp2)) / len(supp2)
    else:
        weights2 = weights2.astype('float32')
        q = weights2 / weights2.sum()
    
    # Initially, the growth rate is uniform across all cells
    growth = np.ones(cost_matrix.shape[0])
    for _ in range(growth_iters):
        p = growth
        q = np.ones(cost_matrix.shape[1]) * np.average(growth)
        ### Compute the UOT solution here
        uot_plan, logs = sinkhorn_unbalanced(a=p, b=q, M=cost_matrix, reg=reg, reg_m=reg_m, 
                                            method="sinkhorn", log=True)
        logger.debug(f"UOT logs: {logs}")
        # Estimate the new growth
        growth = np.sum(uot_plan, axis=1)
    return uot_plan, growth


def get_rand_transport_plans(t1: float,
                            t2: float,
                            reducer: GMVAENet,
                            num_samples: int,
                            num_burnin: int,
                            with_leiden_clusters: bool,
                            cache: None) -> tuple[list[torch.Tensor], list[tuple] | None, list[tuple] | None, list]:
    """_summary_

    Args:
        t1 (float): _description_
        t2 (float): _description_
        reducer (_type_): _description_

    Returns:
        list[torch.Tensor]: _description_
    """
    if cache:
        with open(cache, "r") as ch:
            cached_rand = json.load(ch)
        if f"{t1}_{t2}" in cached_rand:
            return cached_rand[f"{t1}_{t2}"]

    # Compute the raw datasets, that is, non-reduced, so to compute uncertainty radii.
    raw_start_dataset = read_daily_dataset(t1, None)
    raw_end_dataset = read_daily_dataset(t2, None)
    logger.info(f"Start and end dataset shapes, resp.: {raw_start_dataset.shape}, {raw_end_dataset.shape}")

    eta_0 = compute_uncertainty_radius(raw_start_dataset, reducer, gumbel_hard=True, gumbel_temp=0.1)
    eta_1 = compute_uncertainty_radius(raw_end_dataset, reducer, gumbel_hard=True, gumbel_temp=0.1)
    
    del raw_start_dataset
    del raw_end_dataset
    gc.collect()
    logger.info(f"Computed uncertainty radii: {eta_0, eta_1}")

    # Reduce the input datasets using GM-VAE.
    start_dataset = read_daily_dataset(t1, reducer)
    end_dataset = read_daily_dataset(t2, reducer)

    # Extract the source and target supports
    source_cells_idx = list(start_dataset.index.values)
    end_cells_idx = list(end_dataset.index.values)
    source_leiden_clusters = start_dataset["leiden"].values if "leiden" in start_dataset else None
    end_leiden_clusters = end_dataset["leiden"].values if "leiden" in end_dataset else None
    source_cells_mapping = [*zip(source_cells_idx, source_leiden_clusters, strict=True)] if source_leiden_clusters else source_cells_idx
    end_cells_mapping = [*zip(end_cells_idx, end_leiden_clusters, strict=True)] if end_leiden_clusters else end_cells_idx

    # Compute the Kantorovitch potentials
    lambda_1, lambda_2 = compute_kanto_potentials(start_dataset, end_dataset, reducer, eta_0, eta_1)

    # Reduce the dataset using the Leiden clusters.
    start_dataset = start_dataset.groupby("leiden").mean() if "leiden" in start_dataset else start_dataset
    end_dataset = end_dataset.groupby("leiden").mean() if "leiden" in end_dataset else end_dataset

    # Now p0_x and p1_x have a shape of: num_leiden_clusters x latent_dim if leiden_clusters else num_cells x latent_dim
    p0_x = start_dataset.values
    p1_x = end_dataset.values
    del start_dataset
    del end_dataset
    gc.collect()

    # Compute the euclidean cost matrix
    M = compute_cost_matrix(cloud1=p0_x, cloud2=p1_x)
    logger.info(f"Shape of cost matrix: {M.shape}")

    # Instantiate the HFPD-OT model
    weights1 = np.ones(len(p0_x)) / len(p0_x)
    weights2 = np.ones(len(p1_x)) / len(p1_x)

    # Dimension of the support, very big...
    dim = len(p0_x) * len(p1_x)

    target_prior = HFPDOTHyperprior(mu_0=weights1,
                                    nu_0=weights2,
                                    lambda_1=lambda_1,
                                    lambda_2=lambda_2,
                                    lambda_I_1=0.01,
                                    lambda_I_2=0.01,
                                    cost_fn=M,
                                    epsilon=1e-2)

    logger.info("Instantiated the HFPD-OT target prior")
    hfpd_ot_log_density = target_prior.hyperprior_log_prob_fn
    hfpd_ot_score_func = target_prior.hyperprior_score_fun

    sampler = MetropolisAdjustedLangevinSampler(target_log_prob_fn=hfpd_ot_log_density,
                                                target_score_fn=hfpd_ot_score_func,
                                                num_burnin=num_burnin,
                                                shape=dim,
                                                sample_from_simplex=False,
                                                num_parallel_chains=2)

    logger.info("Proceeding to sampling from the HFPD-OT target prior.")
    hfpot_plans, num_accepted_samples, _, mh_log_ratio = sampler.sample(num_samples, with_diagnostics=False,)
    hfpot_plans = hfpot_plans.reshape(-1, len(p0_x), len(p1_x))
    logger.info(f"Size of the sampled HFPDOT plans: {hfpot_plans.shape}")
    logger.info(f"Number of accepted samples: {num_accepted_samples}")
    hfpot_plans_tensors = []
    for plan in hfpot_plans:
        hfpot_plans_tensors.append(torch.tensor(plan))
    # If one or less hfpot plans were sampled, raise an error:
    if len(hfpot_plans_tensors) <= 1:
        raise ValueError("Less than two HFPDOT plans were sampled. Try increasing the number of samples.")
    return hfpot_plans_tensors, source_cells_mapping, end_cells_mapping, mh_log_ratio


def compute_transition_tabs(t1, t2, reducer, cells_sets, prunning_size, num_sample, cache):
    """_summary_

    Args:
        t1 (_type_): _description_
        t2 (_type_): _description_
        reducer (_type_): _description_
        cells_sets (_type_): _description_
        prunning_size (_type_): _description_
        num_sample (_type_): _description_
        cache (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Read the datasets
    start_dataset = read_daily_dataset(t1, reducer)
    end_dataset = read_daily_dataset(t2, reducer)

    # Extract the populations
    start_populations = population_from_cell_sets(start_dataset, cells_sets)
    end_populations = population_from_cell_sets(end_dataset, cells_sets)
    source_labels = [pop.name for pop in start_populations if pop.prob_vec is not None]
    end_labels = [pop.name for pop in end_populations if pop.prob_vec is not None]
    del start_dataset
    del end_dataset
    gc.collect()    

    # Compute the crisp UOT-based transition table
    uot_populations, _ = push_forward_uot_multi_steps(start_populations, t2, reducer, cache, True)
    uot_pop_dist = torch.vstack([pop.prob_vec for pop in uot_populations if pop.prob_vec is not None])
    end_pop_dist = torch.vstack([pop.prob_vec for pop in end_populations if pop.prob_vec is not None])
    uot_table = uot_pop_dist @ end_pop_dist.T

    # # Compute the random HFPDOT-based transition tables
    # rand_tables = []        
    # hfpdot_populations, _ , mh_log_ratio = push_forward_multi_steps_multi_tmaps(start_populations, reducer, t2, num_sample, prunning_size, True)
    # for rand_populations in hfpdot_populations:
    #     rand_pop_dist = torch.vstack([pop.prob_vec for pop in rand_populations if pop.prob_vec is not None])
    #     end_pop_dist = torch.vstack([pop.prob_vec for pop in end_populations if pop.prob_vec is not None])
    #     rand_table = rand_pop_dist @ end_pop_dist.T
    #     rand_tables.append(rand_table)

    return uot_table, source_labels, end_labels


def compute_kanto_potentials(data1, data2, reducer, eta_0, eta_1):
    return 1 / eta_0, 1 / eta_1

from collections import defaultdict

def map_identities_to_cells(path_to_cell_id)->dict:
    """
    Maps cell names to their identities.
    
    :param path_to_cell_id: Description
    :return: Description
    :rtype: dict[Any, Any]
    """
    # Read the identity scores for each sequenced cell
    cells_identities = pd.read_csv(path_to_cell_id)
    # The identity is defined by the max score 
    # Get the identity associated with the highest gene expression score
    identities = ['MEF.identity', 'Pluripotency', 'Cell.cycle', 'ER.stress',
            'Epithelial.identity', 'ECM.rearrangement', 'Apoptosis', 'SASP',
            'Neural.identity', 'Placental.identity', 'X.reactivation', 'XEN',
            'Trophoblast', 'Trophoblast progenitors',
            'Spiral Artery Trophpblast Giant Cells', 'Spongiotrophoblasts',
            'Oligodendrocyte precursor cells (OPC)', 'Astrocytes',
            'Cortical Neurons', 'RadialGlia-Id3', 'RadialGlia-Gdf10',
            'RadialGlia-Neurog2', 'Long-term MEFs', 'Embryonic mesenchyme',
            'Cxcl12 co-expressed', 'Ifitm1 co-expressed', 'Matn4 co-expressed',
            '2-cell', '4-cell', '8-cell', '16-cell', '32-cell', ] 

    cells_identities["cell_identity"] = cells_identities[identities].idxmax(axis=1)
    cells_identities = cells_identities[["id", "cell_identity"]].set_index("id").to_dict("index")
    cells_identities = {key: val["cell_identity"] for key, val in cells_identities.items()}
    identities_to_cells = defaultdict(list)
    for cell, identity in cells_identities.items():
        identities_to_cells[identity].append(cell)
    return identities_to_cells

def compute_fle_coords_with_rand_plans(start_time: int, end_time: int, reducer: GMVAENet,
                                       cells_sets: dict, cache: Path, path_fle_coords: Path,
                                       num_iterations: int, num_burnin: int, pruning_size: int,) -> dict:
    """Compute the random descendant distributions, in addition to the FLE coordinates associated with each cell in the
    computed distributions. The resulting dict contains, for each step, a list of dataframes, one per random realisation.

    Args:
        start_time (int): _description_
        end_time (int): _description_
        reducer (GMVAENet): _description_
        cells_sets (dict): _description_
        cache (Path): _description_
        path_fle_coords (Path): _description_
        num_samples (int): _description_
        pruning_size (int): _description_
        ignore_interm_steps (bool): _description_

    Returns:
        dict: _description_
    """
    # Read the dataset at the start_time
    start_dataset = read_daily_dataset(start_time, reducer)

    # Extract the populations
    start_populations = population_from_cell_sets(start_dataset, cells_sets)
    del start_dataset
    gc.collect()

    # Compute the propagated population
    _, intermediate_populations, mh_log_ratio = push_forward_multi_steps_multi_tmaps(start_populations,
                                                                                    reducer,
                                                                                    end_time,
                                                                                    num_iterations,
                                                                                    num_burnin,
                                                                                    pruning_size,
                                                                                    ignore_interm_steps=False)
    fle_coords = extract_fle_coords(Path(FLE_COORDS_PATH))
    cells_identities = read_cell_identity(CELL_IDENTITIES)
    fle_coords_with_identities = extract_fle_coords_with_identities(fle_coords, cells_identities)   

    # Get the identity associated with the highest gene expression score
    identities = ['MEF.identity', 'Pluripotency', 'Cell.cycle', 'ER.stress',
                'Epithelial.identity', 'ECM.rearrangement', 'Apoptosis', 'SASP',
                'Neural.identity', 'Placental.identity', 'X.reactivation', 'XEN',
                'Trophoblast', 'Trophoblast progenitors',
                'Spiral Artery Trophpblast Giant Cells', 'Spongiotrophoblasts',
                'Oligodendrocyte precursor cells (OPC)', 'Astrocytes',
                'Cortical Neurons', 'RadialGlia-Id3', 'RadialGlia-Gdf10',
                'RadialGlia-Neurog2', 'Long-term MEFs', 'Embryonic mesenchyme',
                'Cxcl12 co-expressed', 'Ifitm1 co-expressed', 'Matn4 co-expressed',
                '2-cell', '4-cell', '8-cell', '16-cell', '32-cell', ] 

    all_populations_metadata = defaultdict(list)
    for _, populations in tqdm(enumerate(intermediate_populations)):
        # Given a step, all_populations_metadata will store the list of DFs, one per realisation.
        # get the day from populations
        day = set(pop.day for pop in populations[0])
        if len(day) > 1:
            raise ValueError("Invalid payload, different days found in this step's samples")
        day = next(iter(day))
        all_populations_metadata[day] = []
        for sample in populations:
            sample_metadata = []
            for population in sample:
                metadata = pd.DataFrame()
                metadata["prob_vec"] = population.prob_vec
                metadata["cell_names"] = population.support
                metadata["day"] = population.day
                metadata["population"] = population.name
                sample_metadata.append(metadata)

            sample_metadata = pd.concat(sample_metadata, axis=0)
            sample_metadata = pd.merge(sample_metadata, fle_coords_with_identities, left_on="cell_names", 
                                       right_on="id", how="inner")
            sample_metadata["cell_identity"] = sample_metadata[identities].idxmax(axis=1)
                        
            all_populations_metadata[day].append(sample_metadata)

    return all_populations_metadata

def extract_fle_coords(cell_name, path_to_fle_coords) -> dict:
    # Read the fle coords if they exist
    if Path(path_to_fle_coords).is_file():
        coord_df = pd.read_csv(path_to_fle_coords, index_col="id", sep="\t")
    else:
        raise ValueError("No FLE coords were found")
    cell_indices = coord_df.index.str.contains(cell_name, regex=True)
    cell_indices = coord_df.index[cell_indices]
    # Select Serum cells only.
    cell_indices = [idx for idx in cell_indices if any([course in idx for course in ["Serum", "serum"]])]
    if len(cell_indices) == 0:
        logger.warning("The cell does not belong to the Serum path.")
        return {"x": 0., "y": 0.}
    cell_coords = coord_df.loc[cell_indices[0]]
    return {"x": cell_coords["x"], "y": cell_coords["y"]}

# def push_forward_with_rand_plans(populations: list[Population] | list[list[Population]],
#                                  rand_tmaps,
#                                  normalize=True,
#                                  pruning: bool = True,
#                                  as_list=False) -> list[list[Population]]:
#     """
#     Perfoms the one-step push-forward of the populations through a list of random transport plans.
#     Accepts either one list of population (one sample) or a list of lists of populations (multiple samples)
#     """
#     new_populations_all_realisations = []
#     if all(isinstance(pop, Population) for pop in populations):
#         populations = [populations]

#     def push_forward_one_population(populations: list[Population]):
#         """Pushes forward a single list of population (one sample) using the random transport plans.

#         Args:
#             populations (list[Population]): _description_

#         Raises:
#             ValueError: _description_
#         """
#         new_populations = []
#         keys = [p.name for p in populations]
#         i = set([p.day for p in populations])
#         if len(i) > 1:
#             raise ValueError("Found multiple days in the starting population")
#         t0 = next(iter(i))
#         t1 = t0 + 1
#         for tmap in rand_tmaps:
#             # Normalise the tmap plan to get its Markov kernel
#             tmap_kern = compute_markov_ker(tmap)
#             p = np.vstack([pop.prob_vec for pop in populations])
#             p = p @ tmap_kern
#             if normalize:
#                 p = (p.T / np.sum(p, axis=1)).T
#             result = [Population(support=None, prob_vec=p[k, :], day=t1, name=keys[k]) for k in range(p.shape[0])]
#             new_populations.append(result)
#     for population_set in populations:
#         new_populations_all_realisations.append(push_forward_one_population(population_set))
#     new_populations_all_realisations = [pop for ll in new_populations_all_realisations for pop in ll]
#     # Compute the uncertainty set radius
#     if pruning:
#         # Uniformly sample from all new populations realisations to reduce the dim. of the problem.
#         pruning_size = len(populations)
#         choices = np.random.choice(new_populations_all_realisations, pruning_size, replace = False)
#         new_populations_all_realisations = choices
#     return new_populations_all_realisations

# def multi_push_forward_with_rand_plans(populations: list[Population] | list[list[Population]],
#                                        rand_tmaps: list[list[NDArray]],
#                                        to_time: int,
#                                        pruning: bool) -> list[list[Population]]:
#     """
#     Generalizes the one-step push-forward of the populations through a list of random transport plans to multi-steps ahead.
#     Controls for the exponential growth of the number of transport plans with the pruning parameter.
#     """
#     if all(isinstance(pop, Population) for pop in populations):
#         populations = [populations]
#     day = set([p.day for population in populations for p in population])
#     if len(day) > 1:
#         raise ValueError("Found multiple days in the starting population")
#     t0 = next(iter(day))
#     assert (delta := to_time - t0) == len(rand_tmaps)
#     for idx in range(delta):
#         curr_rand_tmaps = rand_tmaps[idx]
#         populations = push_forward_with_rand_plans(populations, curr_rand_tmaps, pruning=pruning)
#     return populations


# def propagate_particles(populations: list[list[Population]]) -> list[float]:
#     """_summary_

#     Args:
#         populations (list[list[Population]]): _description_

#     Returns:
#         float: _description_
#     """
#     # First compute a mixture of population distributions with equal weight (this is our refernece measure)
#     reference_measure = [np.zeros_like(populations[0][0].prob_vec) for _ in range(len(populations[0]))]
#     wass_distance_all_pop = [[] for _ in range(len(populations[0]))]
#     pop_len = set([len(pop) for pop in populations])
#     if len(pop_len) > 1:
#         raise ValueError("The lists of populations are not aligned. Aborting")
#     for sample_idx in range(len(populations)):
#         pop_sample = populations[sample_idx]
#         for idx in range(len(pop_sample)):
#             reference_measure[idx] += (1 / len(populations)) * np.array(pop_sample[idx].prob_vec)
#     reference_measure = np.vstack(reference_measure)
#     # Compute the wasserstein distance between each distr. in populations and the reference measure
#     for pop_sample in populations:
#         for idx, pop in enumerate(pop_sample):
#             pop_proba_vec = np.array(pop.prob_vec)
#             wass_dist = compute_wasserstein_dist(pop_proba_vec, reference_measure[idx])
#             wass_distance_all_pop[idx].append(wass_dist)
#     # The uncertainity radius is the max Wasserstein distance
#     uncertainty_radius = [max(np.array(pop)) for pop in wass_distance_all_pop]
#     return uncertainty_radius
