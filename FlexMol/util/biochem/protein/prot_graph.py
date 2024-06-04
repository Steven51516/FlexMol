import dgl
import torch
import os
import dgl.backend as F
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sys
import contextlib

@contextlib.contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    yield
    sys.stdout = original_stdout


with suppress_stdout():
    from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_k_nn_edges, add_peptide_bonds
    from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, hydrogen_bond_acceptor, hydrogen_bond_donor
    from graphein.protein.features.sequence.embeddings import compute_esm_embedding
    from graphein.protein.features.sequence.utils import (
        subset_by_node_feature_value,
    )
    from graphein.protein.graphs import initialise_graph_with_metadata, add_nodes_to_graph
    from graphein.protein.graphs import annotate_node_metadata, annotate_graph_metadata, annotate_edge_metadata, compute_edges
    from graphein.protein.graphs import process_dataframe, deprotonate_structure, convert_structure_to_centroids, remove_insertions


import networkx as nx
import numpy as np
import pandas as pd

from biopandas.pdb import PandasPdb

#TODO Clean this file

def general_amino_acid_property(
    n: str, d: Dict[str, Any], return_array: bool = False
) -> Union[pd.Series, np.ndarray]:

    def onehot(idx, size):
        """One hot encoder"""
        onehot = torch.zeros(size)
        onehot[idx] = 1
        return np.array(onehot)


    residue_charge = {'CYS': -0.64, 'HIS': -0.29, 'ASN': -1.22, 'GLN': -1.22, 'SER': -0.80, 'THR': -0.80, 'TYR': -0.80,
                            'TRP': -0.79, 'ALA': -0.37, 'PHE': -0.37, 'GLY': -0.37, 'ILE': -0.37, 'VAL': -0.37, 'MET': -0.37,
                            'PRO': 0.0, 'LEU': -0.37, 'GLU': -1.37, 'ASP': -1.37, 'LYS': -0.36, 'ARG': -1.65}


    residue_polarity = {'CYS': 'polar', 'HIS': 'polar', 'ASN': 'polar', 'GLN': 'polar', 'SER': 'polar', 'THR': 'polar', 'TYR': 'polar', 'TRP': 'polar',
                                'ALA': 'apolar', 'PHE': 'apolar', 'GLY': 'apolar', 'ILE': 'apolar', 'VAL': 'apolar', 'MET': 'apolar', 'PRO': 'apolar', 'LEU': 'apolar',
                                'GLU': 'neg_charged', 'ASP': 'neg_charged', 'LYS': 'neg_charged', 'ARG': 'pos_charged'}

    polarity_encoding = {
        'apolar': 0, 'polar': 1, 'neg_charged': 2, 'pos_charged': 3}

    amino_acid = d["residue_name"]
    features = []
    features.append(residue_charge.get(amino_acid))
    polarity = residue_polarity.get(amino_acid)
    encoded_polarity = onehot(polarity_encoding[polarity], len(polarity_encoding))
    features.extend(encoded_polarity)
    if return_array:
        features = np.array(features)

    d["amino_acid_property"] = features

    return features



def esm_residue_embedding(
    G: nx.Graph,
    model_name: str = "esm1b_t33_650M_UR50S",
    output_layer: int = 33,
) -> nx.Graph:
    for chain in G.graph["chain_ids"]:
        embedding = compute_esm_embedding_long(
            G.graph[f"sequence_{chain}"],
            representation="residue",
            model_name=model_name,
            output_layer=output_layer,
        )
        # remove start and end tokens from per-token residue embeddings
        embedding = embedding[0, 1:-1]
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["esm_embedding"] = embedding[i]

    return G



# copied from dgl
def merge(graphs):
    r"""Merge a sequence of graphs together into a single graph.

    Nodes and edges that exist in ``graphs[i+1]`` but not in ``dgl.merge(graphs[0:i+1])``
    will be added to ``dgl.merge(graphs[0:i+1])`` along with their data.
    Nodes that exist in both ``dgl.merge(graphs[0:i+1])`` and ``graphs[i+1]``
    will be updated with ``graphs[i+1]``'s data if they do not match.

    Parameters
    ----------
    graphs : list[DGLGraph]
        Input graphs.

    Returns
    -------
    DGLGraph
        The merged graph.

    Notes
    ----------
    * Inplace updates are applied to a new, empty graph.
    * Features that exist in ``dgl.graphs[i+1]`` will be created in
      ``dgl.merge(dgl.graphs[i+1])`` if they do not already exist.

    Examples
    ----------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch
    >>> g = dgl.graph((torch.tensor([0,1]), torch.tensor([2,3])))
    >>> g.ndata["x"] = torch.zeros(4)
    >>> h = dgl.graph((torch.tensor([1,2]), torch.tensor([0,4])))
    >>> h.ndata["x"] = torch.ones(5)
    >>> m = dgl.merge([g, h])

    ``m`` now contains edges and nodes from ``h`` and ``g``.

    >>> m.edges()
    (tensor([0, 1, 1, 2]), tensor([2, 3, 0, 4]))
    >>> m.nodes()
    tensor([0, 1, 2, 3, 4])

    ``g``'s data has updated with ``h``'s in ``m``.

    >>> m.ndata["x"]
    tensor([1., 1., 1., 1., 1.])

    See Also
    ----------
    add_nodes
    add_edges
    """

    ref = graphs[0]
    ntypes = ref.ntypes
    etypes = ref.canonical_etypes
    data_dict = {etype: ([], []) for etype in etypes}
    num_nodes_dict = {ntype: 0 for ntype in ntypes}
    merged = dgl.heterograph(data_dict, num_nodes_dict, ref.idtype, ref.device)

    # Merge edges and edge data.
    for etype in etypes:
        unmerged_us = []
        unmerged_vs = []
        edata_frames = []
        for graph in graphs:
            etype_id = graph.get_etype_id(etype)
            us, vs = graph.edges(etype=etype)
            unmerged_us.append(us)
            unmerged_vs.append(vs)
            edge_data = graph._edge_frames[etype_id]
            edata_frames.append(edge_data)
        keys = ref.edges[etype].data.keys()
        if len(keys) == 0:
            edges_data = None
        else:
            edges_data = {k: F.cat([f[k] for f in edata_frames], dim=0) for k in keys}
        merged_us = F.copy_to(F.astype(F.cat(unmerged_us, dim=0), ref.idtype), ref.device)
        merged_vs = F.copy_to(F.astype(F.cat(unmerged_vs, dim=0), ref.idtype), ref.device)
        merged.add_edges(merged_us, merged_vs, edges_data, etype)

    for next_graph in graphs:
        for ntype in ntypes:
            merged_ntype_id = merged.get_ntype_id(ntype)
            next_ntype_id = next_graph.get_ntype_id(ntype)
            next_ndata = next_graph._node_frames[next_ntype_id]
            node_diff = (next_graph.num_nodes(ntype=ntype) -
                         merged.num_nodes(ntype=ntype))
            n_extra_nodes = max(0, node_diff)
            merged.add_nodes(n_extra_nodes, ntype=ntype)
            next_nodes = F.arange(
                0, next_graph.num_nodes(ntype=ntype), merged.idtype, merged.device
            )
            merged._node_frames[merged_ntype_id].update_row(
                next_nodes, next_ndata
            )

    return merged




# I changed several lines from the graphein library since the model_index caused errors sometimes
def read_pdb_to_dataframe(
    path: Optional[Union[str, os.PathLike]] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    model_index: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reads PDB file to ``PandasPDB`` object.

    Returns ``atomic_df``, which is a DataFrame enumerating all atoms and
    their cartesian coordinates in 3D space. Also contains associated metadata
    from the PDB file.

    :param path: path to PDB or MMTF file. Defaults to ``None``.
    :type path: str, optional
    :param pdb_code: 4-character PDB accession. Defaults to ``None``.
    :type pdb_code: str, optional
    :param uniprot_id: UniProt ID to build graph from AlphaFoldDB. Defaults to
        ``None``.
    :type uniprot_id: str, optional
    :param model_index: Index of model to read. Only relevant for structures
        containing ensembles. Defaults to ``1``.
    :type model_index: int, optional
    :returns: ``pd.DataFrame`` containing protein structure
    :rtype: pd.DataFrame
    """
    if pdb_code is None and path is None and uniprot_id is None:
        raise NameError(
            "One of pdb_code, path or uniprot_id must be specified!"
        )

    if path is not None:
        if isinstance(path, Path):
            path = os.fsdecode(path)
        if (
            path.endswith(".pdb")
            or path.endswith(".pdb.gz")
            or path.endswith(".ent")
        ):
            atomic_df = PandasPdb().read_pdb(path)
        # elif path.endswith(".mmtf") or path.endswith(".mmtf.gz"):
        #     atomic_df = PandasMmtf().read_mmtf(path)
        else:
            raise ValueError(
                f"File {path} must be either .pdb(.gz), .mmtf(.gz) or .ent, not {path.split('.')[-1]}"
            )
    elif uniprot_id is not None:
        atomic_df = PandasPdb().fetch_pdb(
            uniprot_id=uniprot_id, source="alphafold2-v3"
        )
    else:
        atomic_df = PandasPdb().fetch_pdb(pdb_code)

    if(model_index):
        atomic_df = atomic_df.get_model(model_index)
        if len(atomic_df.df["ATOM"]) == 0:
            raise ValueError(f"No model found for index: {model_index}")

    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])

    

# calculate esm embedding for long sequences > 1022
def compute_esm_embedding_long(sequence, representation="residue", model_name="esm1b_t33_650M_UR50S", output_layer=33, chunk_size=1022):
    seq_length = len(sequence)
    embeddings = []
    token_dim = None
    overlap_size = 100

    num_chunks = max(1, (seq_length - overlap_size) // (chunk_size - overlap_size) + 1)

    for i in range(num_chunks):
        start_idx = i * (chunk_size - overlap_size)
        end_idx = min(start_idx + chunk_size, seq_length)

        chunk_embedding = compute_esm_embedding(sequence[start_idx:end_idx], representation, model_name, output_layer)
        chunk_embedding = chunk_embedding[0, 1:-1] 

        if token_dim is None:
            token_dim = chunk_embedding.shape[1]

        embeddings.append(chunk_embedding)

    # Combine embeddings from all chunks
    # For overlapping regions, take the average of embeddings from adjacent chunks
    combined_embedding = embeddings[0]
    for i in range(1, len(embeddings)):
        overlap_with_prev = overlap_size if i * (chunk_size - overlap_size) < seq_length else 0
        if overlap_with_prev > 0:
            combined_embedding[-overlap_with_prev:] = (combined_embedding[-overlap_with_prev:] + embeddings[i][:overlap_with_prev]) / 2
            combined_embedding = np.concatenate([combined_embedding, embeddings[i][overlap_with_prev:]], axis=0)
        else:
            combined_embedding = np.concatenate([combined_embedding, embeddings[i]], axis=0)

    # add empty start and end token that will be removed by graphein
    empty_start_token = np.zeros((1, token_dim))
    empty_end_token = np.zeros((1, token_dim))
    combined_embedding = np.concatenate([empty_start_token, combined_embedding, empty_end_token], axis=0)

    combined_embedding = combined_embedding[np.newaxis, :]
    return combined_embedding



def create_prot_dgl_graph(pdb_name, data_dir):

    def get_graph(pdb_name, data_dir):
        processing_funcs = [deprotonate_structure, convert_structure_to_centroids, remove_insertions]
        raw_df = read_pdb_to_dataframe(path=os.path.join(data_dir, f"{pdb_name}.pdb"))
        df = process_dataframe(raw_df, atom_df_processing_funcs=processing_funcs)
        g = initialise_graph_with_metadata(protein_df=df,
                                        raw_pdb_df=raw_df,
                                        pdb_code=pdb_name,
                                        granularity="centroid"
                                        )
        g = add_nodes_to_graph(g)
        configured_add_knn_bonds = partial(add_k_nn_edges, k=4)
        g = compute_edges(g, get_contacts_config=None, funcs=[add_peptide_bonds, add_hydrogen_bond_interactions, configured_add_knn_bonds])
        g = annotate_node_metadata(g, [amino_acid_one_hot, hydrogen_bond_acceptor, hydrogen_bond_donor, general_amino_acid_property])
        node_features = ['amino_acid_one_hot', 'amino_acid_property']
        g_dgl = dgl.from_networkx(g, node_attrs=node_features)
        features_to_concat = [g_dgl.ndata[attr].float() for attr in node_features]
        g_dgl.ndata['h'] = torch.cat(features_to_concat, dim=1)
        g_dgl = dgl.add_self_loop(g_dgl)
        return g_dgl

    pdb_path = os.path.join(data_dir, pdb_name)
    if os.path.isfile(pdb_path + ".pdb"):
        single_graph = get_graph(pdb_name, data_dir)
        return single_graph
    elif os.path.isdir(pdb_path): # not recommanded
            graphs = []
            for file in os.listdir(pdb_path):
                if file.endswith(".pdb"):
                    pdb_name = os.path.splitext(file)[0]
                    graphs.append(get_graph(pdb_name, pdb_path+'/'))
            merged_graph = merge(graphs)
            return merged_graph
    else:
        print("error finding protein strcuture!")



def create_prot_esm_dgl_graph(pdb_name, data_dir):
    def get_graph(pdb_name, data_dir):
        processing_funcs = [deprotonate_structure, convert_structure_to_centroids, remove_insertions]
        raw_df = read_pdb_to_dataframe(path=os.path.join(data_dir, f"{pdb_name}.pdb"))
        df = process_dataframe(raw_df, atom_df_processing_funcs=processing_funcs)
        g = initialise_graph_with_metadata(protein_df=df,
                                        raw_pdb_df=raw_df, 
                                        pdb_code=pdb_name,
                                        granularity="centroid" 
                                        )
        g = add_nodes_to_graph(g)
        configured_add_knn_bonds = partial(add_k_nn_edges, k=4)
        g = compute_edges(g, get_contacts_config=None, funcs=[add_peptide_bonds, add_hydrogen_bond_interactions, configured_add_knn_bonds])
        g = annotate_graph_metadata(g, [esm_residue_embedding])
        g = annotate_node_metadata(g, [amino_acid_one_hot, hydrogen_bond_acceptor, hydrogen_bond_donor, general_amino_acid_property])
        node_features = ['esm_embedding', 'amino_acid_one_hot', 'amino_acid_property']
        g_dgl = dgl.from_networkx(g, node_attrs=node_features)
        features_to_concat = [g_dgl.ndata[attr].float() for attr in node_features]
        g_dgl.ndata['h'] = torch.cat(features_to_concat, dim=1)
        g_dgl = dgl.add_self_loop(g_dgl)
        return g_dgl
    
    pdb_path = os.path.join(data_dir, pdb_name)
    if os.path.isfile(pdb_path + ".pdb"):
        single_graph = get_graph(pdb_name, data_dir)
        return single_graph
    elif os.path.isdir(pdb_path):
            graphs = []
            for file in os.listdir(pdb_path):
                if file.endswith(".pdb"):
                    pdb_name = os.path.splitext(file)[0]
                    graphs.append(get_graph(pdb_name, pdb_path+'/'))
            merged_graph = merge(graphs)
            return merged_graph
    else:
        print("error finding protein strcuture!")
        