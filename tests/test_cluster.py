"""Clustering reads the stored distances + connectivities graph and Leiden
separates two well-separated blobs.
"""

import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from cluster import build_adata, cluster_leiden
from knn import write_neighbors_graph

K = 15
FLAVORS = ["igraph", "leidenalg"]


@pytest.fixture
def neighbors_h5(tmp_path):
    rng = np.random.RandomState(0)
    n_per = 30
    emb = np.vstack([rng.randn(n_per, 5), rng.randn(n_per, 5) + 25.0])
    ids = [f"c{i}" for i in range(2 * n_per)]

    a = ad.AnnData(X=np.zeros((2 * n_per, 1)))
    a.obs_names = ids
    a.obsm["X_pca"] = emb
    sc.pp.neighbors(a, n_neighbors=K, method="umap", use_rep="X_pca", random_state=0)
    write_neighbors_graph(a, tmp_path, "t")
    return tmp_path / "t_neighbors.h5", ids


@pytest.mark.parametrize("flavor", FLAVORS)
def test_cluster_leiden_separates_blocks(neighbors_h5, flavor):
    path, ids = neighbors_h5
    adata, _ = build_adata(path)
    labels = cluster_leiden(adata, flavor, "RBConfiguration", resolution=1.0, random_seed=0)

    assert len(labels) == len(ids)
    n = len(ids) // 2
    assert len(set(labels[:n])) == 1 and len(set(labels[n:])) == 1
    assert set(labels[:n]) != set(labels[n:])


@pytest.mark.parametrize("flavor", FLAVORS)
def test_cluster_leiden_deterministic(neighbors_h5, flavor):
    path, _ = neighbors_h5
    a1, _ = build_adata(path)
    a2, _ = build_adata(path)
    assert (cluster_leiden(a1, flavor, "RBConfiguration", 1.0, 42)
            == cluster_leiden(a2, flavor, "RBConfiguration", 1.0, 42))


def test_cluster_leiden_until_converged(neighbors_h5):
    """n_iterations=-1 (leidenalg) still separates the blocks, and is reproducible per seed."""
    path, ids = neighbors_h5
    a1, _ = build_adata(path)
    a2, _ = build_adata(path)
    l1 = cluster_leiden(a1, "leidenalg", "RBConfiguration", 1.0, 7, n_iterations=-1)
    l2 = cluster_leiden(a2, "leidenalg", "RBConfiguration", 1.0, 7, n_iterations=-1)
    assert l1 == l2
    n = len(ids) // 2
    assert len(set(l1[:n])) == 1 and len(set(l1[n:])) == 1
