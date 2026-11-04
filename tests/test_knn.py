"""Format-level test for the kNN graph artifact: flat CSR distances at the file
root plus a nested /connectivities CSR, round-tripping through
readers.read_neighbors.
"""

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

from knn import compute_neighbors, write_neighbors_graph
from readers import read_neighbors


def test_neighbors_h5_layout_and_roundtrip(tmp_path):
    ids = ["a", "b", "c", "d"]
    d = sp.csr_matrix(
        np.array([[0, 1, 2, 0], [1, 0, 0, 3], [2, 0, 0, 4], [0, 3, 4, 0]], dtype=float)
    )
    c = sp.csr_matrix(
        np.array([[0, 0.5, 0.2, 0], [0.5, 0, 0, 0.1], [0.2, 0, 0, 0.3], [0, 0.1, 0.3, 0]])
    )
    adata = ad.AnnData(X=np.zeros((4, 1)))
    adata.obs_names = ids
    adata.obsp["distances"] = d
    adata.obsp["connectivities"] = c
    write_neighbors_graph(adata, tmp_path, "t")

    p = tmp_path / "t_neighbors.h5"
    with h5py.File(p, "r") as h5:
        # distances flat at the root, connectivities under their own group.
        assert set(h5.keys()) == {"cell_ids", "data", "indices", "indptr", "connectivities"}
        assert set(h5["connectivities"].keys()) == {"data", "indices", "indptr"}

    distances, connectivities, cell_ids = read_neighbors(p)
    assert cell_ids == ids
    np.testing.assert_array_equal(distances.toarray(), d.toarray())
    np.testing.assert_array_equal(connectivities.toarray(), c.toarray())


def _neighbor_sets(distances):
    d = distances.tocsr()
    return [set(d.indices[d.indptr[i]:d.indptr[i + 1]]) for i in range(d.shape[0])]


def test_transformer_sklearn_is_exact_above_scanpy_threshold():
    """Above 8,192 cells scanpy's default switches to approximate search;
    --transformer sklearn must still return the exact neighbours."""
    n, k = 8300, 15
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, 5))
    adata = ad.AnnData(X=np.zeros((n, 1)))
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obsm["X_pca"] = emb
    compute_neighbors(adata, k, "umap", 0, transformer="sklearn")

    # scanpy counts the cell itself among n_neighbors, so k - 1 others.
    sample = rng.choice(n, 50, replace=False)
    got = _neighbor_sets(adata.obsp["distances"])
    for i in sample:
        dist = ((emb - emb[i]) ** 2).sum(axis=1)
        dist[i] = np.inf
        assert got[i] == set(np.argsort(dist)[: k - 1])


def test_blas_threads_caps_the_pools_during_the_search(monkeypatch):
    """--blas_threads N: every BLAS/OpenMP pool is capped at N while scanpy runs."""
    import knn
    from threadpoolctl import threadpool_info

    seen = []
    monkeypatch.setattr(knn.sc.pp, "neighbors",
                        lambda *a, **k: seen.append([p["num_threads"] for p in threadpool_info()]))
    adata = ad.AnnData(X=np.zeros((10, 1)))
    adata.obsm["X_pca"] = np.zeros((10, 2))
    compute_neighbors(adata, 5, "umap", 0, transformer="sklearn", blas_threads=1)
    assert seen and seen[0] and all(n == 1 for n in seen[0])
