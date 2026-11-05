"""Format-level test for the kNN graph artifact: flat CSR distances at the file
root plus a nested /connectivities CSR, round-tripping through
readers.read_neighbors.
"""

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

from knn import write_neighbors_graph
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
    write_neighbors_graph(adata, tmp_path / "t_neighbors.h5")

    p = tmp_path / "t_neighbors.h5"
    with h5py.File(p, "r") as h5:
        # distances flat at the root, connectivities under their own group.
        assert set(h5.keys()) == {"cell_ids", "data", "indices", "indptr", "connectivities"}
        assert set(h5["connectivities"].keys()) == {"data", "indices", "indptr"}

    distances, connectivities, cell_ids = read_neighbors(p)
    assert cell_ids == ids
    np.testing.assert_array_equal(distances.toarray(), d.toarray())
    np.testing.assert_array_equal(connectivities.toarray(), c.toarray())
