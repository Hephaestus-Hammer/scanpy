"""pca.load_matrix reads every input format the input stage can write (omni, 10x, h5ad sparse
and dense), detects which it got, and keeps the layout as stored unless told to densify."""

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from pca import detect_format, load_matrix

GENES = [f"g{i}" for i in range(12)]
CELLS = [f"c{i}" for i in range(40)]


def _X():
    return sp.random(40, 12, density=0.3, random_state=0, format="csr", dtype=np.float64)


def _csc(path, X, tenx):
    with h5py.File(path, "w") as h5:
        g = h5.create_group("matrix")
        for k in ("data", "indices", "indptr"):
            g.create_dataset(k, data=getattr(X, k))
        g.create_dataset("shape", data=np.array([X.shape[1], X.shape[0]]))
        g.create_dataset("barcodes", data=np.array(CELLS, dtype="S"))
        if tenx:
            f = g.create_group("features")
            for k, v in (("id", GENES), ("name", GENES), ("feature_type", ["Gene Expression"] * len(GENES)),
                         ("genome", ["x"] * len(GENES))):
                f.create_dataset(k, data=np.array(v, dtype="S"))
            f.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))
        else:
            g.create_dataset("genes", data=np.array(GENES, dtype="S"))


def _write(path, fmt, X):
    if fmt == "omni":
        _csc(path, X, False)
    elif fmt == "10x":
        _csc(path, X, True)
    else:
        a = ad.AnnData(X=X if fmt == "h5ad" else X.toarray(), obs=pd.DataFrame(index=CELLS),
                       var=pd.DataFrame(index=GENES))
        a.write_h5ad(path)


@pytest.mark.parametrize("fmt", ["omni", "10x", "h5ad", "h5ad-dense"])
def test_every_format_loads_the_same_matrix(tmp_path, fmt):
    X = _X()
    path = tmp_path / "m.h5"
    _write(path, fmt, X)
    assert detect_format(path) == fmt.split("-")[0]
    a = load_matrix(path)
    assert sp.issparse(a.X) == (fmt != "h5ad-dense")
    assert list(a.obs_names) == CELLS and list(a.var_names) == GENES
    got = a.X.toarray() if sp.issparse(a.X) else a.X
    np.testing.assert_allclose(got, X.toarray())
    assert not sp.issparse(load_matrix(path, dense=True).X)
