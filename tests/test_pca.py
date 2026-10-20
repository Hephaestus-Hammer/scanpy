"""PCA solvers: covariance_eigh is accepted and agrees with arpack, and does not
depend on the seed."""

import numpy as np
import scipy.sparse as sp
import anndata as ad
from types import SimpleNamespace

from pca import run_pca


def _adata(seed=0, n=300, g=40):
    X = sp.random(n, g, density=0.3, random_state=seed, format="csr") * 5
    return ad.AnnData(X=X.astype(np.float64))


def _args(solver, seed):
    return SimpleNamespace(solver=solver, n_components=5, random_seed=seed, blas_threads=0)


def _abs(m):
    # PC signs are arbitrary between solvers; compare up to sign.
    return np.abs(m)


def test_covariance_eigh_matches_arpack():
    emb_c, *_ = run_pca(_adata(), _args("covariance_eigh", 0))
    emb_a, *_ = run_pca(_adata(), _args("arpack", 0))
    np.testing.assert_allclose(_abs(emb_c), _abs(emb_a), rtol=1e-6, atol=1e-8)


def test_covariance_eigh_ignores_seed():
    emb_0, *_ = run_pca(_adata(), _args("covariance_eigh", 0))
    emb_1, *_ = run_pca(_adata(), _args("covariance_eigh", 1))
    np.testing.assert_array_equal(emb_0, emb_1)
