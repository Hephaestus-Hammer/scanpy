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


# ---- randomized and lanczos: implicit centring on sparse input ----------------------------

def _subspace_gap(a, b):
    """Largest principal angle (radians) between the column spaces of a and b."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1, 1)))


def _adata_lowrank(seed=0, n=400, g=60, rank=8):
    """Sparse data with a clear low-rank signal, so the top components are well defined."""
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, g)) * 3 + rng.normal(size=(n, g))
    dense[rng.random((n, g)) < 0.5] = 0
    return ad.AnnData(X=sp.csr_matrix(dense))


def _args2(solver, seed, **kw):
    return SimpleNamespace(solver=solver, n_components=5, random_seed=seed, blas_threads=0, **kw)


def test_randomized_and_lanczos_match_exact_subspace():
    _, load_e, var_e, ratio_e = run_pca(_adata_lowrank(), _args2("covariance_eigh", 0))
    for solver in ("randomized", "lanczos"):
        _, load, var, ratio = run_pca(_adata_lowrank(), _args2(solver, 0))
        assert _subspace_gap(load, load_e) < 1e-3, solver
        np.testing.assert_allclose(var, var_e, rtol=1e-4)
        np.testing.assert_allclose(ratio, ratio_e, rtol=1e-4)


def test_randomized_scores_are_centred_projection():
    a = _adata_lowrank()
    emb, load, *_ = run_pca(a, _args2("randomized", 3))
    x = a.X.toarray()
    np.testing.assert_allclose(emb, (x - x.mean(0)) @ load, atol=1e-8)


def test_randomized_seed_moves_the_result():
    """Few power iterations on a flat spectrum: an approximate solver, the seed matters."""
    a = ad.AnnData(X=sp.random(300, 80, density=0.3, random_state=1, format="csr"))
    e0, *_ = run_pca(a, _args2("randomized", 0, n_iter=0, n_oversamples=2))
    e1, *_ = run_pca(a, _args2("randomized", 1, n_iter=0, n_oversamples=2))
    assert not np.allclose(np.abs(e0), np.abs(e1), atol=1e-6)
    e0b, *_ = run_pca(a, _args2("randomized", 0, n_iter=0, n_oversamples=2))
    np.testing.assert_array_equal(e0, e0b)  # same seed, same result


def test_signs_fixed_largest_loading_positive():
    for solver in ("randomized", "lanczos"):
        _, load, *_ = run_pca(_adata_lowrank(), _args2(solver, 0))
        idx = np.argmax(np.abs(load), axis=0)
        assert (load[idx, np.arange(load.shape[1])] > 0).all(), solver


def test_input_stays_sparse(monkeypatch):
    """No densifying: toarray on the input would defeat the point of these solvers."""
    a = _adata_lowrank()
    def boom(*_a, **_k):
        raise AssertionError("densified")
    monkeypatch.setattr(type(a.X), "toarray", boom)
    for solver in ("randomized", "lanczos"):
        run_pca(a, _args2(solver, 0))
