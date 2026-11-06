#!/usr/bin/env python3
"""
PCA module (scanpy-backed) for omnibenchmark.

Output
------
File: {output_dir}/{name}_embedding.tsv

Tab-separated, with header row:
  cell_id  PC1  PC2  ...  PC{n_components}

One row per cell; values are float64 PCA scores.

Implementation notes
--------------------
- Genes are mean-centered (only) before PCA via sc.pp.pca(zero_center=True).
  No per-gene variance scaling — matches scrapper/rapids-singlecell. If
  alternative scaling is needed later, expose it as a new --pca_type variant
  rather than as an independent flag.

--solver randomized / lanczos (added for concordance-bench Phase 2)
-------------------------------------------------------------------
Both now run here, on the sparse matrix, with the centring kept implicit (fix 2b
below), instead of going through sc.pp.pca:

  randomized  Halko-Martinsson-Tropp randomized range finder: a Gaussian test
              matrix of n_components + --n_oversamples columns, --n_iter power
              iterations (QR-normalised), then an exact SVD of the small projected
              matrix. Seed = the test matrix. Approximate; the seed moves the result.
  lanczos     Lanczos bidiagonalization with partial reorthogonalisation (SciPy's
              PROPACK, svds(solver="propack")), the family R's irlba belongs to,
              stopped at --tol (default 1e-5, irlba's default). Seed = the start
              vector. Close to exact; the seed moves the result within the tolerance.

Every matrix product is X @ V - 1 (mu^T V) or X^T U - mu (1^T U): X stays CSR and
nothing n_cells x n_genes is materialised. Signs are fixed as sklearn does for PCA
(each component's largest-magnitude loading positive), so seeds compare like for like.
arpack, covariance_eigh and full still go through sc.pp.pca, unchanged.

History: before this, --solver randomized was silently ignored (measured 2026-08-15)
------------------------------------------------------------------------------------
This module always feeds sc.pp.pca a SPARSE matrix (load_matrix returns CSR),
and sklearn's PCA accepts only {'arpack', 'covariance_eigh'} for sparse input.
scanpy therefore coerces svd_solver='randomized' to 'arpack' and warns:

    UserWarning: Ignoring svd_solver='randomized' and using arpack,
    sklearn.decomposition._pca.PCA (with sparse input) only supports
    dict_keys(['arpack', 'covariance_eigh'])

The warning goes to stderr and is invisible in benchmark results, so the
'randomized' arm has been producing byte-identical output to 'arpack' — a
duplicate job, not a second solver. `choices=["arpack", "randomized"]` in the
parser advertises a solver this module cannot deliver.

WHY sklearn refuses: PCA must mean-centre, and for sparse input the centring
has to stay implicit — arpack does it through a LinearOperator, covariance_eigh
through the Gram matrix. sklearn's randomized path calls randomized_svd on the
centred matrix, which would mean materialising X - mean, i.e. densifying. So
sklearn rejects it rather than silently blowing up memory.

This is an sklearn-PCA limitation, NOT a mathematical one. Implicit centring
composes fine with a randomised range finder — (X - 1 mu^T) O = X O - 1 (mu^T O),
all matvecs — which is how R's irlba does it via center=. Demonstrated by the
benchmark's own sibling modules, all on the same sparse CSR input and without
densifying: rapids-singlecell randomized-halko (seed-sensitive), scrapper
random (seed-sensitive), and sklearn's own TruncatedSVD with
algorithm="randomized" (which is allowed precisely because it does not centre).

Measured on be1 (1715 x 2000), n_comps=10, vs sparse arpack:
    dense full          1.0e-12   (a third exact solver; adds nothing)
    dense randomized    9.9e-4    (genuinely different)
    dense randomized, seed 42 vs 43   1.7e-3   (genuinely seed-sensitive)
Note the seed effect exceeds the approximation bias. Also covariance_eigh
agrees with arpack to 8e-13, so on sparse input the solver axis is degenerate.

FIX OPTIONS, in increasing order of work:
  1. Drop "randomized" from choices — stop advertising it. Honest, one line.
  2. Densify only when solver == randomized, documenting the memory cost
     (27MB for be1, ~2.5GB for pbmc at 157k x 2000). Buys a CPU approximate arm
     with a real seed axis -- but note the benchmark ALREADY has approximate,
     seed-sensitive arms from scrapper random (CPU) and rapids randomized-halko
     (GPU), so this is a second one, not the only one.
  2b. Better if the arm is wanted: implement implicit centring around a
     randomised range finder, as irlba does, and keep the input sparse. More
     work than densifying, but it is the thing sklearn is missing rather than a
     workaround for it.
  3. Expose covariance_eigh instead — but it is numerically the same as arpack
     here, so it adds a job, not information.

--solver covariance_eigh
------------------------
A direct method: eigendecomposition of the gene covariance (Gram) matrix, with
no iterations, tolerance or random start, so --random_seed has no effect on it.
Works on sparse and dense input. Exposed for deterministic baselines, where it
removes arpack's random start vector as a source of run-to-run variation even
though the two agree closely (see above).
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local writers
from common import cli  # noqa: E402
from writers import Embedding, Loadings, write_embeddings, write_loadings  # noqa: E402
from phases import phase  # noqa: E402
from obkit.logger import init_logger  # noqa: E402


def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `PCA` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="PCA module (scanpy-backed)")
    cli.add_base_args(p)             # --output_dir, --name
    cli.add_stage_args(p, "PCA")     # --normalized_selected_h5
    p.add_argument("--solver", type=str, required=True,
                   choices=["arpack", "randomized", "lanczos", "full", "covariance_eigh"],
                   help="PCA solver (randomized and lanczos: implicit centring, sparse input kept)")
    p.add_argument("--n_iter", type=int, default=-1,
                   help="randomized: power iterations (-1 = sklearn's rule: 7 if n_components "
                        "< 10%% of min(n_cells, n_genes), else 4)")
    p.add_argument("--n_oversamples", type=int, default=10,
                   help="randomized: extra test vectors beyond n_components")
    p.add_argument("--tol", type=float, default=1e-5,
                   help="lanczos: convergence tolerance (irlba's default 1e-5; 0 = machine precision)")
    # Dense vs sparse is its own axis, not a side effect of the solver: the
    # paper contrasts the two, and only the dense path puts real work through
    # BLAS-3. Materialised in the loader so peak RSS reflects the format under
    # test instead of holding a sparse and a dense copy at once.
    p.add_argument("--dense", type=str, default="false", choices=["true", "false"],
                   help="materialise the matrix dense before PCA")
    # Snakemake exports OMP_NUM_THREADS = <rule threads> (default 1), pinning
    # the BLAS to one thread whichever implementation is linked. 0 = inherit.
    # Compute precision. The FEAT matrix arrives float64 (R numeric), so this
    # module computes in float64 today; scanpy's own dtype argument sets the
    # dtype of the *result*, not of the computation, so it cannot isolate this.
    # "input" leaves the matrix as read.
    p.add_argument("--dtype", type=str, default="input",
                   choices=["input", "float32", "float64"],
                   help="cast the matrix before PCA (input = leave as read)")
    p.add_argument("--blas_threads", type=int, default=0,
                   help="BLAS threads (0 = inherit OMP_NUM_THREADS)")
    p.add_argument("--n_components", type=int, required=True,
                   help="Number of principal components to compute")
    p.add_argument("--random_seed", type=int, required=True,
                   help="Seed for randomized solvers (and for reproducibility)")
    return p.parse_args()


def load_matrix(h5_path, dense=False):
    with h5py.File(h5_path, "r") as h5:
        g = h5["matrix"]
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(g["shape"][:])
        gene_ids = g["genes"][:].astype(str)
        cell_ids = g["barcodes"][:].astype(str)

    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()  # cells x genes
    if dense:
        # Drop the sparse copy before AnnData is built, so peak reflects one
        # representation rather than both.
        X = X.toarray()
        print(f"  dense matrix: {X.nbytes / 1e6:.0f} MB")
    adata = ad.AnnData(X=X)
    adata.obs_names = cell_ids
    adata.var_names = gene_ids
    return adata


def _centred_ops(X):
    """Products with the centred matrix X - 1 mu^T, without forming it."""
    n = X.shape[0]
    mu = np.asarray(X.mean(axis=0), dtype=X.dtype).ravel()

    def mm(V):   # (X - 1 mu^T) V
        V = np.asarray(V)
        return np.asarray(X @ V) - np.outer(np.ones(n), mu @ V) if V.ndim == 2 else np.asarray(X @ V).ravel() - mu @ V

    def rmm(U):  # (X - 1 mu^T)^T U
        U = np.asarray(U)
        return np.asarray(X.T @ U) - np.outer(mu, U.sum(axis=0)) if U.ndim == 2 else np.asarray(X.T @ U).ravel() - mu * U.sum()

    return mu, mm, rmm


def _total_variance(X, mu):
    n = X.shape[0]
    sq = np.asarray(X.multiply(X).sum(axis=0)).ravel() if sp.issparse(X) else (np.asarray(X) ** 2).sum(axis=0)
    return float(((sq - n * mu ** 2) / (n - 1)).sum())


def _flip(u, vt):
    """sklearn's PCA sign convention: each component's largest-|loading| entry positive."""
    signs = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
    signs[signs == 0] = 1
    return u * signs, vt * signs[:, None]


def centred_svd(X, n_components, solver, seed, n_iter=-1, n_oversamples=10, tol=1e-5):
    """Top components of X - 1 mu^T by a randomized or Lanczos solver, X kept sparse.

    Returns (embedding, loadings V, variance, variance_ratio) as sc.pp.pca stores them.
    The embedding is the projection (X - 1 mu^T) V onto the returned loadings, not the
    solver's U S: the two agree for an exact solver, and for an approximate one the
    projection keeps the embedding consistent with the loadings written beside it.
    variance = each score column's variance; ratio against the total variance.
    """
    from scipy.sparse.linalg import LinearOperator, svds

    n, g = X.shape
    mu, mm, rmm = _centred_ops(X)
    rng = np.random.default_rng(seed)
    if solver == "randomized":
        if n_iter < 0:
            n_iter = 7 if n_components < 0.1 * min(n, g) else 4
        omega = rng.standard_normal((g, n_components + n_oversamples)).astype(X.dtype, copy=False)
        q, _ = np.linalg.qr(mm(omega))
        for _ in range(n_iter):
            z, _ = np.linalg.qr(rmm(q))
            q, _ = np.linalg.qr(mm(z))
        b = rmm(q).T                                   # Q^T (X - 1 mu^T), small: l x g
        ub, s, vt = np.linalg.svd(b, full_matrices=False)
        u = q @ ub
        u, s, vt = u[:, :n_components], s[:n_components], vt[:n_components]
    elif solver == "lanczos":
        op = LinearOperator((n, g), matvec=mm, rmatvec=rmm, matmat=mm, rmatmat=rmm, dtype=X.dtype)
        u, s, vt = svds(op, k=n_components, solver="propack", tol=tol,
                        random_state=np.random.default_rng(seed))
        order = np.argsort(s)[::-1]
        u, s, vt = u[:, order], s[order], vt[order]
    else:
        raise ValueError(solver)
    u, vt = _flip(u, vt)
    emb = mm(vt.T)
    variance = (emb ** 2).sum(axis=0) / (n - 1)
    return emb, vt.T, variance, variance / _total_variance(X, mu)


def run_pca(adata, args):
    # Chunked mode triggers IncrementalPCA. It is most effective with backed
    # AnnData; for in-memory data it provides no memory benefit and is slower.
    # Sparse inputs get densified per chunk during partial_fit.
    #chunked = args.chunked == "true"
    chunked = False

    limits = args.blas_threads if args.blas_threads > 0 else None
    if limits:
        print(f"  blas threads limited to {limits} "
              f"(OMP_NUM_THREADS was {os.environ.get('OMP_NUM_THREADS', 'unset')})")
    if args.solver in ("randomized", "lanczos"):
        X = adata.X  # computed in X's dtype (--dtype), like the other solvers
        with threadpool_limits(limits=limits):
            return centred_svd(X, args.n_components, args.solver, args.random_seed,
                               n_iter=getattr(args, "n_iter", -1),
                               n_oversamples=getattr(args, "n_oversamples", 10),
                               tol=getattr(args, "tol", 1e-5))
    with threadpool_limits(limits=limits):
      sc.pp.pca(
        adata,
        n_comps=args.n_components,
        zero_center=True,
        svd_solver=args.solver if not chunked else None,
        random_state=args.random_seed,
        chunked=chunked,
        chunk_size=args.chunk_size if chunked else None,
      )

    embedding = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    loadings = np.asarray(adata.varm["PCs"], dtype=np.float64)
    variance = np.asarray(adata.uns["pca"]["variance"], dtype=np.float64)
    variance_ratio = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=np.float64)
    return embedding, loadings, variance, variance_ratio


def validate_args(args):
    #chunked = args.chunked == "true"
    chunked = False

    if chunked:
        if args.chunk_size is None:
            sys.exit("error: --chunk_size is required when --chunked=true")
        if args.solver is not None:
            sys.exit("error: --solver must not be set when --chunked=true (IncrementalPCA ignores it)")
    else:
        if args.chunk_size is not None:
            sys.exit("error: --chunk_size must not be set when --chunked=false")
        if args.solver is None:
            sys.exit("error: --solver is required when --chunked=false")



def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "normalized_selected_h5", "solver", "n_components", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output_dir))

    with phase("load") as attrs:
        if args.solver == "full" and args.dense != "true":
            raise SystemExit("--solver full is dense-only (sklearn refuses sparse "
                             "input); pass --dense true rather than let scanpy "
                             "silently substitute another solver")
        adata = load_matrix(args.normalized_selected_h5, dense=args.dense == "true")
        if args.dtype != "input":
            adata.X = adata.X.astype(args.dtype)
        print(f"  compute dtype: {adata.X.dtype}")
        attrs["n_cells"] = adata.n_obs
        attrs["n_genes"] = adata.n_vars
    gene_ids = np.array(adata.var_names)
    cell_ids = np.array(adata.obs_names)
    print(f"  matrix (cells x genes): {adata.shape}")

    with phase("pca") as attrs:
        embedding, loadings, variance, variance_ratio = run_pca(adata, args)
        attrs["solver"] = args.solver or "chunked"
        attrs["n_components"] = args.n_components
    print(f"  embedding: {embedding.shape}, loadings: {loadings.shape}")

    with phase("write"):
        col_names = [f"PC{i + 1}" for i in range(embedding.shape[1])]
        embedding_out = Path(args.output_dir) / f"{args.name}_embedding.tsv"
        write_embeddings(Embedding(embedding, list(cell_ids), col_names), embedding_out)

        loadings_out = Path(args.output_dir) / f"{args.name}_loadings.tsv"
        write_loadings(Loadings(loadings, list(gene_ids), col_names), loadings_out)
    print(f"  wrote: {embedding_out}")
    print(f"  wrote: {loadings_out}")


if __name__ == "__main__":
    main()
