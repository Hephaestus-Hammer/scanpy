#!/usr/bin/env python3
"""kNN graph module (scanpy-backed) for omnibenchmark.

Output HDF5: {output_dir}/{name}_neighbors.h5
  Flat CSR of the kNN distance graph at the file root, matching the R metrics
  reader (graph.R::read_csr_h5), which reads these datasets from the root:
    /cell_ids   string array (n_cells,)
    /data       CSR data
    /indices    CSR column indices (0-based)
    /indptr     CSR row pointers
  The connectivities scanpy builds alongside the distances are stored under a
  nested group so the clustering stage can read them directly instead of
  reconstructing them (both graphs share /cell_ids):
    /connectivities/{data,indices,indptr}  CSR (n_cells, n_cells)

Neighbour search (--transformer)
--------------------------------
auto         scanpy's default: exact for euclidean below 8,192 cells, then
             pynndescent (approximate, seeded) above, silently.
sklearn      exact brute force (sklearn KNeighborsTransformer) at every size.
pynndescent  approximate at every size.

Stored neighbours
-----------------
scanpy builds the connectivities (what Leiden clusters on) from n_neighbors - 1
neighbours per cell, self excluded, on every path. The *distances* it stores differ:
the exact path rebuilds them with the same n_neighbors - 1, the pynndescent path keeps
the transformer's raw n_neighbors. The distances written here are trimmed to each
cell's n_neighbors - 1 nearest (self excluded) on every path, so every kNN module
stores the same k. Connectivities are written as scanpy built them, untouched.

Connectivities (--connectivities)
---------------------------------
The edge weights written as /connectivities, on the same neighbours:
native  what --flavor gives (scanpy's UMAP or gauss weights): today's behaviour.
umap    UMAP's fuzzy simplicial set (scanpy's method="umap"), whatever --flavor is.
snn     Seurat's shared-nearest-neighbour graph: for cells i and j with neighbour
        sets N_i and N_j (each cell itself included, k = n_neighbors),
        |N_i & N_j| / |N_i | N_j|, over every pair sharing a neighbour; edges below
        --snn_prune (default 1/15, Seurat's prune.SNN) removed; diagonal 1. The same
        definition every kNN module uses, so --connectivities snn gives one graph
        whichever module found the neighbours.

Threads (--n_jobs)
------------------
scanpy's settings.n_jobs defaults to 1, and scanpy passes it to pynndescent, which
then runs its numba-parallel search on one thread (numba.set_num_threads(n_jobs)),
and to sklearn's brute-force search (joblib). --n_jobs sets it: 1 (default) is
scanpy's own behaviour; N > 1 runs NN-descent on N threads, which is faster and is
not reproducible at a fixed seed (parallel updates race).

Threads (--blas_threads)
------------------------
0 (default)  inherit OMP_NUM_THREADS etc. from the caller (Snakemake sets them to
             the rule's threads).
N            cap BLAS and OpenMP pools at N threads for the search and the
             connectivities (threadpoolctl, as pca.py does). This is what bounds
             the exact sklearn path, whose distance computation is a BLAS GEMM.
             pynndescent parallelises with numba, which threadpoolctl does not
             control; set NUMBA_NUM_THREADS for that.
"""

import argparse
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` package (src/common)
from common import cli  # noqa: E402

from writers import NeighborGraph, read_embeddings, write_graph  # noqa: E402

try:
    from phases import phase  # noqa: E402
    from obkit.logger import init_logger  # noqa: E402
except ImportError:                               # obkit not installed: profiling off
    from contextlib import contextmanager

    def init_logger(*args, **kwargs):
        pass

    @contextmanager
    def phase(name):
        yield {}


def parse_args():
    p = argparse.ArgumentParser(description="kNN graph module (scanpy-backed)")
    cli.add_base_args(p)                # --output_dir, --name
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--embedding_tsv", dest="embedding_tsv", type=Path,
                   help="Embedding TSV (cell_ids as rownames) from any "
                        "embedding-producing stage (PCA, ISOMAP, ...)")
    g.add_argument("--corrected_tsv", dest="embedding_tsv", type=Path,
                   help="Batch-corrected embedding TSV (cell_ids as rownames)")
    p.add_argument("--n_neighbors", type=int, required=True,
                   help="Number of nearest neighbors")
    p.add_argument("--flavor", type=str, required=True,
                   choices=["umap", "gauss"], help="Method to compute connectivities")
    p.add_argument("--random_seed", type=int, required=True, help="Random seed")
    p.add_argument("--transformer", type=str, default="auto",
                   choices=["auto", "sklearn", "pynndescent"],
                   help="Neighbour search: auto = scanpy's size-based default, "
                        "sklearn = exact brute force, pynndescent = approximate")
    p.add_argument("--blas_threads", type=int, default=0,
                   help="BLAS/OpenMP threads (0 = inherit OMP_NUM_THREADS)")
    p.add_argument("--connectivities", type=str, default="native",
                   choices=["native", "umap", "snn"],
                   help="edge weights: native (--flavor's), umap or snn (Seurat's SNN); see module docstring")
    p.add_argument("--snn_prune", type=float, default=1 / 15,
                   help="--connectivities snn: drop edges below this Jaccard index (Seurat's prune.SNN)")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="scanpy settings.n_jobs: pynndescent / sklearn search threads "
                        "(1 = scanpy's default; >1 is not reproducible at a fixed seed)")
    return p.parse_args()


def write_neighbors_graph(adata, out):
    write_graph(NeighborGraph(adata.obsp["distances"], adata.obsp["connectivities"],
                              adata.obs_names.to_list()), out)
    print(f"  wrote: {out}")


def trim_distances(d, k):
    """Each row's k nearest stored entries, self excluded (stored zeros to other cells
    are duplicates, real neighbours, and kept). Ties broken by column index."""
    c = d.tocoo()
    keep = c.row != c.col
    rows, cols, dist = c.row[keep], c.col[keep], c.data[keep]
    order = np.lexsort((cols, dist, rows))
    rows, cols, dist = rows[order], cols[order], dist[order]
    rank = np.arange(len(rows)) - np.searchsorted(rows, rows)
    take = rank < k
    return sp.csr_matrix((dist[take], (rows[take], cols[take])), shape=d.shape)


def snn_connectivities(distances, prune=1 / 15):
    """Seurat's SNN graph from a kNN graph's neighbour sets (its stored entries, each cell
    itself added): the Jaccard index |N_i & N_j| / |N_i | N_j| for every pair sharing a
    neighbour, entries below `prune` dropped. Symmetric, diagonal 1."""
    n = distances.shape[0]
    a = sp.csr_matrix((np.ones(distances.nnz, dtype=np.float32), distances.indices.copy(),
                       distances.indptr.copy()), shape=(n, n))
    a = ((a + sp.identity(n, dtype=np.float32, format="csr")) > 0).astype(np.float32)
    k = np.asarray(a.sum(axis=1)).ravel()
    shared = (a @ a.T).tocoo()
    jac = shared.data / (k[shared.row] + k[shared.col] - shared.data)
    keep = jac >= prune
    return sp.csr_matrix((jac[keep], (shared.row[keep], shared.col[keep])), shape=(n, n))


def compute_neighbors(adata, n_neighbors, flavor, random_seed, transformer="auto",
                      blas_threads=0, n_jobs=1):
    old = sc.settings.n_jobs
    sc.settings.n_jobs = n_jobs
    try:
        with threadpool_limits(limits=blas_threads if blas_threads > 0 else None):
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, method=flavor, use_rep="X_pca",
                            random_state=random_seed,
                            transformer=None if transformer == "auto" else transformer)
    finally:
        sc.settings.n_jobs = old
    if "distances" in adata.obsp:
        adata.obsp["distances"] = trim_distances(adata.obsp["distances"], n_neighbors - 1)


def main():
    args = parse_args()

    # some LOGGING
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "embedding_tsv", "n_neighbors", "flavor", "random_seed",
              "transformer", "n_jobs", "blas_threads", "connectivities"):
        print(f"  {k}: {getattr(args, k)}")
    if args.blas_threads > 0:
        print(f"  blas threads limited to {args.blas_threads} "
              f"(OMP_NUM_THREADS was {os.environ.get('OMP_NUM_THREADS', 'unset')})")
    outputs = dict(args.outputs)
    for key, path in outputs.items():
        print(f"[outputs] {key}: {path}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    oid = "neighbors_corrected_h5" if "neighbors_corrected_h5" in dict(args.outputs) else "neighbors_h5"  # NNG_C
    out = cli.output_path(args, oid, "_neighbors.h5")
    init_logger(str(out.parent))

    with phase("load") as attrs:
        emb = read_embeddings(args.embedding_tsv)

        adata = ad.AnnData(X=np.zeros((emb.matrix.shape[0], 1)))
        adata.obs_names = emb.row_ids
        adata.obsm["X_pca"] = emb.matrix
        attrs["n_cells"], attrs["n_dims"] = emb.matrix.shape

    # The search and the connectivities; with pynndescent this includes numba
    # compiling its functions on first use.
    with phase("knn") as attrs:
        method = "umap" if args.connectivities == "umap" else args.flavor
        compute_neighbors(adata, args.n_neighbors, method, args.random_seed,
                          args.transformer, args.blas_threads, args.n_jobs)
        if args.connectivities == "snn":
            adata.obsp["connectivities"] = snn_connectivities(adata.obsp["distances"], args.snn_prune)
        attrs["connectivities"] = args.connectivities
        attrs["transformer"] = args.transformer
        attrs["n_neighbors"] = args.n_neighbors
        attrs["blas_threads"] = args.blas_threads
        attrs["n_jobs"] = args.n_jobs

    with phase("write"):
        write_neighbors_graph(adata, out)


if __name__ == "__main__":
    main()
