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
import h5py
import numpy as np
import polars as pl
import scanpy as sc
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` package (src/common)
from common import cli  # noqa: E402
from phases import phase  # noqa: E402
from obkit.logger import init_logger  # noqa: E402

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
    p.add_argument("--n_jobs", type=int, default=1,
                   help="scanpy settings.n_jobs: pynndescent / sklearn search threads "
                        "(1 = scanpy's default; >1 is not reproducible at a fixed seed)")
    return p.parse_args()


def _write_csr(grp, m):
    m = m.tocsr()
    grp.create_dataset("data",    data=m.data)
    grp.create_dataset("indices", data=m.indices)
    grp.create_dataset("indptr",  data=m.indptr)


def write_neighbors_graph(adata, out_dir, name):
    out = Path(out_dir) / f"{name}_neighbors.h5"
    with h5py.File(out, "w") as h5:
        # dtype="S": h5py can't write numpy unicode ('<U') arrays; bytes give portable fixed-length HDF5 strings.
        h5.create_dataset("cell_ids", data=np.array(adata.obs_names.to_list(), dtype="S"))
        _write_csr(h5, adata.obsp["distances"])  # distances flat at the root (R metrics reader)
        _write_csr(h5.create_group("connectivities"), adata.obsp["connectivities"])
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
    print(f"Full command: {' '.join(sys.argv)}")
    if args.blas_threads > 0:
        print(f"  blas threads limited to {args.blas_threads} "
              f"(OMP_NUM_THREADS was {os.environ.get('OMP_NUM_THREADS', 'unset')})")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output_dir))

    with phase("load") as attrs:
        # TSV has N header cols and N+1 data cols (first data col = row IDs, unnamed).
        df = pl.read_csv(args.embedding_tsv, separator="\t", skip_rows=1, has_header=False)
        embedding = df[:, 1:].to_numpy().astype(np.float64)

        adata = ad.AnnData(X=np.zeros((embedding.shape[0], 1)))
        adata.obs_names = df[:, 0].to_list()
        adata.obsm["X_pca"] = embedding
        attrs["n_cells"], attrs["n_dims"] = embedding.shape

    # The search and the connectivities; with pynndescent this includes numba
    # compiling its functions on first use.
    with phase("knn") as attrs:
        compute_neighbors(adata, args.n_neighbors, args.flavor, args.random_seed,
                          args.transformer, args.blas_threads, args.n_jobs)
        attrs["transformer"] = args.transformer
        attrs["n_neighbors"] = args.n_neighbors
        attrs["blas_threads"] = args.blas_threads
        attrs["n_jobs"] = args.n_jobs

    with phase("write"):
        write_neighbors_graph(adata, args.output_dir, args.name)


if __name__ == "__main__":
    main()
