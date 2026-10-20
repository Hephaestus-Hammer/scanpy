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
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import polars as pl
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` package (src/common)
from common import cli  # noqa: E402

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


def compute_neighbors(adata, n_neighbors, flavor, random_seed, transformer="auto"):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, method=flavor, use_rep="X_pca",
                    random_state=random_seed,
                    transformer=None if transformer == "auto" else transformer)


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # TSV has N header cols and N+1 data cols (first data col = row IDs, unnamed).
    df = pl.read_csv(args.embedding_tsv, separator="\t", skip_rows=1, has_header=False)
    embedding = df[:, 1:].to_numpy().astype(np.float64)

    adata = ad.AnnData(X=np.zeros((embedding.shape[0], 1)))
    adata.obs_names = df[:, 0].to_list()
    adata.obsm["X_pca"] = embedding

    compute_neighbors(adata, args.n_neighbors, args.flavor, args.random_seed,
                      args.transformer)

    write_neighbors_graph(adata, args.output_dir, args.name)


if __name__ == "__main__":
    main()
