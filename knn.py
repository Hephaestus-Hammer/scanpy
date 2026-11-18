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
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` package (src/common)
from common import cli  # noqa: E402
from writers import NeighborGraph, read_embeddings, write_graph  # noqa: E402

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
    return p.parse_args()


def write_neighbors_graph(adata, out_dir, name):
    out = Path(out_dir) / f"{name}_neighbors.h5"
    write_graph(NeighborGraph(adata.obsp["distances"], adata.obsp["connectivities"],
                              adata.obs_names.to_list()), out)
    print(f"  wrote: {out}")


def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    emb = read_embeddings(args.embedding_tsv)

    adata = ad.AnnData(X=np.zeros((emb.matrix.shape[0], 1)))
    adata.obs_names = emb.row_ids
    adata.obsm["X_pca"] = emb.matrix

    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, method=args.flavor,
                    use_rep="X_pca", random_state=args.random_seed)

    write_neighbors_graph(adata, args.output_dir, args.name)


if __name__ == "__main__":
    main()
