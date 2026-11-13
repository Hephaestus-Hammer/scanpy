#!/usr/bin/env python3
"""Leiden clustering module (scanpy-backed) for omnibenchmark.

Reads {name}_neighbors.h5 (the kNN distances + connectivities from the knn
entrypoint), runs Leiden on the connectivities, and writes:
  {output_dir}/{name}_clusters.tsv  — cell_id<TAB>cluster

Flavor / partition type
------------------------
--flavor igraph     sc.tl.leiden(flavor="igraph")         — igraph C-core
--flavor leidenalg  sc.tl.leiden(flavor="leidenalg", ...) — leidenalg backend

When --flavor leidenalg, --partition_type selects the objective:
  RBConfiguration   resolution-aware (Reichardt-Bornholdt)
  CPM               resolution-aware (Constant Potts Model)
  Modularity        no resolution (classic modularity)

--n_iterations
--------------
Leiden iterations, passed to sc.tl.leiden for both flavors. 2 (the default)
matches scanpy's default and earlier versions of this module; -1 iterates until
the partition stops changing. Note scanpy's igraph flavor has known problems
with -1; leidenalg handles it.

--blas_threads
--------------
0 (default) inherits OMP_NUM_THREADS etc.; N caps BLAS and OpenMP pools at N
threads around the clustering (threadpoolctl, as pca.py does). Both Leiden
backends are single-threaded C, so this only bounds the numpy/scipy work around
them; it is there so a blessed run can pin every stage to one thread.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli  # noqa: E402
from readers import read_neighbors  # noqa: E402
from phases import phase  # noqa: E402
from obkit.logger import init_logger  # noqa: E402

log = logging.getLogger(__name__)

_PARTITION_CLASSES = {
    "RBConfiguration": "RBConfigurationVertexPartition",
    "CPM": "CPMVertexPartition",
    "Modularity": "ModularityVertexPartition",
}
_RESOLUTION_AWARE = {"RBConfiguration", "CPM"}


def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `CLUST` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="Leiden clustering module (scanpy-backed)")
    cli.add_base_args(p)            # --output_dir, --name
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--neighbors_h5", dest="neighbors_h5", type=Path,
                   help="H5 file with neighbors")
    g.add_argument("--neighbors_corrected_h5", dest="neighbors_h5", type=Path,
                   help="H5 file with neighbors (corrected embedding)")
    p.add_argument("--resolution", type=float, required=True,
                   help="Resolution controlling cluster granularity")
    p.add_argument("--random_seed", type=int, required=True, help="Random seed")
    p.add_argument("--flavor", choices=["igraph", "leidenalg"], required=True,
                   help="Leiden backend flavor")
    p.add_argument("--partition_type", choices=list(_PARTITION_CLASSES),
                   default="RBConfiguration",
                   help="Partition type (only used when --flavor leidenalg)")
    p.add_argument("--n_iterations", type=int, default=2,
                   help="Leiden iterations; -1 runs until convergence")
    p.add_argument("--blas_threads", type=int, default=0,
                   help="BLAS/OpenMP threads (0 = inherit OMP_NUM_THREADS)")
    return p.parse_args()


def build_adata(neighbors_h5):
    """AnnData with the stored distances/connectivities graph wired up for scanpy."""
    distances, connectivities, cell_ids = read_neighbors(neighbors_h5)
    adata = ad.AnnData(X=np.zeros((len(cell_ids), 1)))
    adata.obs_names = cell_ids
    adata.obsp["distances"] = distances
    adata.obsp["connectivities"] = connectivities
    adata.uns["neighbors"] = {
        "distances_key": "distances",
        "connectivities_key": "connectivities",
    }
    return adata, cell_ids


def cluster_leiden(adata, flavor, partition_type, resolution, random_seed,
                   n_iterations=2):
    if flavor == "igraph": # TODO(Noemi): we want it as a constant not hard coded
        sc.tl.leiden(adata, resolution=resolution, random_state=random_seed,
                     flavor="igraph", n_iterations=n_iterations, directed=False)
    else:
        import leidenalg #TODO(Noemi): possibly change and put into env
        pt = getattr(leidenalg, _PARTITION_CLASSES[partition_type])
        kw = dict(flavor="leidenalg", partition_type=pt,
                  random_state=random_seed, n_iterations=n_iterations, directed=False)
        if partition_type in _RESOLUTION_AWARE:
            kw["resolution"] = resolution
        elif resolution != 1.0:
            log.warning("%s does not support resolution — value %.4f ignored.",
                        partition_type, resolution)
        sc.tl.leiden(adata, **kw)
    return adata.obs["leiden"].to_list()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "neighbors_h5", "flavor", "partition_type",
              "resolution", "random_seed", "n_iterations", "blas_threads"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output_dir))

    with phase("load") as attrs:
        adata, cell_ids = build_adata(args.neighbors_h5)
        attrs["n_cells"] = adata.n_obs

    with phase("leiden") as attrs, threadpool_limits(
        limits=args.blas_threads if args.blas_threads > 0 else None
    ):
        labels = cluster_leiden(
            adata, args.flavor, args.partition_type, args.resolution, args.random_seed,
            args.n_iterations,
        )
        attrs["flavor"] = args.flavor
        attrs["n_iterations"] = args.n_iterations
        attrs["blas_threads"] = args.blas_threads
        attrs["n_clusters"] = len(set(labels))

    with phase("write"):
        out = Path(args.output_dir) / f"{args.name}_clusters.tsv"
        pl.DataFrame({"cell_id": cell_ids, "cluster": labels}).write_csv(out, separator="\t")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
