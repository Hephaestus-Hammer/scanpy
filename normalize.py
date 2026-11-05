#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from datetime import datetime
import h5py
import gzip
import numpy as np
from scipy import sparse
import anndata as ad
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local writers
from common import cli  # noqa: E402
#from writers import Embedding, Loadings, write_embeddings, write_loadings  # noqa: E402
#from phases import phase  # noqa: E402
#from obkit.logger import init_logger  # noqa: E402

def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `NORM` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="NORM module (scanpy-backed)")
    cli.add_base_args(p)              # --output_dir, --name
    cli.add_stage_args(p, "NORM")     # --rawdata_h5ad, --filtered_cellids
    p.add_argument("--flavor", type=str, required=True, help="Normalization type")
    return p.parse_args()

def write_h5_matrix(x, output_file, feature_ids, cell_ids):
    """Write sparse matrix to H5."""
    x = sparse.csc_matrix(x)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    feature_ids = np.asarray(feature_ids, dtype=object)
    cell_ids = np.asarray(cell_ids, dtype=object)

    with h5py.File(output_file, "w") as f:
        g = f.create_group("matrix")

        g.create_dataset("data", data=x.data)
        g.create_dataset("indices", data=x.indices.astype(np.uint32))
        g.create_dataset("indptr", data=x.indptr.astype(np.uint32))
        g.create_dataset("shape", data=np.asarray(x.shape, dtype=np.uint32))

        g.create_dataset("genes", data=feature_ids, dtype=string_dtype)
        g.create_dataset("barcodes", data=cell_ids, dtype=string_dtype)

        features = g.create_group("features")
        features.create_dataset("id", data=feature_ids, dtype=string_dtype)
        features.create_dataset("name", data=feature_ids, dtype=string_dtype)
        features.create_dataset(
            "feature_type",
            data=np.asarray(["Gene Expression"] * len(feature_ids), dtype=object),
            dtype=string_dtype,
        )
        features.create_dataset(
            "genome",
            data=np.asarray([""] * len(feature_ids), dtype=object),
            dtype=string_dtype,
        )

def main():
    args = parse_args()

    # logging
    print(f"Output directory: {args.output_dir}")
    print(f"Module name: {args.name}")
    print(f"normalization method: {args.flavor}")
    print(f"rawdata.h5ad: {args.rawdata_h5ad}")
    print(f"filtered.cellids: {args.filtered_cellids}")

    # Read filtered cell IDs
    with gzip.open(args.filtered_cellids, "rt") as f:
        cellids = [line.strip() for line in f if line.strip()]

    print(f"  number of filtered cells: {len(cellids)}")

    # Read filtered feature IDs
    with gzip.open(args.filtered_featureids, "rt") as f:
        featureids = [line.strip() for line in f if line.strip()]

    print(f"  number of filtered features: {len(featureids)}")
    
    adata = sc.read_h5ad(args.rawdata_h5ad)
    adata = adata[cellids, featureids].copy()
    adata.X = adata.layers["counts"].copy()

    print(f"  shape of adata.X: {adata.X.shape}")

    # Normalize
    if args.flavor == "log1pCP10k":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    elif args.flavor == "analyticalPearsonResiduals":
        sc.experimental.pp.normalize_pearson_residuals(adata)
    else:
        raise ValueError(f"Unknown flavor: {args.flavor}")

    # transpose to features x cells
    d = adata.X.T
    feature_ids = list(adata.var_names)
    cell_ids = list(adata.obs_names)

    print(f"class(d): {type(d)}")
    print(f"dim(d): {d.shape}")

    # Write a simple output file
    output_file = cli.output_path(args, "normalized_h5", "_normalized.h5")
    print(f"output_file: {output_file}")

    # transpose to features x cells
    write_h5_matrix(d, output_file, feature_ids=feature_ids, cell_ids=cell_ids)
    
    stat = output_file.stat()
    print(f"file size: {stat.st_size}")
    print(f"ctime: {datetime.fromtimestamp(stat.st_ctime)}")

if __name__ == "__main__":
    main()

