#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy import sparse
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local writers
from common import cli  # noqa: E402
#from writers import Embedding, Loadings, write_embeddings, write_loadings  # noqa: E402
#from phases import phase  # noqa: E402
#from obkit.logger import init_logger  # noqa: E402

def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `FEAT` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="FEAT module (scanpy-backed)")
    cli.add_base_args(p)              # --output_dir, --name
    cli.add_stage_args(p, "FEAT")     # --rawdata_h5ad, --normalized_h5, --filtered_cellids, --properties_info
    p.add_argument("--flavor", type=str, required=True, help="Feature selection type")
    p.add_argument("--number_selected", type=int, required=True, help="Number of features to select")
    return p.parse_args()


def _decode(x):
    return np.array([
        v.decode("utf-8") if isinstance(v, bytes) else str(v)
        for v in x
    ])


def read_tenx_matrix(h5_path):
    """Load TENx HDF5 (genes x cells) as AnnData.

    H5 stores genes x cells.
    AnnData stores cells x genes.
    """
    with h5py.File(h5_path, "r") as h5:
        g = h5["matrix"]

        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(g["shape"][:])

        if "features" in g and "id" in g["features"]:
            gene_ids = g["features/id"][:]
        elif "genes" in g:
            gene_ids = g["genes"][:]
        else:
            gene_ids = np.array([f"gene_{i}".encode() for i in range(shape[0])])

        if "barcodes" in g:
            cell_ids = g["barcodes"][:]
        else:
            cell_ids = np.array([f"cell_{i}".encode() for i in range(shape[1])])

    gene_ids = _decode(gene_ids)
    cell_ids = _decode(cell_ids)

    m = sparse.csc_matrix((data, indices, indptr), shape=shape)

    adata = sc.AnnData(X=m.T.tocsr())
    adata.obs_names = cell_ids
    adata.var_names = gene_ids
    return adata


def write_tenx_matrix(adata, h5_path):
    """Write AnnData cells x genes as TENx-like HDF5 matrix genes x cells."""
    X = adata.X
    if sparse.issparse(X):
        m = X.T.tocsc()
    else:
        m = sparse.csc_matrix(X.T)

    gene_ids = np.asarray(adata.var_names.astype(str), dtype=object)
    cell_ids = np.asarray(adata.obs_names.astype(str), dtype=object)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(h5_path, "w") as h5:
        g = h5.create_group("matrix")
        g.create_dataset("data", data=m.data, compression="gzip")
        g.create_dataset("indices", data=m.indices, compression="gzip")
        g.create_dataset("indptr", data=m.indptr, compression="gzip")
        g.create_dataset("shape", data=np.asarray(m.shape, dtype=np.int64))
        g.create_dataset("genes", data=gene_ids, dtype=str_dtype)
        g.create_dataset("barcodes", data=cell_ids, dtype=str_dtype)


def select_by_scanpy_hvg(adata, number_selected, flavor):
    """Select HVGs using Scanpy's standard HVG method."""
    adata = adata.copy()

    gene_sums = np.asarray(adata.X.sum(axis=0)).ravel()
    keep = np.isfinite(gene_sums) & (gene_sums > 0)
    print(f"keeping {keep.sum()} / {adata.n_vars} genes with nonzero sum before scanpy HVG")

    adata = adata[:, keep].copy()
    if number_selected > adata.n_vars:
        raise ValueError(
            f"number_selected={number_selected} is larger than number of features={adata.n_vars}"
        )

    sc.pp.highly_variable_genes(adata, n_top_genes=number_selected, flavor=flavor)

    selected = adata.var_names[adata.var["highly_variable"]].tolist()
    return selected[:number_selected]

# This method expects raw counts as input.
def select_by_scanpy_pearson_residuals(adata, number_selected):
    """Select HVGs by Scanpy analytic Pearson residuals."""
    # https://scanpy.readthedocs.io/en/stable/tutorials/experimental/pearson_residuals.html
    adata = adata.copy()

    sc.experimental.pp.highly_variable_genes(adata, flavor="pearson_residuals", n_top_genes=number_selected, clip=None)

    return adata.var_names[adata.var["highly_variable"]].tolist()

# This method expects raw counts as input.
def select_by_scanpy_seurat_v3(adata, number_selected):
    """Select HVGs by Scanpy's seurat_v3 flavor (vst on raw counts)."""
    # https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.highly_variable_genes.html
    adata = adata.copy()

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=number_selected)

    return adata.var_names[adata.var["highly_variable"]].tolist()

def read_raw_counts(rawdata_h5ad, adata_norm, flavor):
    """Raw counts from rawdata.h5ad, subset to the cells/genes of the normalized matrix."""
    adata_raw = sc.read_h5ad(rawdata_h5ad)

    if "counts" not in adata_raw.layers:
        raise ValueError(f"{flavor} requires rawdata.h5ad layers['counts']")

    adata_filtered = adata_raw[adata_norm.obs_names, adata_norm.var_names].copy()
    adata_filtered.X = adata_filtered.layers["counts"].copy()

    if sparse.issparse(adata_filtered.X):
        adata_filtered.X = sparse.csr_matrix(adata_filtered.X)
    else:
        adata_filtered.X = np.asarray(adata_filtered.X)

    return adata_filtered

def main():
    args = parse_args()

    # logging
    print(f"Output directory: {args.output_dir}")
    print(f"Module name: {args.name}")
    print(f"rawdata.h5ad: {args.rawdata_h5ad}")
    print(f"normalized.h5: {args.normalized_h5}")
    print(f"flavor: {args.flavor}")
    print(f"number_selected: {args.number_selected}")


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_norm = read_tenx_matrix(args.normalized_h5)

    if args.number_selected > adata_norm.n_vars:
        raise ValueError(
            f"number_selected={args.number_selected} is larger than number of features={adata_norm.n_vars}"
        )
    
    if args.flavor == "scanpy_seurat":
        sel_feats = select_by_scanpy_hvg(adata_norm, args.number_selected, flavor="seurat")

    elif args.flavor == "scanpy_cell_ranger":
        sel_feats = select_by_scanpy_hvg(adata_norm, args.number_selected, flavor="cell_ranger")

    # TODO：order by gini coef and select top  N; currently it is based on pvalue
    # elif args.flavor == "giniclust3":
    #     sel_feats = select_by_giniclust3(adata, args.number_selected)

    elif args.flavor == "pearson_residuals":
        adata_filtered = read_raw_counts(args.rawdata_h5ad, adata_norm, args.flavor)
        sel_feats = select_by_scanpy_pearson_residuals(adata_filtered, args.number_selected)

    elif args.flavor == "scanpy_seurat_v3":
        adata_filtered = read_raw_counts(args.rawdata_h5ad, adata_norm, args.flavor)
        sel_feats = select_by_scanpy_seurat_v3(adata_filtered, args.number_selected)

    else:
        raise ValueError(f"Unknown selection_type: {args.flavor}")

    print(f"length(sel_feats): {len(sel_feats)}")

    # Write a simple output file
    adata_selected = adata_norm[:, sel_feats].copy()

    output_file = output_dir / f"{args.name}_normalized_selected.h5"
    print(f"output_file: {output_file}")

    write_tenx_matrix(adata_selected, output_file)

    stat = output_file.stat()
    print(f"Results written to: {output_file}")
    print(f"size: {stat.st_size}")

if __name__ == "__main__":
    main()

