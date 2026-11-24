"""Reusable writers for objects produced by this module, conforming to the omni-scrna benchmark spec."""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

@dataclass
class Embedding:
    matrix: np.ndarray        # shape (n_cells, n_dims)
    row_ids: list             # cell barcodes, length n_cells
    col_names: list = field(default_factory=list)  # dim labels; auto-generated if empty


@dataclass
class Loadings:
    matrix: np.ndarray        # shape (n_genes, n_dims)
    row_ids: list             # gene ids, length n_genes
    col_names: list = field(default_factory=list)  # dim labels; auto-generated if empty


def _col_names(embedding):
    if embedding.col_names:
        return embedding.col_names
    return [f"dim_{i + 1}" for i in range(embedding.matrix.shape[1])]


def _write_tsv(path, embedding, row_label):
    # Header has N cols, data rows N+1 (row_ids unnamed first column);
    # read.table(f, header=TRUE) auto-promotes the extra leading column to row.names.
    cols = _col_names(embedding)
    df = pl.from_numpy(embedding.matrix, schema=cols).insert_column(
        0, pl.Series("", embedding.row_ids))
    with open(path, "w") as f:
        f.write(row_label + "\t" + "\t".join(cols) + "\n")
        df.write_csv(f, separator="\t", include_header=False)


def write_embeddings(obj, path, format="tsv"):
    if format == "tsv":
        _write_tsv(path, obj, "cell_id")
    else:
        raise ValueError(f"unsupported format: {format!r}")


def write_loadings(obj, path, format="tsv"):
    # Same on-disk layout as embeddings, but rows are genes: first column is gene_id.
    if format == "tsv":
        _write_tsv(path, obj, "gene_id")
    else:
        raise ValueError(f"unsupported format: {format!r}")


# --- read/write pairs, one per stage output -----------------------------------
# Each reader returns exactly what its writer takes, so a stage's output object can be
# handed to the next stage with or without the file in between (same layouts as
# omni-scrna/rapids-singlecell's writers).

def read_embeddings(path, format="tsv"):
    """Inverse of write_embeddings. polars, which parses floats exactly."""
    if format != "tsv":
        raise ValueError(f"unsupported format: {format!r}")
    # N header names, N+1 data columns (first = unnamed row ids)
    df = pl.read_csv(path, separator="\t", skip_rows=1, has_header=False)
    cols = pl.read_csv(path, separator="\t", n_rows=0).columns[1:]
    return Embedding(df[:, 1:].to_numpy().astype(np.float64), df[:, 0].to_list(), list(cols))


@dataclass
class NeighborGraph:
    distances: object       # n_cells x n_cells, scipy sparse
    connectivities: object  # n_cells x n_cells, scipy sparse
    row_ids: list           # cell barcodes, length n_cells


def write_graph(obj, path, format="h5"):
    """{name}_neighbors.h5: cell_ids, the distance CSR flat at the root (the R metrics
    reader), connectivities nested."""
    import h5py
    if format != "h5":
        raise ValueError(f"unsupported format: {format!r}")
    with h5py.File(path, "w") as h5:
        # dtype="S": h5py can't write numpy unicode ('<U') arrays
        h5.create_dataset("cell_ids", data=np.array(obj.row_ids, dtype="S"))
        _write_csr(h5, obj.distances)  # root first, then the group: same file layout as before
        _write_csr(h5.create_group("connectivities"), obj.connectivities)


def _write_csr(grp, m):
    m = m.tocsr()
    grp.create_dataset("data",    data=m.data)
    grp.create_dataset("indices", data=m.indices)
    grp.create_dataset("indptr",  data=m.indptr)


def read_graph(path, format="h5"):
    """Inverse of write_graph."""
    from readers import read_neighbors
    if format != "h5":
        raise ValueError(f"unsupported format: {format!r}")
    distances, connectivities, row_ids = read_neighbors(path)
    return NeighborGraph(distances, connectivities, row_ids)


@dataclass
class Labels:
    values: list   # length n_cells
    row_ids: list  # length n_cells


def write_labels(obj, path, format="tsv"):
    """{name}_clusters.tsv: cell_id<TAB>cluster."""
    if format != "tsv":
        raise ValueError(f"unsupported format: {format!r}")
    pl.DataFrame({"cell_id": obj.row_ids, "cluster": obj.values}).write_csv(path, separator="\t")


def read_labels(path, format="tsv"):
    """Inverse of write_labels; labels stay strings."""
    if format != "tsv":
        raise ValueError(f"unsupported format: {format!r}")
    df = pl.read_csv(path, separator="\t", schema_overrides={"cluster": pl.String})
    return Labels(df["cluster"].to_list(), df["cell_id"].to_list())
