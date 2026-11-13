"""knn.py and cluster.py run end to end and log load / compute / write phases to
<output_dir>/obkit-events.jsonl, as pca.py does."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _phases(out_dir):
    ev = [json.loads(line) for line in (out_dir / "obkit-events.jsonl").read_text().splitlines()]
    return [(e["event"], e["phase"]) for e in ev], ev


def _expected(*names):
    return [(n, p) for n in names for p in ("start", "end")]


def test_knn_and_cluster_log_phases(tmp_path):
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(c, 0.3, (60, 5)) for c in (0, 4, 8)])
    tsv = tmp_path / "e_embedding.tsv"
    with open(tsv, "w") as f:
        f.write("\t".join(f"PC{i + 1}" for i in range(emb.shape[1])) + "\n")
        for i, row in enumerate(emb):
            f.write(f"c{i}\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")

    knn_out, cl_out = tmp_path / "knn", tmp_path / "cl"
    subprocess.run([sys.executable, str(ROOT / "knn.py"), "--output_dir", str(knn_out), "--name", "nn",
                    "--embedding_tsv", str(tsv), "--n_neighbors", "10", "--flavor", "umap",
                    "--random_seed", "0"], check=True)
    seq, ev = _phases(knn_out)
    assert seq == _expected("load", "knn", "write")
    assert ev[1]["attrs"] == {"n_cells": 180, "n_dims": 5}
    assert ev[3]["attrs"]["transformer"] == "auto"

    subprocess.run([sys.executable, str(ROOT / "cluster.py"), "--output_dir", str(cl_out), "--name", "cl",
                    "--neighbors_h5", str(knn_out / "nn_neighbors.h5"), "--flavor", "igraph",
                    "--resolution", "1.0", "--random_seed", "0", "--blas_threads", "1"],
                   check=True)
    seq, ev = _phases(cl_out)
    assert seq == _expected("load", "leiden", "write")
    assert ev[3]["attrs"]["n_clusters"] >= 3
    assert ev[3]["attrs"]["blas_threads"] == 1
