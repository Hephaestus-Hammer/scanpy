# Scanpy

Scanpy-backed PCA module for omnibenchmark scRNA pipelines.

## Setup

```sh
pixi install
pixi run check
```

`pixi run check` imports all runtime dependencies and prints `OK`. Run it after install to confirm the environment is healthy.

## Usage

### PCA

```sh
pixi run python pca.py \
  --output_dir <dir> \
  --name <name> \
  --normalized.h5 <normalized.h5> \
  --selected.genes <selected.genes.gz> \
  --solver <arpack|randomized|full|covariance_eigh> \
  --n_components <int> \
  --random_seed <int>
```

Output: `<output_dir>/<name>_<solver>_n_<n_components>.h5` — see [`docs/pca_output.md`](docs/pca_output.md) for the full format spec.

#### Validation


```sh
pixi run validate <output_dir>/<name>_pca.h5
```

Exit codes: `0` = valid, `1` = validation failure, `2` = IO / usage error.

## Conda environment export

```sh
pixi run export-env
```

Exports the resolved environment to `envs/scanpy.yml`. The environment is named after the repo root folder.

## Citation

If you use this module in your research, please cite it using the information in `CITATION.cff`.
