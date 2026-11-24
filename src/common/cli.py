"""Shared CLI helpers for omnibenchmark module entrypoints (Python).

Reserved, overwrite-on-update path (``src/common/python/``) — see AGENTS.md.

The module author writes their **own** ``argparse``-based CLI and owns the parser.
This module just *injects* the shared, synced contract onto it — the universal
base args and the stage's I/O contract — so the author doesn't re-type (and can't
drift from) what the benchmark defines, while their own method parameters stay
hand-written and fully visible in their entrypoint::

    import argparse
    from common import cli

    p = argparse.ArgumentParser()
    cli.add_base_args(p)                 # --output_dir, --name  (schema/_base.json), --output
    cli.add_stage_args(p, "embedding")   # the stage I/O contract (schema/embedding.json)
    # the author's own method params — plain argparse, fully visible & owned:
    p.add_argument("--solver", choices=["arpack", "randomized"], required=True)
    p.add_argument("--n_components", type=int, required=True)
    args = p.parse_args()

The shared args come from JSON in ``src/common/schema/`` and are added identically
here and in ``src/common/r/cli.R`` so Python and R entrypoints share one contract.
Schema shape::

    {
      "stage": "two-filter",
      "benchmark": "omni-scrna/split-stages-plan",
      "args": [
        {"flag": "--pcas.tsv", "dest": "pcas", "type": "path", "help": "..."},
        {"flag": "--solver", "type": "string", "choices": ["arpack", "randomized"]}
      ]
    }

Conventions for schema-declared args: each is **required** (a run is reproducible
from its invocation line); ``dest`` defaults to the flag with dots/dashes turned
into ``_`` (``--pcas.tsv`` -> ``pcas_tsv``) unless the schema overrides it. Types:
``path | string | integer | number``. An optional ``choices`` list restricts
accepted values (an enum), like ``argparse``.

Two synced schema files back the helpers:

  * ``_base.json``        — universal args every module gets
                            (``--output_dir``, ``--name``); added by ``add_base_args``.
  * ``<stage-id>.json``   — the stage's I/O contract (benchmark-owned); added by
                            ``add_stage_args(parser, "<interface>")``.

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def _find_schema_dir() -> Path:
    # Works under both layouts: the copied-in `src/common/cli.py` (schema is a sibling)
    # and the template's `src/common/python/cli.py` (schema is one level up).
    here = Path(__file__).resolve().parent
    for base in (here, here.parent):
        if (base / "schema").is_dir():
            return base / "schema"
    return here / "schema"


_SCHEMA_DIR = _find_schema_dir()
_TYPES = {"path": Path, "string": str, "integer": int, "number": float}
_BASE_SCHEMA = "_base"  # universal args (--output_dir, --name)

def _default_dest(flag: str) -> str:
    return flag.lstrip("-").replace(".", "_").replace("-", "_")


def _read_schema(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _add_args(parser: argparse.ArgumentParser, args: list[dict]) -> argparse.ArgumentParser:
    """Add each schema arg to ``parser`` as a required option (the shared
    convention: a run is reproducible from its invocation line)."""
    for arg in args:
        parser.add_argument(
            arg["flag"],
            required=True,
            dest=arg.get("dest", _default_dest(arg["flag"])),
            type=_TYPES.get(arg["type"], str),
            choices=arg.get("choices"),  # None -> unconstrained
            help=arg.get("help", ""),
        )
    return parser


def add_base_args(parser: argparse.ArgumentParser,
                  schema_dir: Path = _SCHEMA_DIR) -> argparse.ArgumentParser:
    """Add the universal base args (``--output_dir``, ``--name``) from
    ``schema/_base.json`` to the author's parser. Returns the parser for chaining."""
    base_path = schema_dir / f"{_BASE_SCHEMA}.json"
    if not base_path.is_file():
        raise SystemExit(f"base schema not found: {base_path}")
    _add_args(parser, _read_schema(base_path).get("args", []))
    # Named outputs (omnibenchmark api >= 0.8): one `--output id=PATH` per stage
    # output. Not in the schema: it repeats, and older runtimes never pass it.
    parser.add_argument("--output", dest="outputs", action="append", default=[],
                        type=_output_pair, metavar="ID=PATH",
                        help="where to write output ID (repeatable)")
    return parser


def _output_pair(s: str) -> tuple[str, Path]:
    oid, sep, path = s.partition("=")
    if not sep or not oid or not path:
        raise argparse.ArgumentTypeError(f"--output wants ID=PATH, got {s!r}")
    return oid, Path(path)


def output_path(args: argparse.Namespace, output_id: str, suffix: str) -> Path:
    """Where to write output ``output_id``: the path the runtime passed with
    ``--output <id>=PATH`` (api >= 0.8), else ``<output_dir>/<name><suffix>``,
    the pre-0.8 convention."""
    for oid, path in getattr(args, "outputs", None) or []:
        if oid == output_id:
            return path
    return Path(args.output_dir) / f"{args.name}{suffix}"


def add_stage_args(parser: argparse.ArgumentParser, schema: str,
                   schema_dir: Path = _SCHEMA_DIR) -> argparse.ArgumentParser:
    """Add the stage's I/O contract from ``schema/<stage-id>.json`` to the
    author's parser. Returns the parser for chaining."""
    stage_path = schema_dir / f"{schema}.json"
    if not stage_path.is_file():
        available = sorted(
            p.stem for p in schema_dir.glob("*.json") if p.stem != _BASE_SCHEMA
        ) if schema_dir.is_dir() else []
        have = f"available: {', '.join(available)}" if available else "no stage schemas vendored"
        raise SystemExit(f"interface '{schema}' not found in {schema_dir} ({have})")
    return _add_args(parser, _read_schema(stage_path).get("args", []))
