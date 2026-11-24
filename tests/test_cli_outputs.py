import argparse
from pathlib import Path

from common import cli


def test_output_flag_does_not_abbreviate_output_dir():
    # Before api 0.8 support, argparse took `--output X` as `--output_dir X`.
    p = cli.add_base_args(argparse.ArgumentParser())
    a = p.parse_args(["--output_dir", "d", "--name", "m",
                      "--output", "neighbors_corrected_h5=/x/m_neighbors.h5"])
    assert str(a.output_dir) == "d"
    assert cli.output_path(a, "neighbors_corrected_h5", "_neighbors.h5") == Path("/x/m_neighbors.h5")
    assert cli.output_path(a, "neighbors_h5", "_neighbors.h5") == Path("d") / "m_neighbors.h5"
