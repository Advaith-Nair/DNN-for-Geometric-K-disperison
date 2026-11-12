"""
Tests for instance generation with --independent-sizes.

Run from the repository root: python -m pytest eval/test_independent_seeds.py -q
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_eval import derive_independent_size_seed, make_points, point_set_fingerprint


def test_seed_is_reproducible():
    """The same inputs always give the same seed."""
    a = derive_independent_size_seed("uniform", 30, 1100)
    b = derive_independent_size_seed("uniform", 30, 1100)
    assert a == b


def test_seed_depends_on_size_replicate_and_distribution():
    """Changing n, the replicate or the distribution changes the seed."""
    base = derive_independent_size_seed("uniform", 30, 1100)
    assert base != derive_independent_size_seed("uniform", 40, 1100)
    assert base != derive_independent_size_seed("uniform", 30, 1101)
    assert base != derive_independent_size_seed("clustered", 30, 1100)


def test_same_seed_gives_same_point_set():
    """The same seed produces identical coordinates."""
    seed = derive_independent_size_seed("uniform", 30, 1100)
    a = make_points("uniform", 30, seed)
    b = make_points("uniform", 30, seed)
    assert point_set_fingerprint(a) == point_set_fingerprint(b)
    assert (a.coords == b.coords).all()


def test_sizes_are_independent_uniform():
    """A uniform instance with n = 30 is not the first 30 points of the instance with n = 40."""
    seed_30 = derive_independent_size_seed("uniform", 30, 1100)
    seed_40 = derive_independent_size_seed("uniform", 40, 1100)
    p30 = make_points("uniform", 30, seed_30)
    p40 = make_points("uniform", 40, seed_40)
    assert not (p30.coords == p40.coords[:30]).all()


def test_sizes_are_independent_clustered():
    """The same holds for clustered instances."""
    seed_30 = derive_independent_size_seed("clustered", 30, 1100)
    seed_40 = derive_independent_size_seed("clustered", 40, 1100)
    p30 = make_points("clustered", 30, seed_30)
    p40 = make_points("clustered", 40, seed_40)
    assert not (p30.coords == p40.coords[:30]).all()