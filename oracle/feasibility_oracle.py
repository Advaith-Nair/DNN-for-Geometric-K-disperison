"""
Feasibility oracle: decide whether a conflict graph has an independent set
of size at least k.

Each attempt optimises the dNN parameters, rounds them to a vertex set and
post-processes the set (conflict repair, greedy augmentation and, optionally,
2-improvement local search). If no attempt produces a set of size k within
the restart budget, the instance is reported infeasible. The answer
"feasible" is always backed by a verified independent set; the answer
"infeasible" may be wrong unless it comes from the clique-cover certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.conflict_graph import ConflictGraph

from dnn_core import TrainConfig, train_dials, extract_set
from repair import repair
from certificates import certified_infeasible


@dataclass
class OracleConfig:
    """Settings of the feasibility oracle.

    Attributes:
        train_cfg: Optimisation settings for each attempt.
        alpha: Rounding threshold for theta.
        use_certificate: If True, first check the clique-cover bound and
            report infeasibility without running the network when the bound
            is below k.
        local_search: Apply 2-improvement local search after repair.
        max_restarts: Maximum number of attempts.
    """

    train_cfg: TrainConfig = None
    alpha: float = 0.5
    use_certificate: bool = False
    local_search: bool = True
    max_restarts: int = 15

    def __post_init__(self):
        if self.train_cfg is None:
            self.train_cfg = TrainConfig()


@dataclass
class OracleResult:
    """Answer of the oracle.

    Attributes:
        feasible: Whether an independent set of size k was found.
        witness: The k vertices found, or None.
        restarts_used: Number of attempts made.
    """

    feasible: bool
    witness: list[int] | None
    restarts_used: int


def feasibility_oracle(g: ConflictGraph, k: int, cfg: OracleConfig,
                        base_seed: int | None = None) -> OracleResult:
    """Decide whether g has an independent set of size at least k.

    Attempt i uses seed base_seed + i. Even-numbered attempts use the
    degree-based initialisation and odd-numbered attempts a uniform random
    one, so that the restarts start from different regions of [0, 1]^n.

    Args:
        g: Conflict graph.
        k: Required set size.
        cfg: Oracle settings.
        base_seed: Base seed for the attempts; None for unseeded runs.
    """
    if cfg.use_certificate and certified_infeasible(g.adj, k):
        return OracleResult(feasible=False, witness=None, restarts_used=0)

    for attempt in range(cfg.max_restarts):
        seed = None if base_seed is None else base_seed + attempt
        init_strategy = "degree" if attempt % 2 == 0 else "random"

        result = train_dials(g, k, cfg.train_cfg, seed=seed, init_strategy=init_strategy)
        candidate = extract_set(result.theta, alpha=cfg.alpha)
        repaired = repair(candidate, g, local_search=cfg.local_search)

        if len(repaired) >= k:
            # Any k vertices of an independent set are independent.
            return OracleResult(feasible=True, witness=repaired[:k], restarts_used=attempt + 1)

    return OracleResult(feasible=False, witness=None, restarts_used=cfg.max_restarts)