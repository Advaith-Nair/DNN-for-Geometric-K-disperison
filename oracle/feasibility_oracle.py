"""
The complete feasibility oracle: given a conflict graph and k, answer
"does an independent set of size >= k exist?" -- using the dNN.

This wraps together:
  - dial training (target = -k/2, fix #1)
  - extraction (threshold at alpha)
  - repair (swap-before-delete, fix #2)
  - multi-restart (perturbed re-init on failure, before declaring infeasible)

This is the function binary search will call at every guessed radius x.
"""

from __future__ import annotations

from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.conflict_graph import ConflictGraph

from dnn_core import TrainConfig, train_dials, extract_set, degree_based_init
from repair import repair
from certificates import certified_infeasible


@dataclass
class OracleConfig:
    train_cfg: TrainConfig = None
    alpha: float = 0.5
    use_certificate: bool = False   # if True, first try a cheap PROOF that no size-k
                                    # independent set exists (clique cover bound) and skip
                                    # the network entirely when it succeeds. Only ever
                                    # certifies "infeasible", so it cannot cause wrong answers.
    local_search: bool = True   # 2-improvement local search after repair (paper Sec. 5.1)
    max_restarts: int = 15   # raised from 5: near-threshold instances (where the
                              # answer is exactly k, the hardest case) empirically
                              # needed more attempts, especially mixed with random init

    def __post_init__(self):
        if self.train_cfg is None:
            self.train_cfg = TrainConfig()


@dataclass
class OracleResult:
    feasible: bool
    witness: list[int] | None       # k nodes, pairwise conflict-free, if feasible
    restarts_used: int
    # diagnostics -- useful for instrumenting WHY a call failed (Week 3 tuning)
    per_attempt_final_value: list[float]
    per_attempt_target: float
    per_attempt_converged: list[bool]
    per_attempt_extracted_size: list[int]
    per_attempt_repaired_size: list[int]


def feasibility_oracle(g: ConflictGraph, k: int, cfg: OracleConfig,
                        base_seed: int | None = None) -> OracleResult:
    """
    Answer: does G have an independent set of size >= k?

    Tries up to cfg.max_restarts times, perturbing the dial initialization
    each time, before giving up and declaring infeasible. This is fix #1's
    companion: a single non-convergent run near a hard threshold radius is
    not reliable evidence of true infeasibility.
    """
    if cfg.use_certificate and certified_infeasible(g.adj, k):
        return OracleResult(
            feasible=False, witness=None, restarts_used=0,
            per_attempt_final_value=[], per_attempt_target=-float(k) / 2.0,
            per_attempt_converged=[], per_attempt_extracted_size=[],
            per_attempt_repaired_size=[],
        )

    per_attempt_final_value = []
    per_attempt_converged = []
    per_attempt_extracted_size = []
    per_attempt_repaired_size = []
    target = None

    for attempt in range(cfg.max_restarts):
        seed = None if base_seed is None else base_seed + attempt

        # ALTERNATE init strategies across restarts: degree-based init is a
        # good prior but is nearly deterministic (tiny noise only), so it
        # tends to converge to the SAME local optimum every attempt. Mixing
        # in fully random inits gives real diversity across restarts and
        # empirically finds larger independent sets that degree-based init
        # alone misses. Even-numbered attempts: degree-based. Odd: random.
        init_strategy = "degree" if attempt % 2 == 0 else "random"

        result = train_dials(g, k, cfg.train_cfg, seed=seed, init_strategy=init_strategy)
        target = result.target

        per_attempt_final_value.append(result.final_penalty_value)
        per_attempt_converged.append(result.converged)

        candidate = extract_set(result.theta, alpha=cfg.alpha)
        per_attempt_extracted_size.append(len(candidate))

        repaired = repair(candidate, g, local_search=cfg.local_search)
        per_attempt_repaired_size.append(len(repaired))

        if len(repaired) >= k:
            # trim to exactly k -- any subset of a conflict-free set is
            # still conflict-free
            witness = repaired[:k]
            return OracleResult(
                feasible=True,
                witness=witness,
                restarts_used=attempt + 1,
                per_attempt_final_value=per_attempt_final_value,
                per_attempt_target=target,
                per_attempt_converged=per_attempt_converged,
                per_attempt_extracted_size=per_attempt_extracted_size,
                per_attempt_repaired_size=per_attempt_repaired_size,
            )

    return OracleResult(
        feasible=False,
        witness=None,
        restarts_used=cfg.max_restarts,
        per_attempt_final_value=per_attempt_final_value,
        per_attempt_target=target,
        per_attempt_converged=per_attempt_converged,
        per_attempt_extracted_size=per_attempt_extracted_size,
        per_attempt_repaired_size=per_attempt_repaired_size,
    )
