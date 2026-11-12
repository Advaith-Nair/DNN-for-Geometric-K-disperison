"""
Dataless neural network (dNN) for the independent-set decision problem.

Implements the differentiable formulation of the maximum independent set
problem of Alkhouri, Atia and Velasquez, "A differentiable approach to the
maximum independent set problem using dataless neural networks", Neural
Networks 155 (2022). Each vertex v has a parameter theta_v in [0, 1]. The
base loss (Eq. 7 of the paper) is

    f(theta) = - sum_v  sigma(theta_v - 1/2)
               + n * sum_{(u,v) in E}  sigma(theta_u + theta_v - 1),

where sigma is the ReLU. The first term rewards selected vertices and the
second penalises adjacent pairs that are both selected. An independent set
of size k corresponds to f = -k/2.

Differences from the paper:

* The optimisation target is -k/2 instead of -n/2, since the decision
  problem only asks for a set of size k.
* A leaky ReLU is used during optimisation, with a slope that is annealed
  towards zero.
* A term theta * (1 - theta), with a weight that increases during training,
  pushes the parameters towards 0 or 1.
* Restarts alternate between degree-based (Eq. 12) and uniform random
  initialisation (see feasibility_oracle.py).

The enhanced loss of the paper (Eq. 10), which also uses the edges of the
complement graph, is available as target_mode="k2".
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.conflict_graph import ConflictGraph


def degree_based_init(g: ConflictGraph, seed: int | None = None,
                       noise_scale: float = 0.05) -> torch.Tensor:
    """Degree-based initialisation (Eq. 12 of the paper).

    Low-degree vertices start with large values and high-degree vertices
    with small values. Uniform noise in [0, noise_scale] is added so that
    restarts differ, and the result is rescaled to [0, 1]:

        theta_v = 1 - deg(v) / max_deg + s_v,   theta <- theta / max(theta).
    """
    rng = np.random.default_rng(seed)
    degrees = g.degrees()
    max_deg = degrees.max() if degrees.max() > 0 else 1.0

    s = rng.uniform(0.0, noise_scale, size=g.n)
    theta = 1.0 - (degrees / max_deg) + s
    theta = np.clip(theta, 1e-6, None)
    theta = theta / theta.max()
    theta = np.clip(theta, 0.0, 1.0)

    return torch.tensor(theta, dtype=torch.float32, requires_grad=True)


def random_uniform_init(g: ConflictGraph, seed: int | None = None) -> torch.Tensor:
    """Initialise every parameter uniformly at random in [0, 1]."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 1.0, size=g.n)
    return torch.tensor(theta, dtype=torch.float32, requires_grad=True)


def _edges_as_tensors(g: ConflictGraph, device=None):
    """Return the edge endpoints as two index tensors."""
    if len(g.edges) == 0:
        u = torch.empty(0, dtype=torch.long, device=device)
        v = torch.empty(0, dtype=torch.long, device=device)
    else:
        arr = np.array(g.edges, dtype=np.int64)
        u = torch.tensor(arr[:, 0], device=device)
        v = torch.tensor(arr[:, 1], device=device)
    return u, v


def _soft_relu(x: torch.Tensor, leak: float = 0.05) -> torch.Tensor:
    """Leaky ReLU, used in place of the ReLU of the paper.

    With a plain ReLU, a parameter below 1/2 receives no gradient from the
    reward term and cannot increase. The small slope for negative inputs
    keeps a gradient on every parameter.
    """
    return torch.where(x > 0, x, leak * x)


def base_penalty(theta: torch.Tensor, g: ConflictGraph, u_idx: torch.Tensor,
                  v_idx: torch.Tensor, leak: float = 0.05) -> torch.Tensor:
    """Base loss f(theta) (Eq. 7 of the paper), with a leaky ReLU."""
    n = g.n
    reward = -torch.sum(_soft_relu(theta - 0.5, leak=leak))
    if u_idx.numel() > 0:
        edge_violation = _soft_relu(theta[u_idx] + theta[v_idx] - 1.0, leak=leak)
        penalty = float(n) * torch.sum(edge_violation)
    else:
        penalty = torch.tensor(0.0, device=theta.device)
    return reward + penalty


def enhanced_penalty(theta: torch.Tensor, g: ConflictGraph, u_idx: torch.Tensor,
                      v_idx: torch.Tensor, comp_u_idx: torch.Tensor,
                      comp_v_idx: torch.Tensor, leak: float = 0.05) -> torch.Tensor:
    """Enhanced loss h(theta) (Eq. 10 of the paper).

    Adds a reward for every pair of non-adjacent vertices that are both
    selected, so that an independent set of size k has value -k^2/2
    (Theorem 4.2 of the paper). The complement graph can have O(n^2)
    edges, which makes this loss more expensive than f.
    """
    f_val = base_penalty(theta, g, u_idx, v_idx, leak=leak)
    if comp_u_idx.numel() > 0:
        comp_violation = _soft_relu(theta[comp_u_idx] + theta[comp_v_idx] - 1.0, leak=leak)
        comp_penalty = -torch.sum(comp_violation)
    else:
        comp_penalty = torch.tensor(0.0, device=theta.device)
    return f_val + comp_penalty


def _complement_edges_as_tensors(g: ConflictGraph, device=None):
    """Return the edges of the complement graph as two index tensors."""
    n = g.n
    iu, ju = np.triu_indices(n, k=1)
    is_edge = g.adj[iu, ju]
    comp_mask = ~is_edge
    comp_u = torch.tensor(iu[comp_mask], dtype=torch.long, device=device)
    comp_v = torch.tensor(ju[comp_mask], dtype=torch.long, device=device)
    return comp_u, comp_v


@dataclass
class TrainConfig:
    """Optimisation settings for one run of the dNN.

    Attributes:
        target_mode: "k" (loss f, target -k/2) or "k2" (loss h, target -k^2/2).
        lr: Adam learning rate.
        max_steps: Maximum number of Adam steps.
        convergence_tol: Stop when |loss value - target| is below this.
        device: Torch device.
        leak_start, leak_end: Slope of the leaky ReLU at the first and last
            step; the slope is interpolated linearly in between.
        binarize_gamma_max: Final weight of the term sum theta * (1 - theta),
            which increases linearly from 0.
    """

    target_mode: str = "k"
    lr: float = 0.1
    max_steps: int = 500
    convergence_tol: float = 1e-3
    device: str = "cpu"
    leak_start: float = 0.15
    leak_end: float = 0.01
    binarize_gamma_max: float = 2.0


@dataclass
class TrainResult:
    """Outcome of one optimisation run.

    Attributes:
        theta: Final parameters.
        final_penalty_value: Loss value at the last step.
        target: Target value used.
        steps_run: Number of Adam steps performed.
        converged: Whether the target was reached within the tolerance.
    """

    theta: torch.Tensor
    final_penalty_value: float
    target: float
    steps_run: int
    converged: bool


def train_dials(g: ConflictGraph, k: int, cfg: TrainConfig,
                 seed: int | None = None,
                 init_strategy: str = "degree") -> TrainResult:
    """Optimise theta with Adam for the decision question "is alpha(G) >= k?".

    The objective is (loss(theta) - target)^2 + gamma * sum theta(1 - theta),
    and theta is projected onto [0, 1] after every step.

    Args:
        g: Conflict graph.
        k: Required independent-set size.
        cfg: Optimisation settings.
        seed: Seed for the initialisation.
        init_strategy: "degree" (Eq. 12) or "random".
    """
    device = torch.device(cfg.device)

    if init_strategy == "random":
        theta = random_uniform_init(g, seed=seed).to(device).requires_grad_(True)
    else:
        theta = degree_based_init(g, seed=seed).to(device).requires_grad_(True)

    u_idx, v_idx = _edges_as_tensors(g, device=device)

    if cfg.target_mode == "k":
        target = -float(k) / 2.0
        loss_fn = lambda th, leak: base_penalty(th, g, u_idx, v_idx, leak=leak)
    elif cfg.target_mode == "k2":
        comp_u, comp_v = _complement_edges_as_tensors(g, device=device)
        target = -float(k) ** 2 / 2.0
        loss_fn = lambda th, leak: enhanced_penalty(th, g, u_idx, v_idx, comp_u, comp_v, leak=leak)
    else:
        raise ValueError(f"unknown target_mode: {cfg.target_mode}")

    optimizer = torch.optim.Adam([theta], lr=cfg.lr)

    final_val = None
    steps_run = 0
    for step in range(cfg.max_steps):
        # Linear schedules: the leak decreases and the binarisation weight
        # increases over the course of training.
        frac = step / max(cfg.max_steps - 1, 1)
        leak = cfg.leak_start + frac * (cfg.leak_end - cfg.leak_start)
        gamma = cfg.binarize_gamma_max * frac

        optimizer.zero_grad()
        val = loss_fn(theta, leak)
        binarize_penalty = torch.sum(theta * (1.0 - theta))
        loss = (val - target) ** 2 + gamma * binarize_penalty
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            theta.clamp_(0.0, 1.0)

        final_val = val.item()
        steps_run = step + 1

        if abs(final_val - target) < cfg.convergence_tol:
            break

    converged = final_val is not None and abs(final_val - target) < cfg.convergence_tol
    return TrainResult(
        theta=theta.detach(),
        final_penalty_value=final_val,
        target=target,
        steps_run=steps_run,
        converged=converged,
    )


def extract_set(theta: torch.Tensor, alpha: float = 0.5) -> list[int]:
    """Return the vertices with theta_v >= alpha (Eq. 4 of the paper)."""
    theta_np = theta.detach().cpu().numpy()
    return [i for i, val in enumerate(theta_np) if val >= alpha]