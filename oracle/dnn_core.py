"""
The dataless neural network (dNN) feasibility oracle core.

This implements the MIS-reduction penalty function from Alkhouri, Atia &
Velasquez (2022), adapted for k-dispersion's DECISION question rather than
their original MAXIMIZATION question.

We train toward −k/2 instead of the paper’s −n/2 because we only need a set of size k.

Base formula (Eq. 7 in the paper), theta in [0,1]^n, one dial per node:

    f(theta) = -sum_v sigma(theta_v - 0.5)  +  n * sum_(u,v) in E  sigma(theta_u + theta_v - 1)

where sigma = ReLU. First term rewards turning dials on (past 0.5), second
term punishes any edge where both endpoints are on, weighted by n so it's
never worth violating an edge to gain one extra node.

Enhanced formula (Eq. 10, using the complement graph) is provided as an
optional alternative loss -- sharper gradient landscape, useful if the base
version is indecisive near the threshold radius.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.conflict_graph import ConflictGraph


def degree_based_init(g: ConflictGraph, seed: int | None = None,
                       noise_scale: float = 0.05) -> torch.Tensor:
    """
    Eq. 12 from the paper: initialize dials HIGH for low-degree nodes (more
    likely to be in an independent set) and LOW for high-degree nodes, plus
    a small random perturbation so repeated restarts don't all start
    identically.

        theta_v = 1 - degree(v)/max_degree + s,   s ~ Uniform(0, noise_scale)
        theta <- theta / max(theta)               (renormalize into [0,1])
    """
    rng = np.random.default_rng(seed)
    degrees = g.degrees()
    max_deg = degrees.max() if degrees.max() > 0 else 1.0

    s = rng.uniform(0.0, noise_scale, size=g.n)
    theta = 1.0 - (degrees / max_deg) + s
    theta = np.clip(theta, 1e-6, None)   # keep strictly positive before renorm
    theta = theta / theta.max()
    theta = np.clip(theta, 0.0, 1.0)

    return torch.tensor(theta, dtype=torch.float32, requires_grad=True)


def random_uniform_init(g: ConflictGraph, seed: int | None = None) -> torch.Tensor:
    """
    Plain uniform-random dial init over [0,1], as an alternative to
    degree_based_init. FINDING FROM TESTING: degree-based init alone is
    prone to converging to a small-but-valid local optimum (it consistently
    settles on the same low-degree nodes every restart, since the init
    itself is deterministic modulo tiny noise). Mixing in fully random
    inits across restart attempts gives much more diverse starting points
    and meaningfully raises the chance of finding a genuine size-k
    independent set near the hardest (threshold) instances.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 1.0, size=g.n)
    return torch.tensor(theta, dtype=torch.float32, requires_grad=True)


def _edges_as_tensors(g: ConflictGraph, device=None):
    """Split edge list into two index tensors for vectorized gather ops."""
    if len(g.edges) == 0:
        u = torch.empty(0, dtype=torch.long, device=device)
        v = torch.empty(0, dtype=torch.long, device=device)
    else:
        arr = np.array(g.edges, dtype=np.int64)
        u = torch.tensor(arr[:, 0], device=device)
        v = torch.tensor(arr[:, 1], device=device)
    return u, v


def _soft_relu(x: torch.Tensor, leak: float = 0.05) -> torch.Tensor:
    """
    A leaky-ReLU used IN PLACE OF the paper's exact ReLU during training.

    BUG FOUND DURING TESTING: torch.relu(x) has EXACTLY ZERO gradient for
    x < 0. Since the reward term is relu(theta_v - 0.5), any dial that
    starts below 0.5 (which, empirically, is MOST dials under degree-based
    init -- see Eq. 12) receives zero gradient from the reward term and can
    never rise, unless the edge-penalty term happens to push it (which has
    the same dead-zone problem). This produces exactly the underconvergence
    symptom: optimization plateaus early and never reaches the target.

    Fix: use a small positive slope below the threshold so every dial keeps
    *some* gradient pressure toward 1, regardless of where it started. This
    does not change where the TRUE minimum sits at convergence (leak is
    small), it only fixes the ability to actually reach that minimum via
    gradient descent.
    """
    return torch.where(x > 0, x, leak * x)


def base_penalty(theta: torch.Tensor, g: ConflictGraph, u_idx: torch.Tensor,
                  v_idx: torch.Tensor, leak: float = 0.05) -> torch.Tensor:
    """
    f(theta), Eq. 7. Base MIS penalty using only G's own edges.

    Uses a leaky-ReLU (see _soft_relu) instead of the paper's exact ReLU
    during optimization, to avoid the dead-gradient trap described above.
    """
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
    """
    h(theta), Eq. 10. Adds a second penalty term over the COMPLEMENT graph's
    edges (pairs that are NOT connected in G). Sharper landscape: minimum
    value scales as -k^2/2 instead of -k/2 (Theorem 4.2). Use this if the
    base penalty is indecisive near the threshold radius.

    NOTE: complement graph has O(n^2) edges for sparse G, so this is more
    expensive -- only worth it if base_penalty is actually struggling.
    """
    f_val = base_penalty(theta, g, u_idx, v_idx, leak=leak)
    if comp_u_idx.numel() > 0:
        comp_violation = _soft_relu(theta[comp_u_idx] + theta[comp_v_idx] - 1.0, leak=leak)
        comp_penalty = -torch.sum(comp_violation)
    else:
        comp_penalty = torch.tensor(0.0, device=theta.device)
    return f_val + comp_penalty


def _complement_edges_as_tensors(g: ConflictGraph, device=None):
    """Build edge-index tensors for the complement graph (non-edges of G)."""
    n = g.n
    iu, ju = np.triu_indices(n, k=1)
    is_edge = g.adj[iu, ju]
    comp_mask = ~is_edge
    comp_u = torch.tensor(iu[comp_mask], dtype=torch.long, device=device)
    comp_v = torch.tensor(ju[comp_mask], dtype=torch.long, device=device)
    return comp_u, comp_v


@dataclass
class TrainConfig:
    target_mode: str = "k"       # "k" -> base penalty, target -k/2
                                   # "k2" -> enhanced penalty, target -k^2/2
    lr: float = 0.1                # matches paper's ADAM initial LR
    max_steps: int = 500
    convergence_tol: float = 1e-3   # stop early if loss below this
    device: str = "cpu"
    leak_start: float = 0.15       # leaky-relu slope at step 0 (escape dead gradients)
    leak_end: float = 0.01         # leaky-relu slope at final step (near-exact ReLU,
                                     # so the loss value genuinely reflects a near-binary
                                     # theta and can't be gamed by half-on dials)
    binarize_gamma_max: float = 2.0  # FINDING FROM TESTING: without this, the optimizer
                                     # can hit the target loss value using "phantom"
                                     # fractional dials (many small values near 0 that sum
                                     # up via the leak) instead of a genuine near-binary
                                     # solution. This term, theta*(1-theta), is zero at
                                     # theta=0 or 1 and positive in between -- explicitly
                                     # pushing dials toward true binary states. Ramped
                                     # from 0 -> binarize_gamma_max across training so
                                     # early steps can still explore freely.


@dataclass
class TrainResult:
    theta: torch.Tensor
    final_penalty_value: float
    target: float
    steps_run: int
    converged: bool   # True if final_penalty_value reached target within tol


def train_dials(g: ConflictGraph, k: int, cfg: TrainConfig,
                 init_theta: torch.Tensor | None = None,
                 seed: int | None = None,
                 init_strategy: str = "degree") -> TrainResult:
    """
    Run ADAM to push theta toward the target value that corresponds to
    "an independent set of size k exists" (THIS IS FIX #1: target is a
    function of k, not n).

    init_strategy: "degree" uses Eq. 12 (degree-based + small noise),
    "random" uses plain uniform random init. See random_uniform_init's
    docstring for why alternating strategies across restarts matters.
    """
    device = torch.device(cfg.device)

    if init_theta is not None:
        theta = init_theta.clone().detach().to(device).requires_grad_(True)
    elif init_strategy == "random":
        theta = random_uniform_init(g, seed=seed).to(device).requires_grad_(True)
    else:
        theta = degree_based_init(g, seed=seed).to(device).requires_grad_(True)

    u_idx, v_idx = _edges_as_tensors(g, device=device)

    if cfg.target_mode == "k":
        target = -float(k) / 2.0
        loss_fn = lambda th, leak: base_penalty(th, g, u_idx, v_idx, leak=leak)
    elif cfg.target_mode == "n":
        # paper's original choice (Sec. 4.1): aim for the LARGEST independent
        # set (f_d = -n/2); the oracle then just checks whether it has >= k nodes
        target = -float(g.n) / 2.0
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
        # anneal leak linearly from leak_start -> leak_end across training.
        # Early steps: strong leak, so dials below 0.5 still get gradient
        # signal and can escape the dead zone. Late steps: leak -> ~0, so
        # the loss value can no longer be satisfied by "phantom" partial
        # activations -- it has to correspond to a genuinely near-binary
        # theta, restoring the theorem's exact correspondence in practice.
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
            theta.clamp_(0.0, 1.0)   # keep dials in [0,1] (box constraint)

        final_val = val.item()
        steps_run = step + 1

        if abs(final_val - target) < cfg.convergence_tol:
            break

    if cfg.target_mode == "n":
        # -n/2 is only reachable on an edgeless graph, so "converged" means
        # "reached the -k/2 level needed for a size-k set"
        converged = final_val is not None and final_val <= -float(k) / 2.0 + cfg.convergence_tol
    else:
        converged = final_val is not None and abs(final_val - target) < cfg.convergence_tol
    return TrainResult(
        theta=theta.detach(),
        final_penalty_value=final_val,
        target=target,
        steps_run=steps_run,
        converged=converged,
    )


def extract_set(theta: torch.Tensor, alpha: float = 0.5) -> list[int]:
    """Eq. 4: threshold dials >= alpha into a candidate node set."""
    theta_np = theta.detach().cpu().numpy()
    return [i for i, val in enumerate(theta_np) if val >= alpha]
