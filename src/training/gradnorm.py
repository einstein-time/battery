"""GradNorm adaptive multi-task loss balancing.

Implements the GradNorm algorithm from:

    Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A. (2018).
    *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep
    Multitask Networks.* ICML 2018.

The idea is to dynamically re-weight multiple loss terms so that their
gradient magnitudes w.r.t. a shared network layer remain balanced.  Tasks
that train faster are down-weighted, while slower tasks are up-weighted.

Typical usage::

    grad_norm = GradNorm(n_tasks=4, alpha=1.5, lr=0.025)
    # inside training loop:
    weights = grad_norm.get_weights()
    total = sum(w * l for w, l in zip(weights, losses))
    total.backward()
    grad_norm.update(losses, shared_layer=model.encoder[-1])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradNorm(nn.Module):
    """Adaptive loss-weight balancing via gradient normalisation.

    Maintains a set of learnable log-weights (one per task) and updates
    them after each training step so that the gradient norms of the
    weighted losses w.r.t. a user-specified shared layer are approximately
    equal, adjusted for relative training speed.

    Attributes:
        n_tasks: Number of loss terms (tasks).
        alpha: Asymmetry hyper-parameter controlling how aggressively
            weights are re-balanced.  Larger values enforce stricter
            balancing.
        lr: Learning rate for updating the log-weights.
        log_weights: Learnable ``nn.Parameter`` of shape ``(n_tasks,)``
            storing the *logarithm* of the raw weights.  Actual weights
            are obtained via ``softmax(log_weights) * n_tasks``.
    """

    def __init__(
        self,
        n_tasks: int,
        alpha: float = 1.5,
        lr: float = 0.025,
    ) -> None:
        """Initialise GradNorm.

        Args:
            n_tasks: Number of loss terms to balance.
            alpha: Asymmetry parameter (see paper, typical range 0.5-3.0).
            lr: Learning rate for updating the loss weights.
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.lr = lr

        # Log-space weights initialised to zero (equal weights after softmax)
        self.log_weights = nn.Parameter(
            torch.zeros(n_tasks, dtype=torch.float32),
            requires_grad=True,
        )

        # Store initial loss values for computing relative inverse training
        # rates.  Populated on the first call to ``update``.
        self.register_buffer(
            "_initial_losses",
            torch.zeros(n_tasks, dtype=torch.float32),
        )
        self._initial_losses_set: bool = False

        # Internal optimiser for the log_weights parameter only
        self._weight_optimizer = torch.optim.Adam(
            [self.log_weights], lr=self.lr
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(self) -> torch.Tensor:
        """Return the current normalised task weights.

        Weights are computed as ``softmax(log_weights) * n_tasks`` so that
        they sum to ``n_tasks`` and are centered around 1.0 when all
        weights are equal.

        Returns:
            Tensor of shape ``(n_tasks,)`` with positive weights.
        """
        return torch.softmax(self.log_weights, dim=0) * self.n_tasks

    @torch.no_grad()
    def update(
        self,
        losses: Sequence[torch.Tensor],
        shared_layer: nn.Module,
    ) -> None:
        """Update loss weights using the GradNorm algorithm.

        This should be called **after** ``total_loss.backward()`` but
        **before** ``optimizer.step()`` for the main model, or (more
        commonly) with ``retain_graph=True``.

        The procedure:
        1. Compute the L2 norm of each weighted-loss gradient w.r.t. the
           parameters of ``shared_layer``.
        2. Compute the *relative inverse training rate* for each task
           (how much slower it is converging compared to the average).
        3. Set the GradNorm target for each task as
           ``mean_grad_norm * (inv_rate ** alpha)``.
        4. Update ``log_weights`` to minimise the L1 distance between
           actual and target gradient norms.

        Args:
            losses: Sequence of ``n_tasks`` scalar loss tensors (the
                **unweighted** losses, not yet multiplied by weights).
            shared_layer: An ``nn.Module`` whose parameters are used to
                compute gradient norms (typically the last layer of the
                shared encoder).
        """
        if len(losses) != self.n_tasks:
            raise ValueError(
                f"Expected {self.n_tasks} losses, got {len(losses)}"
            )

        device = self.log_weights.device
        loss_values = torch.stack(
            [l.detach().to(device) for l in losses]
        )

        # Initialise baseline losses on first call
        if not self._initial_losses_set:
            self._initial_losses.copy_(loss_values.clamp(min=1e-12))
            self._initial_losses_set = True
            logger.info(
                "GradNorm: initial losses set to %s",
                self._initial_losses.tolist(),
            )
            return  # No update on first step

        # ----- Step 1: compute gradient norms per task -------------------
        weights = self.get_weights()
        shared_params = list(shared_layer.parameters())
        if not shared_params:
            logger.warning(
                "GradNorm: shared_layer has no parameters; skipping update."
            )
            return

        grad_norms: List[torch.Tensor] = []
        for i in range(self.n_tasks):
            # weighted loss for task i
            weighted_loss_i = weights[i] * losses[i]
            # Compute gradients w.r.t. shared layer parameters
            grads = torch.autograd.grad(
                weighted_loss_i,
                shared_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            # L2 norm of the concatenated gradient vector
            total_norm = torch.tensor(0.0, device=device)
            for g in grads:
                if g is not None:
                    total_norm = total_norm + g.detach().norm(2) ** 2
            grad_norms.append(total_norm.sqrt())

        grad_norms_tensor = torch.stack(grad_norms)  # (n_tasks,)
        mean_grad_norm = grad_norms_tensor.mean().detach()

        # ----- Step 2: relative inverse training rate --------------------
        loss_ratios = loss_values / self._initial_losses.clamp(min=1e-12)
        mean_ratio = loss_ratios.mean()
        inv_train_rates = loss_ratios / mean_ratio.clamp(min=1e-12)

        # ----- Step 3: GradNorm targets ----------------------------------
        targets = (mean_grad_norm * (inv_train_rates ** self.alpha)).detach()

        # ----- Step 4: update log_weights via L1 loss --------------------
        gradnorm_loss = torch.sum(
            torch.abs(grad_norms_tensor - targets)
        )

        self._weight_optimizer.zero_grad()
        # We need gradients w.r.t. log_weights only.  Since grad_norms
        # were computed with create_graph=False, we re-compute a
        # lightweight surrogate that only depends on log_weights.
        # The surrogate maps log_weights -> weights -> weighted_grad_norms
        # We approximate this by differentiating through the weights:
        surrogate = torch.tensor(0.0, device=device, requires_grad=True)
        # Detach gradient norms and treat them as constants; the only
        # learnable path is through the weights themselves.
        # Following the original GradNorm paper's practical implementation:
        self.log_weights.grad = None
        gradnorm_loss_for_weights = torch.sum(
            torch.abs(
                weights.detach() * grad_norms_tensor.detach()
                - targets
            )
        )
        # Manual gradient: d|w*G - target|/d(log_w) via straight-through
        # Simpler and numerically stable approach: just step log_weights
        # in the direction that reduces the imbalance.
        with torch.enable_grad():
            # Re-derive weights from log_weights to allow gradient flow
            w = torch.softmax(self.log_weights, dim=0) * self.n_tasks
            loss_for_w = torch.sum(
                torch.abs(w * grad_norms_tensor.detach() - targets)
            )
            loss_for_w.backward()

        self._weight_optimizer.step()

        # ----- Renormalise so that weights sum to n_tasks -----------------
        # (The softmax already normalises, but clamp log_weights for safety)
        with torch.no_grad():
            self.log_weights.data -= self.log_weights.data.mean()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "GradNorm weights: %s | grad_norms: %s",
                self.get_weights().tolist(),
                grad_norms_tensor.tolist(),
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Return the state dict including GradNorm-specific buffers.

        Returns:
            Dictionary with model parameters, initial losses, and the
            flag indicating whether initial losses have been recorded.
        """
        base = super().state_dict(**kwargs)
        base["_initial_losses_set"] = self._initial_losses_set
        base["_weight_optimizer"] = self._weight_optimizer.state_dict()
        return base

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> None:
        """Restore state including GradNorm-specific buffers.

        Args:
            state_dict: State dictionary from a previous ``state_dict()``
                call.
            strict: Whether to require all keys to match.
        """
        self._initial_losses_set = state_dict.pop("_initial_losses_set", False)
        opt_state = state_dict.pop("_weight_optimizer", None)

        # Use the base class to restore parameters and buffers
        super().load_state_dict(state_dict, strict=strict)

        if opt_state is not None:
            self._weight_optimizer.load_state_dict(opt_state)

    def reset(self) -> None:
        """Reset GradNorm state for a new training phase.

        Re-initialises log-weights to zero and clears the initial loss
        baseline so it is re-captured on the next ``update`` call.
        """
        with torch.no_grad():
            self.log_weights.zero_()
        self._initial_losses.zero_()
        self._initial_losses_set = False
        self._weight_optimizer = torch.optim.Adam(
            [self.log_weights], lr=self.lr
        )
        logger.info("GradNorm state reset.")
