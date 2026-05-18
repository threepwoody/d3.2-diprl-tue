"""
Logging utilities for PPO training metrics.

This module contains helper functions for logging programmatic policy metrics
and gradient norms to the training logger.
"""

import torch as th


def compute_and_debug_gradient_norms(
    policy, device, loss, is_last_batch, epoch, n_epochs
):
    """
    Compute gradient norms for programmatic policy components.

    Only computes on the last mini-batch of the last epoch to reduce GPU sync overhead.
    Also calls debug functions if gradients are unexpectedly zero.

    Args:
        policy: The policy network
        device: Device where tensors are located
        loss: Current loss value
        is_last_batch: Whether this is the last batch in the epoch
        epoch: Current epoch number
        n_epochs: Total number of epochs

    Returns:
        tuple: (search_map_grad_norm, fusion_programs_grad_norm)
    """

    if not (is_last_batch and epoch == n_epochs - 1):

        return 0.0, 0.0

    if not (
        hasattr(policy, "programmatic_policy")
        and policy.programmatic_policy is not None
    ):

        return 0.0, 0.0

    if not hasattr(policy.programmatic_policy, "search_map"):

        return 0.0, 0.0

    with th.no_grad():

        search_map_grad_norm_sq = th.tensor(0.0, device=device)

        for p in policy.programmatic_policy.search_map.parameters():

            if p.grad is not None:

                search_map_grad_norm_sq += (p.grad**2).sum()

        prog_search_grad_norm = search_map_grad_norm_sq.sqrt().item()

        fusion_grad_norm_sq = th.tensor(0.0, device=device)

        for p in policy.programmatic_policy.fusion_programs.parameters():

            if p.grad is not None:

                fusion_grad_norm_sq += (p.grad**2).sum()

        prog_fusion_grad_norm = fusion_grad_norm_sq.sqrt().item()

    return prog_search_grad_norm, prog_fusion_grad_norm


def log_programmatic_policy_metrics(
    logger, policy, device, prog_current_phase, prog_phase_counter
):
    """
    Log programmatic policy metrics to the training logger.

    Logs:
    - Current phase ID (0=architecture search, 1=program optimization)
    - Phase counter (iterations in current phase)
    - Program depth probabilities (distribution over different program depths)
    - Trainable parameter counts for search_map and fusion_programs

    Args:
        logger: The training logger
        policy: The policy network
        device: Device where tensors are located
        prog_current_phase: Current phase ID
        prog_phase_counter: Iterations in current phase
    """

    if not (
        hasattr(policy, "programmatic_policy")
        and policy.programmatic_policy is not None
    ):

        return

    if not hasattr(policy.programmatic_policy, "search_map"):

        return

    logger.record("train/prog_phase_id", prog_current_phase)

    logger.record("train/prog_phase_counter", prog_phase_counter)

    with th.no_grad():

        search_map = policy.programmatic_policy.search_map

        if search_map.type == "architecture":

            max_depth = policy.prog_graph_depth - 1

            v = th.ones(max_depth, device=device)

            for i in range(len(v)):

                if i == 0:

                    v[i] = search_map.options[0].softmax(dim=0)[0]

                else:

                    prev = 1

                    for j in range(i):

                        prev *= search_map.options[j].softmax(dim=0)[1]

                    if i == len(v) - 1:

                        v[i] = prev

                    else:

                        v[i] = prev * search_map.options[i].softmax(dim=0)[0]

            for depth_idx, prob in enumerate(v):

                logger.record(f"train/prog_depth_{depth_idx+1}_prob", prob.item())

    prog_search_map_params = sum(
        p.numel()
        for p in policy.programmatic_policy.search_map.parameters()
        if p.requires_grad
    )

    prog_fusion_params = sum(
        p.numel()
        for p in policy.programmatic_policy.fusion_programs.parameters()
        if p.requires_grad
    )

    logger.record("train/prog_search_map_trainable", prog_search_map_params)

    logger.record("train/prog_fusion_trainable", prog_fusion_params)


def log_gradient_norms(logger, search_grad_norm, fusion_grad_norm):
    """
    Log gradient norms for programmatic policy components.

    Args:
        logger: The training logger
        search_grad_norm: Gradient norm for search_map parameters
        fusion_grad_norm: Gradient norm for fusion_programs parameters
    """

    logger.record("train/prog_search_map_grad_norm", search_grad_norm)

    logger.record("train/prog_fusion_grad_norm", fusion_grad_norm)
