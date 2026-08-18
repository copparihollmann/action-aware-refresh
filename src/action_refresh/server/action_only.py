"""Experiment E — ablate the imagined future and see whether the action cares.

The compute anatomy measured the target precisely: of a 3,188-token sequence,
**2,720 tokens (85.3%) are imagined future video** and only 32 are the predicted
action chunk that the robot actually consumes. The model is matmul-bound (99.4% of
counted FLOPs are `aten.mm`, i.e. per-token linear layers), so token count is very
nearly the cost. If the action does not need the imagination, deleting it is worth
roughly 6.8x fewer tokens — and the visual-caching branch of this project (F/G/H)
would be optimizing something that should simply be removed.

So this module ablates the imagination and measures the action deviation. Two modes,
with deliberately different trade-offs:

``freeze``
    Zero the *velocity* of the noised vision latents on every denoising step, so
    the imagined frames never form and stay at their initial noise. The
    conditioning frame (latent frame 0, the real observation) is untouched, and so
    is the action velocity. Token count and therefore compute are **unchanged**,
    which is the point: this isolates "does the action need the imagination?" from
    "is it faster?".

    Implemented through ``generate_samples_from_batch``'s own
    ``velocity_postprocess_builder`` argument — a supported extension point — so it
    needs **no patch to upstream**. Verified against the pinned source: the hook is
    invoked as ``velocity_postprocess(cond_v_full, noise_x, timestep)`` after the
    conditional forward, and each element of ``cond_v_full`` is one sample's flat
    velocity laid out ``[vision | action | sound]`` where the vision block is a
    flattened ``[C, T, H, W]`` latent.

    Cost note: enabling the hook takes a code path that runs the conditional and
    unconditional forwards sequentially. The RoboLab baseline uses ``guidance=3.0``,
    which already runs both, so the forward count is unchanged — but do not read
    latency from this mode without re-checking that, and never quote it as the
    speedup. The speedup number comes from ``drop``.

``drop``
    Actually remove the imagined vision frames so the sequence shrinks. This is the
    engineering payoff and the only mode whose latency means anything. It is
    genuinely invasive: action tokens are positionally bound one-per-inter-frame-gap
    and share a single full-attention split with vision, so the sequence plan and
    action alignment both have to be rebuilt. Attempted only after ``freeze`` shows
    the deviation is tolerable — building it first would be optimizing before
    knowing whether the optimization is allowed to exist.

**The interpretation caveat, stated up front.** ``freeze`` leaves the vision tokens
as pure noise at every timestep, which is out-of-distribution: late in denoising the
model expects nearly-clean vision. So a large action deviation under ``freeze`` is
ambiguous — it could mean the action needed the imagination's *content*, or merely
that the model was handed inputs unlike anything it trained on. That ambiguity is why
the cheap seed-variation test (E0, no code at all: same observation, different
diffusion seed, therefore different but fully in-distribution imagined futures) is
run alongside it. E0 is assumption-free; ``freeze`` is a stronger intervention with a
weaker interpretation. Neither alone settles the question, and this module does not
pretend otherwise.
"""
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # pragma: no cover
    # torch is only needed for annotations here, and `from __future__ import
    # annotations` keeps them as strings. Importing it at runtime would make this
    # module unimportable in the repo venv (which deliberately has no torch — torch
    # lives in third_party/*/.venv), and the analysis path imports its siblings.
    import torch

FREEZE = "freeze"
DROP = "drop"
VALID_MODES = (FREEZE, DROP)


def _vision_block_shapes(gen_data_clean: Any) -> list[tuple[int, ...]]:
    """Per-sample vision latent shapes `[C, T, H, W]`, used to locate the vision
    block inside each flat velocity tensor."""
    tokens = getattr(gen_data_clean, "x0_tokens_vision", None)
    if not tokens:
        raise RuntimeError(
            "gen_data_clean has no x0_tokens_vision — cannot locate the vision block "
            "in the velocity tensor. Refusing to guess an offset: zeroing the wrong "
            "slice would silently ablate the action instead of the imagination."
        )
    return [tuple(t.shape) for t in tokens]


def _make_freeze_builder(stats: dict[str, Any]):
    """Build the per-step hook that keeps noised vision latents at their initial noise."""

    def builder(**kw: Any):
        gen_data_clean = kw.get("gen_data_clean")
        shapes = _vision_block_shapes(gen_data_clean)
        stats["vision_shapes"] = [list(s) for s in shapes]
        # Recorded so a result can be checked against the token census: the temporal
        # extent here must match the latent frame count the census reports, otherwise
        # we froze the wrong axis.
        stats["latent_frames"] = [int(s[-3]) for s in shapes if len(s) >= 3]
        stats["calls"] = 0

        def hook(
            v_list: list[torch.Tensor], _noise_x: Any, _timestep: Any
        ) -> list[torch.Tensor]:
            stats["calls"] += 1
            out: list[torch.Tensor] = []
            for v, shape in zip(v_list, shapes):
                # Rank-agnostic on purpose: the vision latent is documented as
                # [C,T,H,W] but arrives with a leading batch axis in practice, and
                # hard-coding four dims failed at runtime. The temporal axis is -3 for
                # both [C,T,H,W] and [B,C,T,H,W], so index from the end rather than
                # assuming a rank.
                if len(shape) < 3:
                    raise RuntimeError(
                        f"vision latent shape {shape} has fewer than 3 dims, so the "
                        "temporal axis cannot be located. Refusing to guess."
                    )
                n = 1
                for d in shape:
                    n *= int(d)
                if v.numel() < n:
                    raise RuntimeError(
                        f"velocity tensor has {v.numel()} elements but the vision block "
                        f"alone needs {n} ({shape}). The [vision|action|sound] layout "
                        "assumption is wrong for this config; aborting rather than "
                        "zeroing an arbitrary slice."
                    )
                v = v.clone()
                vision = v[:n].view(*shape)
                # Latent frame 0 is the conditioning frame (the real observation, per
                # `condition_frame_indexes_vision == [0]` in wam mode). Frames 1.. are
                # the imagination — zero their velocity so they never develop.
                index: list[Any] = [slice(None)] * vision.ndim
                index[-3] = slice(1, None)
                vision[tuple(index)] = 0
                v[:n] = vision.reshape(-1)
                out.append(v)
            return out

        return hook

    return builder


@contextlib.contextmanager
def action_only(model: Any, mode: str = FREEZE) -> Iterator[dict[str, Any]]:
    """Run `model.generate_samples_from_batch` with the imagination ablated.

    Wraps the *instance* attribute so the class is untouched and any other model in
    the process is unaffected; the wrapper is removed on exit even if the body raises.
    Yields a stats dict populated during the call (hook invocation count, the vision
    latent shape it acted on) so the caller can assert the intervention actually ran
    rather than trusting that it did.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    if mode == DROP:
        raise NotImplementedError(
            "mode='drop' (E2b: remove imagined vision tokens so the sequence actually "
            "shrinks) is not implemented yet. It requires rebuilding the sequence plan "
            "and the action/vision frame alignment, and is only worth building once "
            "mode='freeze' and the E0 seed-variation result show the imagination can be "
            "ablated without wrecking the action."
        )

    stats: dict[str, Any] = {"mode": mode, "calls": 0}
    original = model.generate_samples_from_batch
    builder = _make_freeze_builder(stats)

    def wrapper(*args: Any, **kwargs: Any):
        if kwargs.get("velocity_postprocess_builder") is not None:
            # Silently replacing a caller's hook would make the measurement a lie
            # about which intervention ran.
            raise RuntimeError(
                "generate_samples_from_batch was already given a "
                "velocity_postprocess_builder; refusing to overwrite it."
            )
        kwargs["velocity_postprocess_builder"] = builder
        return original(*args, **kwargs)

    model.generate_samples_from_batch = wrapper
    try:
        yield stats
    finally:
        try:
            del model.generate_samples_from_batch
        except AttributeError:  # pragma: no cover - defensive
            model.generate_samples_from_batch = original
    if stats["calls"] == 0:
        # The hook never fired, so nothing was ablated and any "action-only" result
        # from this call is actually the baseline. Loud failure beats a silently
        # mislabelled number.
        raise RuntimeError(
            "the vision-freeze hook was never invoked — generate_samples_from_batch "
            "did not use the velocity_postprocess path. The result would be the "
            "unmodified baseline mislabelled as an ablation."
        )
