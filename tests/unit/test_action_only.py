"""Tests for the Experiment E vision ablation.

No GPU and no real model: these check the wiring, which is where a silent error
would be most damaging. The failure mode that matters is an ablation that *reports*
success while actually running the unmodified baseline — that would produce a
headline result ("actions do not need imagination!") from a run where nothing was
ablated. Hence the hard requirement that the hook must fire.
"""
from __future__ import annotations

import numpy as np
import pytest

# These tests build real torch tensors to exercise the [vision|action] slicing, so
# they only run where torch exists — i.e. `make test-model`, which uses the
# cosmos-framework venv. The repo venv has no torch on purpose. Skipping here keeps
# `make test` meaningful in the repo venv instead of failing on an absent dependency.
torch = pytest.importorskip("torch", reason="run via `make test-model` (cosmos venv)")

from action_refresh.server.action_only import DROP, FREEZE, action_only  # noqa: E402


class FakeGenData:
    """Stands in for GenerationDataClean: only the vision latent shape is read."""

    def __init__(self, shape=(4, 9, 17, 20)):
        self.x0_tokens_vision = [torch.zeros(shape)]


class FakeModel:
    """Minimal stand-in for the real model.

    `generate_samples_from_batch` calls the injected builder and hook the same way
    the pinned upstream source does: builder(**state) once, then the returned hook as
    `hook(v_list, noise_x, timestep)` per denoising step.
    """

    def __init__(self, vision_shape=(4, 9, 17, 20), action_len=33, action_dim=8, steps=4):
        self.vision_shape = vision_shape
        self.n_vision = int(np.prod(vision_shape))
        self.n_action = action_len * action_dim
        self.steps = steps
        self.captured: list[torch.Tensor] = []
        self.calls = 0

    def generate_samples_from_batch(self, data_batch=None, **kwargs):
        self.calls += 1
        builder = kwargs.get("velocity_postprocess_builder")
        # Velocity layout is [vision | action]; all-ones so any zeroing is visible.
        v = torch.ones(self.n_vision + self.n_action)
        if builder is not None:
            hook = builder(
                model=self,
                net=None,
                cond_tokens=None,
                sequence_plans=None,
                gen_data_clean=FakeGenData(self.vision_shape),
            )
            for step in range(self.steps):
                (v,) = hook([v.clone()], None, float(step))
                self.captured.append(v.clone())
        return {"action": [torch.zeros(33, 8)], "vision": [torch.zeros(self.vision_shape)]}


def test_freeze_zeros_only_the_imagined_vision_frames():
    """Conditioning frame 0 and the whole action block must be untouched.

    Getting the slice wrong is the dangerous bug: zeroing the action block would
    ablate the deliverable instead of the imagination, and the resulting 'action-only'
    number would be meaningless.
    """
    model = FakeModel()
    with action_only(model, mode=FREEZE) as stats:
        model.generate_samples_from_batch()
    v = model.captured[-1]
    c, t, h, w = model.vision_shape
    vision = v[: model.n_vision].view(c, t, h, w)
    action = v[model.n_vision :]

    assert torch.all(vision[:, 0] == 1.0), "conditioning frame 0 must be preserved"
    assert torch.all(vision[:, 1:] == 0.0), "imagined frames 1.. must be zeroed"
    assert torch.all(action == 1.0), "the action block must never be touched"
    assert stats["calls"] == model.steps
    assert stats["vision_shapes"] == [list(model.vision_shape)]


@pytest.mark.parametrize(
    "vision_shape",
    [
        (4, 9, 17, 20),  # documented [C,T,H,W]
        (1, 4, 9, 17, 20),  # what actually arrives: leading batch axis
        (9, 17, 20),  # bare [T,H,W]
    ],
)
def test_freeze_locates_the_temporal_axis_at_any_rank(vision_shape):
    """The rank assumption broke at runtime, so pin the behaviour for each rank.

    The temporal axis is -3 for all of these; hard-coding four dims raised
    'too many values to unpack' against the real 5-D latent.
    """
    model = FakeModel(vision_shape=vision_shape)
    with action_only(model, mode=FREEZE) as stats:
        model.generate_samples_from_batch()
    v = model.captured[-1]
    vision = v[: model.n_vision].view(*vision_shape)
    # Frame 0 preserved, frames 1.. zeroed, along axis -3 whatever the rank.
    keep = [slice(None)] * vision.ndim
    keep[-3] = 0
    drop = [slice(None)] * vision.ndim
    drop[-3] = slice(1, None)
    assert torch.all(vision[tuple(keep)] == 1.0)
    assert torch.all(vision[tuple(drop)] == 0.0)
    assert torch.all(v[model.n_vision :] == 1.0), "action block untouched"
    assert stats["latent_frames"] == [vision_shape[-3]]


def test_freeze_refuses_a_shape_with_no_temporal_axis():
    model = FakeModel(vision_shape=(17, 20))
    with pytest.raises(RuntimeError, match="fewer than 3 dims"):
        with action_only(model, mode=FREEZE):
            model.generate_samples_from_batch()


def test_hook_is_removed_afterwards_and_class_untouched():
    model = FakeModel()
    original = FakeModel.generate_samples_from_batch
    with action_only(model, mode=FREEZE):
        model.generate_samples_from_batch()
        assert "generate_samples_from_batch" in vars(model), "should be instance-wrapped"
    assert "generate_samples_from_batch" not in vars(model), "wrapper must be removed"
    assert FakeModel.generate_samples_from_batch is original, "class must be untouched"

    # And the baseline path still works, with no hook injected.
    model.captured.clear()
    model.generate_samples_from_batch()
    assert model.captured == [], "no hook should run outside the context"


def test_hook_removed_even_when_body_raises():
    model = FakeModel()
    with pytest.raises(RuntimeError, match="boom"):
        with action_only(model, mode=FREEZE):
            model.generate_samples_from_batch()  # fire the hook so the exit check passes
            raise RuntimeError("boom")
    assert "generate_samples_from_batch" not in vars(model)


def test_never_running_the_hook_is_a_hard_error():
    """The load-bearing safety property.

    If the model never takes the postprocess path, the result is the *baseline* — and
    silently labelling it 'no imagination' would invent the project's headline finding.
    """
    model = FakeModel()
    with pytest.raises(RuntimeError, match="never invoked"):
        with action_only(model, mode=FREEZE):
            pass  # never called generate_samples_from_batch


def test_refuses_to_overwrite_a_callers_hook():
    model = FakeModel()
    with pytest.raises(RuntimeError, match="already given"):
        with action_only(model, mode=FREEZE):
            model.generate_samples_from_batch(velocity_postprocess_builder=lambda **kw: None)


def test_drop_mode_is_declared_unimplemented_not_silently_ignored():
    """E2b is the mode with the real speedup; pretending it works would be worse
    than refusing, because its latency would be quoted as the saving."""
    model = FakeModel()
    with pytest.raises(NotImplementedError, match="drop"):
        with action_only(model, mode=DROP):
            pass


def test_unknown_mode_is_rejected():
    model = FakeModel()
    with pytest.raises(ValueError, match="unknown mode"):
        with action_only(model, mode="nonsense"):
            pass


def test_missing_vision_shape_is_refused_rather_than_guessed():
    """Without the vision latent shape the split offset is unknown; guessing it could
    zero the action block."""

    class NoVision(FakeModel):
        def generate_samples_from_batch(self, data_batch=None, **kwargs):
            builder = kwargs["velocity_postprocess_builder"]

            class Empty:
                x0_tokens_vision = []

            builder(gen_data_clean=Empty())
            return {}

    model = NoVision()
    with pytest.raises(RuntimeError, match="x0_tokens_vision"):
        with action_only(model, mode=FREEZE):
            model.generate_samples_from_batch()


def test_short_velocity_tensor_is_refused():
    """A layout mismatch must abort, not zero an arbitrary slice."""

    class TooShort(FakeModel):
        def generate_samples_from_batch(self, data_batch=None, **kwargs):
            hook = kwargs["velocity_postprocess_builder"](gen_data_clean=FakeGenData())
            hook([torch.ones(10)], None, 0.0)  # far smaller than the vision block
            return {}

    model = TooShort()
    with pytest.raises(RuntimeError, match="layout assumption is wrong"):
        with action_only(model, mode=FREEZE):
            model.generate_samples_from_batch()
