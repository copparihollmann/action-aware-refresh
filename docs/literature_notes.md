# Literature notes

Implementation-focused. Each entry: reusable state · unit of compute ·
inference path · training reqs · code availability · computational claim ·
evaluation metric · limitation · what we can reuse. All entries are
provisional until we re-read the current version of each source with the
final commit SHAs recorded in `reproducibility/source_manifest.json`.

---

## Cosmos3-Nano-Policy-DROID (our primary model)

- **Reusable state:** joint (vision, action) latent stream in a
  diffusion transformer. Action chunk of 32; open-loop horizon 32 in
  the RoboLab client. Vision tokens are still generated even when
  VAE decoding is off.
- **Unit of compute:** one denoising step across the joint token sequence
  (default 4 steps). Vision decoder = VAE.
- **Inference path:** obs → vision encoder → context (proprio, prompt) →
  joint diffusion (4 steps) → (action head, vision decoder [optional]).
- **Training reqs:** we do NOT retrain. LoRA / adapters allowed.
- **Code availability:** cosmos-framework policy server, HF weights (gated).
- **Comp claim:** ~real-time on 1 modern GPU; exact numbers must be
  measured, not quoted.
- **Eval metric:** RoboLab task success.
- **Limitation:** joint token stream means "action only" isn't trivially
  free unless attention decouples.
- **What we reuse:** everything — this is the baseline.

## ImageWAM

- **Reusable state:** world model latent conditioned on action history.
- **Unit of compute:** one autoregressive step per predicted frame.
- **Inference path:** action → world latent → next frame prediction.
- **Training reqs:** massive; we don't touch training.
- **Code availability:** paper + partial code.
- **Comp claim:** frame-level world model, high FPS.
- **Eval metric:** image reconstruction + downstream task.
- **Limitation:** image-quality-focused; not our metric.
- **What we reuse:** the framing of "world state as a reusable latent",
  applied to Cosmos3's vision latent.

## Foveated Diffusion (Chao et al.)

- **Reusable state:** high-res tokens only in a saliency mask; low-res
  elsewhere.
- **Unit of compute:** token-wise attention.
- **Inference path:** downsampled base + high-res patches inside mask.
- **Training reqs:** requires training or a training-free variant.
- **Code availability:** project page.
- **Comp claim:** substantial token reduction with limited quality loss.
- **Eval metric:** image fidelity.
- **Limitation:** image-fidelity metric; we care about action success.
- **What we reuse:** saliency-driven token-mask idea for Experiment G
  (oracle spatial refresh).

## DeltaTok / DeltaWorld

- **Reusable state:** delta between consecutive frames' token sequences
  — reuse most tokens, refresh only changed ones.
- **Unit of compute:** transformer forward on updated tokens only.
- **Inference path:** encode delta → sparse update.
- **Training reqs:** requires a delta-aware decoder or fine-tune.
- **Code availability:** project page.
- **Comp claim:** near-linear speedup with sparsity.
- **Eval metric:** video/world model quality.
- **Limitation:** relies on temporal coherence; action-driven changes may
  invalidate large regions abruptly.
- **What we reuse:** token-delta caching approach for Experiment F/G.

## Fast-WAM

- **Reusable state:** none — the paper's point is that visual imagination
  can be dropped without losing action quality.
- **Unit of compute:** action-only forward pass.
- **Inference path:** obs → action, skip visual generation.
- **Training reqs:** dedicated action head or masked training.
- **Code availability:** paper + project page.
- **Comp claim:** action-only is much cheaper with similar success.
- **Eval metric:** robot task success.
- **Limitation:** requires that vision latent is not essential for action.
- **What we reuse:** the *challenge* to our thesis. Experiment E must
  test action-only against imagined-keyframe methods. If Fast-WAM wins,
  visual caching is not the main contribution.

## TeaCache

- **Reusable state:** residual outputs of a diffusion transformer step —
  reuse when input change is small.
- **Unit of compute:** transformer block outputs.
- **Inference path:** timestep-conditioned reuse decision, then either
  reuse cached residual or recompute.
- **Training reqs:** training-free.
- **Code availability:** GitHub repo, MIT-ish license (verify).
- **Comp claim:** ≈1.5–2× on some video diffusion pipelines.
- **Eval metric:** FID/LPIPS.
- **Limitation:** image-metric; we must re-evaluate on action success.
- **What we reuse:** block-level residual reuse as a design reference
  (Experiment F).

## FasterCache

- **Reusable state:** cross-step and cross-block features.
- **Unit of compute:** attention/FFN outputs.
- **Inference path:** hybrid CFG + cache guidance.
- **Training reqs:** training-free.
- **Code availability:** GitHub.
- **Comp claim:** ≈1.5–3× on video diffusion.
- **What we reuse:** design reference for hybrid caching; check license.

## DeepCache

- **Reusable state:** deep-block features across denoising steps.
- **Unit of compute:** transformer blocks.
- **Inference path:** step-conditional shallow-block recompute.
- **Comp claim:** 2–5× on image diffusion.
- **What we reuse:** design reference for the step-block scheduling
  matrix in Experiment F.

## Token-wise feature caching (arXiv 2410.05317)

- **Reusable state:** per-token features across steps.
- **Comp claim:** token-level sparsity.
- **What we reuse:** direct baseline for the token-mask variant of F.

## v2e / ESIM

- **Reusable state:** simulated events from RGB frames.
- **Unit of compute:** temporal-derivative pipeline.
- **What we reuse:** OFFLINE robustness testing of the event-based gate
  (Experiment D), not the online inference path — v2e is heavy.

## RAFT (torchvision)

- **Reusable state:** optical flow field.
- **Comp claim:** ~10–20 ms per pair at low resolution on modern GPUs.
- **What we reuse:** D3 flow signal. Small variant, low resolution.
  Include its compute cost in the total.

## OpenPI π0.5 / π0.7 (Physical Intelligence)

- **Reusable state:** action chunks, flow-matching action head, vision
  encoder.
- **Comp claim:** π0.5 is public; π0.7 is a blog. π0.5 is the
  architecture we can honestly claim transfer against (spec §11.10).
- **What we reuse:** the generic mechanism (chunk reuse, event/flow gate,
  step budget) applied through π0.5's inference path.
