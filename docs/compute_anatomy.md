# Compute anatomy (M2)

**Status:** placeholder. Populated after `make profile` completes. This
file is the single go/no-go artifact for the visual-imagination branch of
the project: it must answer *"is visual imagination a substantial share of
the deployed Cosmos3 policy cost, or is the dominant cost shared/action
computation?"*.

## What must be filled in

For each of B0–B4 (spec §9):
- stage-level latency table (ms mean ± std across N warmed requests)
- CUDA kernel time per stage
- peak allocated / reserved VRAM
- estimated FLOPs + coverage %
- action vs vision token counts (post-inspection of the model modules)
- at least one PyTorch profiler chrome trace path

Then a written conclusion identifying the dominant cost and — critically —
whether action-only and vision-latent-generation can be measured
independently, or whether they share attention.
