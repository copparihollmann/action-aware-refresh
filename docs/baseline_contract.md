# Baseline contract

**Status:** placeholder. Regenerate via `make contract` after
`scripts/clone_sources.sh` runs. This file must contain **observed** values
from the checked-out Cosmos + RoboLab sources, not assumptions from the
spec.

## Values to observe (from spec §3)

- policy server default port (expected 8000)
- default action denoising steps (expected 4)
- default action chunk size (expected 32)
- default conditioning FPS (expected 15)
- default action dimension (expected 8)
- Cosmos3 RoboLab client open-loop horizon (expected 32)
- decode_video default (expected `False`)
- whether the server returns both `samples["action"]` and `samples["vision"]`

## Critical

`decode_video=False` is NOT "no visual imagination". It only disables VAE
decoding. Whether the joint diffusion transformer still generates vision
latent tokens must be verified in the compute anatomy — this determines
the whole M4–M8 branch of the project.
