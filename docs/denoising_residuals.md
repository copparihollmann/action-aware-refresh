# Experiment F (offline): cross-denoising-step residuals

First measurement of **our own mechanism** rather than a mandated baseline. One request (`captured_request_BananaInBowlTask_req0000.npz`), 4 denoising steps, 36 transformer blocks, 2 forward(s) per step, 576 block x step x modality samples.

`d[s,l] = ||R[s,l] - R[s-1,l]|| / ||R[s-1,l]||` per §11.6, computed separately over text, vision and action tokens using the model's **actual** packed-sequence indices — not assumed offsets, since a mis-slice would invalidate the one comparison this experiment exists to make.

Token counts: text 95, vision 3,060, action 33.

## Median relative change between consecutive denoising steps

| modality | median d | share of samples below 1% | below 2% | below 5% | below 10% |
|---|---|---|---|---|---|
| **action** | 0.1792 | 0% | 0% | 0% | 1% |
| **text** | 0.0000 | 100% | 100% | 100% | 100% |
| **vision** | 0.3150 | 0% | 0% | 0% | 0% |

> **The asymmetry is real: action-token residuals are 0.57x the vision-token residuals** (0.179 vs 0.315). The action stream *is* the more stable of the two, which is the direction this project's action-aware framing predicted — now measured rather than assumed.

> **But nothing is cacheable at this step count, so the asymmetry is not exploitable.** Only 0.9% of generation-token samples change by less than 10% between consecutive denoising steps, and **0%** change by less than 5%. Reuse needs near-identical block outputs; an 18–32% change is a different tensor.

This is a **negative result for Experiment F as framed**, and it has a clean mechanical explanation. TeaCache / DeepCache / token-wise caching all operate on 20–50-step diffusion schedules, where consecutive steps genuinely barely differ. Cosmos3-Nano runs **4** steps. The short schedule has already squeezed out the inter-step redundancy those methods harvest — so there is nothing left for a cache to recover, and F3's separate action/vision thresholds are correct in principle but moot in practice here.

It also suggests the one reframing that could revive F: a *longer* schedule plus aggressive caching, targeting equal compute to the 4-step baseline. That is speculative and this data argues against it — at 18–32% per-step change the cache would have to be nearly free to break even — but it is the only version of F these numbers do not already rule out.

> **One exact, free saving does exist.** Text-token residuals are **identically zero** across every block and every step — the understanding tower's output does not depend on the noised latents at all, so it can be computed once per request and reused for every denoising step with *bit-identical* results, not merely small error. The ceiling is small because text is only 95 of 3,188 tokens (~3%), and it is worth checking whether the implementation already avoids recomputing it before claiming the saving.

## Per-step change (does denoising converge?)

| modality | step 1 | step 2 | step 3 |
|---|---|---|---|
| **action** | 0.1976 | 0.1311 | 0.2156 |
| **text** | nan | 0.0000 | 0.0000 |
| **vision** | 0.4273 | 0.3089 | 0.2912 |

A falling trend means later steps change less and are the cheapest to cache; a flat trend means every step matters and uniform whole-step reuse (F0) will hurt.

## Limitations

- **One request.** Enough to establish the shape and to decide whether to build a cache at all; not enough to set thresholds. Those need the full corpus.
- **No cache is implemented yet**, so nothing here is a speedup. §11.6 requires the analysis first precisely so that a cache is only built if there is something to reuse — and any cache must then be charged its own lookup/copy overhead (spec §7), which memory traffic can easily exceed the compute saved.
- **Residual size is not the same as action impact.** A block whose output barely moves may still be the one the action tokens depend on. The closed-loop lesson from Experiments A and E applies here in advance: this session already showed that an offline metric matched to sampling noise failed to predict closed-loop success, so any threshold chosen from this table must be validated against task success before being believed.

