# Legal / access checkpoint

This project depends on three licensed components. **None of them are
auto-accepted by our scripts** — the user must accept explicitly. Scripts in
this repo never set `OMNI_KIT_ACCEPT_EULA` and never write a token to disk;
authorization is recorded here, in prose, with a date.

## 1. Hugging Face repo: `nvidia/Cosmos3-Nano-Policy-DROID`

**Status: not gated — no acceptance action required.** Checked 2026-08-03 via
the HF API:

```
GET https://huggingface.co/api/models/nvidia/Cosmos3-Nano-Policy-DROID
  gated   : False
  private : False
  sha     : 6706d7680581c255ff61e0f3bb49d90eac55c79e
  license : other
```

There is no click-through gate, and **no `HF_TOKEN` is required** to download
the weights (32.9 GB across 43 files). The token machinery below is retained
for future gated models only.

- Never write the token to a file. Options:
  - `export HF_TOKEN=hf_...` in the running shell only, OR
  - `uvx hf@latest auth login` (persists to `~/.cache/huggingface` —
    respect that this is a per-user cache).
- Setup scripts refuse to write the token anywhere.

## 2. NVIDIA Cosmos model license

- The model card declares `license: other` — the NVIDIA Open Model License,
  bundled with the model on Hugging Face.
- Because the repo is **not** gated, nothing is accepted implicitly by
  downloading. Read the license text on the model card before publishing or
  distributing derivatives (adapters, LoRA weights, residual modules).
- Recorded, not accepted on the user's behalf.

## 3. NVIDIA Omniverse / Isaac Sim EULA

**Status: AUTHORIZED by the user on 2026-08-03 (session 2, host `firesim2`).**

- The user was shown that RoboLab's install-verification suite
  (`uv run pytest tests/`) *auto-accepts* the Omniverse EULA — per RoboLab's own
  README: "The suite auto-accepts the NVIDIA Omniverse EULA so the run is fully
  headless with no prompts" — and that other entry points require
  `OMNI_KIT_ACCEPT_EULA=Y`.
- The user explicitly authorized acceptance so that M1 (closed-loop smoke test)
  and M2 (compute anatomy) could proceed. `OMNI_KIT_ACCEPT_EULA=Y` is therefore
  set **in the run environment**, not baked into any committed script — so this
  repo stays honest for anyone else who clones it.
- EULA text:
  https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html

## What to do before pressing go

- [x] Verified access to `nvidia/Cosmos3-Nano-Policy-DROID` on HF — **not gated**.
- [x] `HF_TOKEN` — **not required** for this checkpoint (kept optional).
- [x] Cosmos model license recorded (`other` / NVIDIA Open Model License);
      read before distributing derivatives.
- [x] Isaac Sim EULA — **user authorized 2026-08-03**; set in the run
      environment only.

All four items are settled for this session. If a *new* gated model or a
different simulator stack is introduced, re-open this file and get fresh
authorization — do not carry this one forward by assumption.
