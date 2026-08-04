# Legal / access checkpoint

This project depends on three licensed components. **None of them are
auto-accepted by our scripts** — the user must accept explicitly.

## 1. Hugging Face gated repo: `nvidia/Cosmos3-Nano-Policy-DROID`

- Check by visiting the model card while signed in:
  https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID
- If gated, click "Accept" on the model card, then export an
  access-token-scoped `HF_TOKEN` in your shell.
- Never write the token to a file. Options:
  - `export HF_TOKEN=hf_...` in the running shell only, OR
  - `uvx hf@latest auth login` (persists to `~/.cache/huggingface` —
    respect that this is a per-user cache).
- Setup scripts refuse to write the token anywhere.

## 2. NVIDIA Cosmos model license

- Bundled with the model on Hugging Face. Accepting the HF gate implies
  agreement — read the license text on the model card before accepting.
- If you distribute derivatives (adapters, LoRA), respect the license.

## 3. NVIDIA Omniverse / Isaac Sim EULA

- Isaac Sim refuses to launch unless `OMNI_KIT_ACCEPT_EULA=Y` is set.
- Read the EULA at:
  https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html
- If you accept, set `export OMNI_KIT_ACCEPT_EULA=Y` in the shell that
  runs `scripts/run_robolab.sh`. Setup does not require it.

## What to do before pressing go

- [ ] Verified access to `nvidia/Cosmos3-Nano-Policy-DROID` on HF.
- [ ] Set `HF_TOKEN` (or `hf auth login`) in the running shell.
- [ ] Read the Cosmos model license.
- [ ] Read the Isaac Sim EULA; if accepting, set `OMNI_KIT_ACCEPT_EULA=Y`.

If any of the above is not yet done, stop before `make setup`.
