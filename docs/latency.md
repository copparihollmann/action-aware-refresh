# Measured latency per configuration

Interleaved (round-robin over conditions), cooled (idle gap before each timed request), median-of-many. This protocol exists because the offline study's `wall ms` column is not a latency measurement — it sweeps conditions in back-to-back blocks, and identical work spans 3,377–5,974 ms on this host under sustained load. **Speedups quoted anywhere in this project come from here.**

| condition | steps | frames | CUDA median | MAD | wall median | host overhead | speedup (CUDA) | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| `teacher_steps4` | 4 | 33 | **3,465 ms** | 11 (0.32%) | 3,465 ms | -0 ms | — | 30,996 MiB |
| `steps_3` | 3 | 33 | **2,671 ms** | 17 (0.65%) | 2,670 ms | -0 ms | 1.30x | 30,996 MiB |
| `vision_frames_17` | 4 | 17 | **1,980 ms** | 13 (0.68%) | 1,980 ms | -0 ms | 1.75x | 30,996 MiB |
| `steps_2` | 2 | 33 | **1,886 ms** | 7 (0.35%) | 1,886 ms | -0 ms | 1.84x | 30,996 MiB |
| `vision_frames_9` | 4 | 9 | **1,516 ms** | 7 (0.49%) | 1,516 ms | -0 ms | 2.29x | 30,996 MiB |
| `steps_1` | 1 | 33 | **1,100 ms** | 12 (1.05%) | 1,100 ms | -0 ms | 3.15x | 30,996 MiB |
| `vision_frames_5` | 4 | 5 | **1,073 ms** | 9 (0.80%) | 1,073 ms | -0 ms | 3.23x | 30,996 MiB |
| `steps_1_vision_frames_9` | 1 | 9 | **505 ms** | 10 (2.01%) | 505 ms | -0 ms | 6.86x | 30,996 MiB |
| `steps_1_vision_frames_5` | 1 | 5 | **386 ms** | 11 (2.73%) | 386 ms | -0 ms | 8.97x | 30,996 MiB |

No other GPU compute process was present for any sample — the script refuses to measure otherwise unless explicitly overridden.

## How to read this

- **CUDA median is the model's cost**; wall median includes host-side work. Their difference (`host overhead`) is in-process only and does *not* include the ~1,197 ms of client-side composition, msgpack and websocket round trip measured in the closed loop — that sits on top of every number here and is invariant to anything done inside the model.
- **Speedups are CUDA-time ratios against the 4-step baseline.** They say nothing about task success: `docs/offline_action_study.md` prices the action deviation and `docs/pareto.md` the closed-loop success.
- **A combined condition is measured, not inferred.** Reducing denoising steps and shortening the imagined horizon act on different factors of the same matmul-bound cost, so their speedups might or might not multiply; spec §8 forbids assuming.

