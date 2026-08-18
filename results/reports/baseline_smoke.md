# Baseline smoke — **PASS**

- Primary task: `BananaInBowlTask`
- Alt task: `RubiksCubeAndBananaTask` → PASS

> `INCOMPLETE` means at least one criterion had no evidence either way. It is deliberately not `PASS`: spec §7.3 requires positive evidence for every criterion. Resolve UNKNOWNs by reading the real log and tightening the pattern in `scripts/validate_smoke.py`.

## Server checks

| check | state | detail | provenance |
|---|---|---|---|
| `server_started` | ✅ PASS | [robolab-policy-server] starting host= | VERIFIED: serve() startup line |
| `healthz_advertised` | ✅ PASS | [robolab-policy-server] Health check: http:// | VERIFIED: serve() health-check line |
| `config_logged` | ✅ PASS | action_space=joint_pos action_dim=8 | VERIFIED: RobolabPolicyService logs the resolved config |
| `request_served` | ✅ PASS | [robolab-policy-server] prompt= | VERIFIED: infer() logs prompt+seed per request |
| `no_nans` | ✅ PASS | none found | counter-evidence: numerical failure in the served output |
| `no_traceback` | ✅ PASS | none found | counter-evidence: server exception |
| `no_cuda_oom` | ✅ PASS | none found | counter-evidence: OOM — expected risk at 45 GiB with ~31 GiB of weights |
| `action_dim_matches_contract` | ✅ PASS | server reported action_dim=8, contract says 8 | compared against docs/baseline_contract.md |

## Primary client checks

| check | state | detail | provenance |
|---|---|---|---|
| `client_connected` | ✅ PASS | [Cosmos3Client] Connected to 127.0.0.1:8000. | VERIFIED: policies/cosmos3/client.py __init__ prints this after _connect() |
| `episode_started` | ✅ PASS | [RoboLab] Running BananaInBowlTask_0:  | VERIFIED: robolab/eval/runner.py:251 |
| `output_dir_announced` | ✅ PASS | Output         : /scratch/agustin/robotics/action-aware-refresh/third_party/RoboLab/output/2026-08-03_22-13-27_cosmos3 | VERIFIED: robolab/core/utils/print_utils.py:30 |
| `no_traceback` | ✅ PASS | none found | counter-evidence: client exception |
| `no_terminated_with_error` | ✅ PASS | none found | counter-evidence: VERIFIED: policies/cosmos3/run.py:56 prints this on any exception |
| `results_written` | ✅ PASS | 1 episode record(s) in /scratch/agustin/robotics/action-aware-refresh/third_party/RoboLab/output/2026-08-03_22-13-27_cosmos3/episode_results.jsonl | VERIFIED: episode_results.jsonl (robolab/core/logging/results.py:606) |
| `sim_advanced` | ✅ PASS | episode_step = [145] | VERIFIED: episode_results.jsonl (robolab/core/logging/results.py:606) |
| `episode_terminated` | ✅ PASS | success=True step=145 score=1.0 reason="Completed subtask 'pick_and_place' 1/1" | VERIFIED: episode_results.jsonl (robolab/core/logging/results.py:606) |
| `metrics_finite` | ✅ PASS | all trajectory metrics finite | VERIFIED: episode_results.jsonl (robolab/core/logging/results.py:606) |

## Alt-task client checks (not required for PASS)

| check | state | detail | provenance |
|---|---|---|---|
| `client_connected` | ✅ PASS | [Cosmos3Client] Connected to 127.0.0.1:8000. | VERIFIED: policies/cosmos3/client.py __init__ prints this after _connect() |
| `episode_started` | ✅ PASS | [RoboLab] Running RubiksCubeAndBananaTask_0:  | VERIFIED: robolab/eval/runner.py:251 |
| `output_dir_announced` | ✅ PASS | Output         : /scratch/agustin/robotics/action-aware-refresh/third_party/RoboLab/output/2026-08-03_22-15-37_cosmos3 | VERIFIED: robolab/core/utils/print_utils.py:30 |
| `no_traceback` | ✅ PASS | none found | counter-evidence: client exception |
| `no_terminated_with_error` | ✅ PASS | none found | counter-evidence: VERIFIED: policies/cosmos3/run.py:56 prints this on any exception |
| `results_written` | ✅ PASS | 1 episode record(s) in /scratch/agustin/robotics/action-aware-refresh/third_party/RoboLab/output/2026-08-03_22-15-37_cosmos3/episode_results.jsonl | VERIFIED: episode_results.jsonl (robolab/core/logging/results.py:606) |
| `sim_advanced` | ✅ PASS | episode_step = [900] | VERIFIED: episode_results.jsonl (robolab/core/logging/results.py:606) |
| `episode_terminated` | ✅ PASS | success=False step=900 score=1.0 reason='Condition not satisfied: object_in_container(object=banana, container=bowl, require_contact_with=False, require_gripper_detached=True) (step 2/2)' | VERIFIED: episode_results.jsonl (robolab/core/logging/results.py:606) |
| `metrics_finite` | ✅ PASS | all trajectory metrics finite | VERIFIED: episode_results.jsonl (robolab/core/logging/results.py:606) |

## Primary closed-loop timing (runner-reported)

| run | steps | policy inference | env step | video write | wall | it/s |
|---|---|---|---|---|---|---|
| `BananaInBowlTask_0` | 145 | 44.9 s (309 ms/step) | 32.4 s (223 ms/step) | 3.1 s | 80.4 s | 1.8 |

> Reported by RoboLab itself, so these are end-to-end wall times including serialization and the websocket round-trip — not GPU time. `policy inference` is the client's total time in the policy step averaged over **all** control steps, most of which reuse a cached chunk (`Cosmos3Client.OPEN_LOOP_HORIZON = 32`); divide by the server-side request count for true per-call latency.


## Alt closed-loop timing (runner-reported)

| run | steps | policy inference | env step | video write | wall | it/s |
|---|---|---|---|---|---|---|
| `RubiksCubeAndBananaTask_0` | 900 | 136.7 s (152 ms/step) | 216.8 s (241 ms/step) | 16.7 s | 370.3 s | 2.43 |

> Reported by RoboLab itself, so these are end-to-end wall times including serialization and the websocket round-trip — not GPU time. `policy inference` is the client's total time in the policy step averaged over **all** control steps, most of which reuse a cached chunk (`Cosmos3Client.OPEN_LOOP_HORIZON = 32`); divide by the server-side request count for true per-call latency.


## Task outcome (a result, not a pass criterion)

- `BananaInBowlTask_0` — success=**True**, score=1.0, 145 steps, events={'TARGET_OBJECT_DROPPED': 2}, reason="Completed subtask 'pick_and_place' 1/1"
- `RubiksCubeAndBananaTask_0` — success=**False**, score=1.0, 900 steps, events={'TARGET_OBJECT_DROPPED': 11, 'GRIPPER_FULLY_CLOSED': 2, 'WRONG_OBJECT_GRABBED': 2, 'GRIPPER_HIT_OBJECT': 1, 'OBJECT_BUMPED': 2, 'wrong_objects_grabbed': ['banana', 'banana']}, reason='Condition not satisfied: object_in_container(object=banana, container=bowl, require_contact_with=False, require_gripper_detached=True) (step 2/2)'

## Log tails

> Progress-bar redraws and known-benign Isaac teardown warnings are filtered out. Those warnings appear on successful runs too, so leaving them in buries the actual last words of a failing process — which is exactly what happened in the earlier FAIL reports.

### server (last 30 meaningful lines)

```
Sampling:   0%|          | 0/4 [00:00<?, ?it/s]
Sampling:  25%|██▌       | 1/4 [00:00<00:02,  1.18it/s]
Sampling:  50%|█████     | 2/4 [00:01<00:01,  1.23it/s]
Sampling:  75%|███████▌  | 3/4 [00:02<00:00,  1.22it/s]
Sampling: 100%|██████████| 4/4 [00:03<00:00,  1.21it/s]
Sampling: 100%|██████████| 4/4 [00:03<00:00,  1.22it/s]
[08-03 22:21:35|job=|INFO|cosmos_framework/scripts/action_policy_server_robolab.py:595:infer] [robolab-policy-server] prompt='Put the cube and the banana in the bowl. This video contains concatenated views from multiple camera perspectives. The top row is from the wrist-mounted camera. The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite sides, with the robot visible. The video is 2.0 seconds long and is of 15 FPS. This video is of 544x736 resolution.' seed=377217572
[08-03 22:21:35|job=|INFO|cosmos_framework/model/generator/omni_mot_model.py:3087:generate_samples_from_batch] Using sampler: UniPC (shift=5.0, num_steps=4)
Sampling:   0%|          | 0/4 [00:00<?, ?it/s]
Sampling:  25%|██▌       | 1/4 [00:00<00:02,  1.02it/s]
Sampling:  50%|█████     | 2/4 [00:01<00:01,  1.13it/s]
Sampling:  75%|███████▌  | 3/4 [00:02<00:00,  1.11it/s]
Sampling: 100%|██████████| 4/4 [00:03<00:00,  1.13it/s]
Sampling: 100%|██████████| 4/4 [00:03<00:00,  1.12it/s]
[08-03 22:21:48|job=|INFO|cosmos_framework/scripts/action_policy_server_robolab.py:595:infer] [robolab-policy-server] prompt='Put the cube and the banana in the bowl. This video contains concatenated views from multiple camera perspectives. The top row is from the wrist-mounted camera. The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite sides, with the robot visible. The video is 2.0 seconds long and is of 15 FPS. This video is of 544x736 resolution.' seed=191741831
[08-03 22:21:48|job=|INFO|cosmos_framework/model/generator/omni_mot_model.py:3087:generate_samples_from_batch] Using sampler: UniPC (shift=5.0, num_steps=4)
Sampling:   0%|          | 0/4 [00:00<?, ?it/s]
Sampling:  25%|██▌       | 1/4 [00:00<00:02,  1.07it/s]
Sampling:  50%|█████     | 2/4 [00:01<00:01,  1.12it/s]
Sampling:  75%|███████▌  | 3/4 [00:02<00:00,  1.14it/s]
Sampling: 100%|██████████| 4/4 [00:03<00:00,  1.15it/s]
Sampling: 100%|██████████| 4/4 [00:03<00:00,  1.14it/s]
[08-03 22:22:01|job=|INFO|cosmos_framework/scripts/action_policy_server_robolab.py:595:infer] [robolab-policy-server] prompt='Put the cube and the banana in the bowl. This video contains concatenated views from multiple camera perspectives. The top row is from the wrist-mounted camera. The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite sides, with the robot visible. The video is 2.0 seconds long and is of 15 FPS. This video is of 544x736 resolution.' seed=1853662621
[08-03 22:22:01|job=|INFO|cosmos_framework/model/generator/omni_mot_model.py:3087:generate_samples_from_batch] Using sampler: UniPC (shift=5.0, num_steps=4)
Sampling:   0%|          | 0/4 [00:00<?, ?it/s]
Sampling:  25%|██▌       | 1/4 [00:00<00:02,  1.03it/s]
Sampling:  50%|█████     | 2/4 [00:01<00:01,  1.11it/s]
Sampling:  75%|███████▌  | 3/4 [00:02<00:00,  1.09it/s]
Sampling: 100%|██████████| 4/4 [00:03<00:00,  1.12it/s]
Sampling: 100%|██████████| 4/4 [00:03<00:00,  1.10it/s]
```

### primary (last 30 meaningful lines)

```
+-------+------+--------+
| Index | Name | Weight |
+-------+------+--------+
+-------+------+--------+
[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
+----------------------+
| Active Curriculum Terms |
+-----------+----------+
|   Index   | Name     |
+-----------+----------+
+-----------+----------+
[INFO]: Completed setting up the environment...
[96m┌──────────────────────────────────────────────────────────────┐
│  Environment : BananaInBowlTask                              │
├──────────────────────────────────────────────────────────────┤
│  Instruction : Pick up the banana and place it in the bowl   │
│  Instr. Type : default                                       │
│  Seed        : 0                                             │
│  Policy      : cosmos3                                       │
│  Scene       : BananaInBowlTaskSceneEnvCfg                   │
│  Attributes  : semantics, simple                             │
└──────────────────────────────────────────────────────────────┘[0m
[Cosmos3Client] Awaiting for server on 127.0.0.1:8000 to be ready...
[Cosmos3Client] Connected to 127.0.0.1:8000.
[96m[RoboLab] Running BananaInBowlTask_0: 'Pick up the banana and place it in the bowl' (run 0, 1 envs)[0m
2026-08-04T05:13:43Z [53,078ms] [Warning] [omni.hydra] Parameter 'metallic' of shade node 'usd::rtx_scope0::/__Fabric_StageInfo/MaterialPool/mat_f803a956/OmniPBR::::OmniPBR::OmniPBR((151))' not available in the MDL representation.
2026-08-04T05:13:43Z [53,078ms] [Warning] [omni.hydra] Parameter 'roughness' of shade node 'usd::rtx_scope0::/__Fabric_StageInfo/MaterialPool/mat_f803a956/OmniPBR::::OmniPBR::OmniPBR((151))' not available in the MDL representation.
2026-08-04T05:13:43Z [53,080ms] [Warning] [omni.hydra] Parameter 'metallic' of shade node 'usd::rtx_scope0::/__Fabric_StageInfo/MaterialPool/mat_917d5f76/OmniPBR::::OmniPBR::OmniPBR((170))' not available in the MDL representation.
2026-08-04T05:13:43Z [53,080ms] [Warning] [omni.hydra] Parameter 'roughness' of shade node 'usd::rtx_scope0::/__Fabric_StageInfo/MaterialPool/mat_917d5f76/OmniPBR::::OmniPBR::OmniPBR((170))' not available in the MDL representation.
2026-08-04T05:14:16Z [86,914ms] [Warning] [carb] Client gpu.foundation.plugin has acquired [gpu::unstable::IMemoryBudgetManagerFactory v0.1] 100 times. Consider accessing this interface with carb::getCachedInterface() (Performance warning)
```

### alt (last 30 meaningful lines)

```
|  Active Reward Terms  |
+-------+------+--------+
| Index | Name | Weight |
+-------+------+--------+
+-------+------+--------+
[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
+----------------------+
| Active Curriculum Terms |
+-----------+----------+
|   Index   | Name     |
+-----------+----------+
+-----------+----------+
[INFO]: Completed setting up the environment...
[96m┌──────────────────────────────────────────────────────────────┐
│  Environment : RubiksCubeAndBananaTask                       │
├──────────────────────────────────────────────────────────────┤
│  Instruction : Put the cube and the banana in the bowl       │
│  Instr. Type : default                                       │
│  Seed        : 0                                             │
│  Policy      : cosmos3                                       │
│  Scene       : RubiksCubeAndBananaTaskSceneEnvCfg            │
│  Attributes  : conjunction, simple                           │
└──────────────────────────────────────────────────────────────┘[0m
[Cosmos3Client] Awaiting for server on 127.0.0.1:8000 to be ready...
[Cosmos3Client] Connected to 127.0.0.1:8000.
[96m[RoboLab] Running RubiksCubeAndBananaTask_0: 'Put the cube and the banana in the bowl' (run 0, 1 envs)[0m
2026-08-04T05:15:54Z [42,661ms] [Warning] [omni.hydra] Parameter 'roughness' of shade node 'usd::rtx_scope0::/__Fabric_StageInfo/MaterialPool/mat_f803a956/OmniPBR::::OmniPBR::OmniPBR((113))' not available in the MDL representation.
2026-08-04T05:15:54Z [42,669ms] [Warning] [omni.hydra] Parameter 'metallic' of shade node 'usd::rtx_scope0::/__Fabric_StageInfo/MaterialPool/mat_917d5f76/OmniPBR::::OmniPBR::OmniPBR((227))' not available in the MDL representation.
2026-08-04T05:15:54Z [42,669ms] [Warning] [omni.hydra] Parameter 'roughness' of shade node 'usd::rtx_scope0::/__Fabric_StageInfo/MaterialPool/mat_917d5f76/OmniPBR::::OmniPBR::OmniPBR((227))' not available in the MDL representation.
2026-08-04T05:16:05Z [53,780ms] [Warning] [carb] Client gpu.foundation.plugin has acquired [gpu::unstable::IMemoryBudgetManagerFactory v0.1] 100 times. Consider accessing this interface with carb::getCachedInterface() (Performance warning)
```

