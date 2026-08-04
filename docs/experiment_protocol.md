# Experiment protocol (spec §8, §10)

## Timing layers

1. **Client end-to-end**: `time.perf_counter_ns()` around the whole
   policy-request round trip. Includes serialization + network + queue.
2. **Server stages via CUDA events**: `CudaStageTimer` in
   `src/action_refresh/profiler.py`. Named stages per spec §9:
   `preprocessing`, `vision_encode`, `context`, `denoising`, `vision_decode`,
   `postprocess`. Do NOT call `torch.cuda.synchronize()` outside `.finalize()`.
3. **Deep dive**: PyTorch Profiler chrome traces on a handful of hand-picked
   requests. Nsight Systems if we can install a user-local `nsys`.

## Energy

- `EnergyMeter` samples NVML power at 10 Hz in a background thread.
- Idle-baseline is measured once at the start of a run and stored in
  `results/raw/<run_id>/energy_idle.json`.
- Report both gross and idle-adjusted energy per episode.
- All energy fields carry the sampling cadence + "ESTIMATED" label in `RunMeta`.

## FLOP counting

- `torch.profiler.with_flops=True` when we launch the profiler on a request.
- `torch.utils.flop_counter.FlopCounterMode` for wider coverage on selected
  requests (see `profiler.flop_counter()`).
- Report FLOP coverage % — if a fused/custom kernel isn't counted, subtract
  its share (analytic estimate) instead of claiming perfection.

## Determinism

- Every run records seeds, model revision SHA, cosmos-framework SHA,
  RoboLab SHA, Isaac Sim version, GPU IDs, and the current `nvidia-smi`
  process list at launch (contention snapshot).
- Paired evaluations use identical initial states across methods whenever
  RoboLab supports it (verify by reading the checked-out client code).

## Statistical protocol

Episode counts by phase (spec §10):
- screening: 5–10 paired episodes / task
- promoted configs: 20–30
- final Pareto: 50 where feasible
- final full benchmark: RoboLab-120's standard protocol

Report macro (per task) + micro (per episode) success rates, 95% CI,
paired differences, stratified by competency and difficulty. Non-inferiority
margins: 0, 1, 2, 5 absolute pp.

## Primary axis

`normalized_total_compute = method_total_including_overhead / baseline_full_total`.
Main figure = success vs `normalized_total_compute`.
