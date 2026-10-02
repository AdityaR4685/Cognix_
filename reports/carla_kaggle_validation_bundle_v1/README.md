# COGNIX Kaggle environment and CUDA validation only

This bundle preserves the frozen scientific sources and graph bytes. It contains
the compact FIT-only graph export, the sealed preregistration files, the unchanged
trainer and its required local import dependencies, and separate engineering
observers. Package initializers require several framework modules, including raw
loader and calibration **definitions**; no raw images, raw archives, upstream
cache arrays or calibration calls are included or used.

Use a Kaggle notebook with exactly **one visible T4**. This user's notebook offers
T4 x2; the user explicitly authorized `CUDA_VISIBLE_DEVICES=0` before importing
PyTorch. The lock records the two-GPU allocation and one-GPU exposure. Preserve the installed runtime for observation;
do not install versions matching the local CPU lock. If Kaggle cannot supply a
single visible T4, stop and record that limitation.

1. Attach `cognix_kaggle_validation_v1.zip` as a private notebook input. Keep the
   externally recorded archive hash and `bundle_SHA256SUMS_sha256` from
   `preparation_results.json` for verification. Do not publish the bundle.
2. Extract into `/kaggle/working/cognix_validation` and verify the seal **before
   importing any bundled Python module**. The notebook template beside this
   README supplies the extraction and validation cells, with exact local hashes.
3. Run only this validation command in a fresh subprocess:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/validate_kaggle.py \
  --output /kaggle/working/carla_kaggle_environment_lock_v1
```

The real artifact is loaded exclusively for read-only content/schema/split
verification. All six tiny training runs construct the unchanged 128-row
synthetic fixture: three methods, seed 101, two fresh processes each. The frozen
trainer's maximum of 100 tiny epochs and early-stopping settings are preserved.
One additional synthetic batch of 256 per method records resource use. The
fixed-logit diagnostic alters only a disposable fixture model. No real graph
training, TEST access, conformal fitting, scientific aggregation, source patching,
batch-size tuning, commit or push occurs.

The observer uses Python profiling to read the trainer's initial/final state and
first forward RNG; it does not replace any trainer operation. Repeat comparisons
include every saved checkpoint's tensor/state content. The existing exact-repeat
gate is preserved. CPU-to-GPU errors are measured without adding a tolerance.
Generic-forward checks retain atol=1e-7, rtol=1e-6; attention normalization retains
atol=1e-6 and PyTorch's existing default rtol=1e-5. Any failed deterministic kernel,
OOM, hash, repeat or pairing check stops validation and retains failure evidence.

Successful actual CUDA validation writes environment_lock.json,
cuda_smoke_results.json, hash_verification.json, reproducibility_results.json,
resource_results.json, execution_readiness.json and report.md. The lock binds the
evidence and bundle SHA-256 seal and has an external `.sha256` file. A failed run
never produces a successful lock. CPU harness testing creates no CUDA lock.

Download the complete output folder, including fixture checkpoints, histories,
predictions, metrics and logs. Review that evidence before separately authorizing
the full experiment. Future commands in `future_full_commands.md` are a draft
until an actual validated lock exists. `launch_full.py` verifies the complete
bundle, real graph, sealed lock, evidence and observed runtime first. Its default
action prints the plan; full execution additionally requires
`--execute-full-experiment`. Never use that flag in this milestone.
