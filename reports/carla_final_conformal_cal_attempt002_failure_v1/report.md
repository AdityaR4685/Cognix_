# Attempt 002 immutable failure record

HEAD 1f8cb8cca741ad3f3b471b2d5da5129e0309c30b; v2 manifest dd5e559ebf16d73a67d7ecc720ec7f3200465c493e6f804479155e8b32007c66; v2 ZIP 2f1eeb7aff36a6608b643ef1afca5621df5b621b8da79f83ba1232d83418db79.

One GET; 62,739,644,426 compressed bytes; 455 roots; 92/125 CAL completed; 0 EVAL decoded. INCOMPLETE_NO_RETRY. No final thresholds. No retry/resume; raw archive not retained.

Partial compressed SHA256: 5c48384eee84003a00e672de9265e27c3f8a9634deed04250a0b852780c5ba91. External evidence SHA256: a21a6f2ea64eabcdb2c636f79da1cc18709146e583910cd97a52e991331221ea; LOCALLY_VERIFIED_IMMUTABLE_HISTORICAL_EVIDENCE.

Exact exception: segmentation class ids out of range [0, 29): min=1, max=32. Class 32 semantics, failing scenario/tick and unrecorded properties UNKNOWN. The 92 partial CAL score values are contaminated for repair selection; they were not parsed or used for policy/tuning.

Historical opaque_body_bytes_discarded_by_role increments before CAL decode/discard decisions: CAL=14032840421 is body accounting, NOT entirely opaque/discarded. EVAL=53633451131 and EXCLUDED=240747299 are opaque transport/discard. Original ledger unchanged.

v1/v2 remain immutable. Attempt 002 will never be retried. Exact actual traceback follows (historical, not synthetic).

```text
Traceback (most recent call last):
  File "/kaggle/working/cognix_cal_bundle_v2/runner.py", line 200, in execute
    receipt = scan(response, cal, evaluation, exclusions, processor, ledger, EXPECTED_SIZE)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/cognix_cal_bundle_v2/streaming.py", line 118, in scan
    on_cal(sid, relative, read_exact(size))
  File "/kaggle/working/cognix_cal_bundle_v2/runner.py", line 88, in __call__
    feature = segmentation_histogram_features(array)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/cognix_cal_bundle_v2/frozen_sources/cognix/adapters/carla/real_features.py", line 95, in segmentation_histogram_features
    raise ValueError(
ValueError: segmentation class ids out of range [0, 29): min=1, max=32
```
