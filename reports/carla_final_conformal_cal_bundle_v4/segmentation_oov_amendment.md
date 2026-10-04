# Offline root cause and fixed policy decision

v2 separates strict uint8 HxWx3 camera validation from uint8 2-D/3-D segmentation validation. Failure traceback enters the unchanged extractor at real_features.py:95, whose class range check is [0,29). Thus representation handling is no longer this failure. No unrecorded TEST image shape or mode is inferred.

The frozen extractor selects channel 0 for 3-D, bins 29 dimensions, and divides by total pixels (the old L2 docstring is inaccurate). Frozen graph_fit_export.OFFSETS Seg=(18,47) and restored one-class vectors bind 29-dimensional input. Extending n_classes to 33 invalidates the fitted model; no refit/checkpoint changes are allowed.

Human-supplied classic CARLA taxonomy defines Other=22 and GuardRail=28. The supplied carla-gen README patch names custom labels 29 and 30. These facts justify possibility of custom/OOV labels beyond 28; they establish no semantic identity for observed ID 32. Class 32 remains UNKNOWN. These sources were not fetched.

Decision: uniformly collapse every uint8 semantic ID >28 into existing generic Other=22 after selecting channel 0. Preserve 0..28 and the unchanged 29-bin extractor. This is a dimensional compatibility policy, not an assertion that an OOV class has the CARLA Other ground-truth meaning. No partial CAL score, method comparison, outcome, frequency or class count enters the decision.

Transparent post-failure scientific compatibility amendment: 92 partial CAL scenarios had already been accessed. The policy can change scores for OOV inputs. No v2 science-identity claim is made. Attempt 003, if separately authorized, starts at byte zero and recomputes all 125 CAL scenarios. No Attempt-002 thresholds exist or may be reused; no EVAL science was decoded. v1/v2 remain immutable.

Runtime audit: v2 runtime_requirements.txt already pins pandas==2.2.3, but runtime_check omits its comparison. v3 adds an explicit check of that existing pin while preserving runtime_lock.json and runtime_requirements.txt byte-for-byte. This changes enforcement, not version selection or scientific method.

The existing device pin is exactly one visible Tesla T4. V3 explicitly enforces the full device name; v2 checked a T4 substring. The pinned hardware is unchanged.
