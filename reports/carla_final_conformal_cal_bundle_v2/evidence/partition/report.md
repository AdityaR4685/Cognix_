# COGNIX final CARLA partition v1

FINAL_PARTITION_SEALED

Amendment 003 executed once with NumPy 2.3.5, PCG64, seed 334227055836169582741732795999208677637. The seed is permanently spent. One Generator/PCG64 instantiation and one permutation; no stratification, balancing, redraw or reassignment.

The committed structure-only inventory passed independent manifest, hash, readiness, count and seal checks before any RNG was instantiated. The pre-execution JSON was sealed before execution; its hash and initial manifest seal are retained in the execution receipt.

Source inventory seal: `e5219b9dae9447bf8d818828e5ef9ee7987cc5d375f345c4be7d52ad045a333c`.

Eligible inventory seal: `9c08a8cc404b857d8030d316eac12f26fd9145e301ecdce1c081a4c82e981218`.

Sorted input is the exact UTF-8/LF representation in eligible_inventory_sorted.txt; its SHA-256 is `d7badb7fb3db3ef4006da603ce319c9616cc9549a26a2c542e506ca3b7b7bfcb`.

Permutation order and zero-based assignment positions are preserved. Positions [0:125] are FINAL_CONFORMAL_CAL (125); [125:625] are FINAL_EVALUATION (500). Overlap is zero; union is exactly the 625 eligible IDs; both historical exclusions are absent.

Membership SHA-256: `e41392264aae01e47c67b27fa8ad3a1c28c7c589c55e35e778d0502d4e567ada`.

The aggregate partition seal is the SHA-256 of SHA256SUMS, recorded in SHA256SUMS.sha256. The manifest covers this script and all ten JSON/TXT/Markdown outputs; seal files do not hash themselves.

Descriptive counts below are observational only, derived from canonical ID paths. NORMAL is the normal-scenario anomaly-type bucket. No composition affected assignment.

```json
{
  "FINAL_CONFORMAL_CAL": {
    "anomaly_type": {
      "NORMAL": 23,
      "change-weather": 11,
      "running-pedestrian": 4,
      "spawn-props": 20,
      "steer-driver": 3,
      "street-light-flicker": 25,
      "traffic-light-flicker": 2,
      "traffic-light-off": 3,
      "traffic-light-yellow-blinking": 15,
      "vanish-actor": 19
    },
    "normal_anomaly": {
      "anomaly": 102,
      "normal": 23
    },
    "total": 125,
    "town": {
      "Town01": 20,
      "Town02": 26,
      "Town03": 20,
      "Town04": 24,
      "Town05": 20,
      "Town10HD": 15
    }
  },
  "FINAL_EVALUATION": {
    "anomaly_type": {
      "NORMAL": 84,
      "change-weather": 84,
      "running-pedestrian": 5,
      "spawn-props": 100,
      "steer-driver": 7,
      "street-light-flicker": 73,
      "traffic-light-flicker": 28,
      "traffic-light-off": 5,
      "traffic-light-yellow-blinking": 63,
      "vanish-actor": 51
    },
    "normal_anomaly": {
      "anomaly": 416,
      "normal": 84
    },
    "total": 500,
    "town": {
      "Town01": 89,
      "Town02": 108,
      "Town03": 79,
      "Town04": 64,
      "Town05": 70,
      "Town10HD": 90
    }
  },
  "membership_reassigned": false,
  "observational_only": true
}
```

Validation used only persisted lists, sets and hashes; randomization was never replayed. No TEST archive payload, images, Feather contents, anomaly labels/timesteps, sensor values, features, models, predictions, conformal fitting or performance metrics were accessed. No source inventory or preregistration file was modified; no commit or push was performed.
