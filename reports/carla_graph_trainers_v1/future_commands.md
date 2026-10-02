# Future execution commands (not executed in this milestone)

All full commands require separately authorized execution and an observed environment lock.
The lock must pass artifact/source verification and deterministic CUDA fixture repeats on one T4.
Replace the placeholders with existing paths; no upload commands are supplied.

```text
python reports/carla_graph_trainers_v1/validate_environment.py --validate-kaggle --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --cuda-smoke-output <NEW_CUDA_FIXTURE_DIR> --output <NEW_VALIDATED_LOCK_FILE>
python reports/carla_graph_trainers_v1/train.py train --method nograph --seed 101 --execute-full --device cuda --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --environment-lock <VALIDATED_LOCK_FILE> --output <NEW_NOGRAPH_RUN_ROOT>
python reports/carla_graph_trainers_v1/train.py train --method standard_gat --seed 101 --execute-full --device cuda --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --environment-lock <VALIDATED_LOCK_FILE> --output <NEW_STANDARD_RUN_ROOT>
python reports/carla_graph_trainers_v1/train.py train --method epistemic_gat --seed 101 --execute-full --device cuda --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --environment-lock <VALIDATED_LOCK_FILE> --output <NEW_EPISTEMIC_RUN_ROOT>
python reports/carla_graph_trainers_v1/train.py all --execute-full --device cuda --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --environment-lock <VALIDATED_LOCK_FILE> --output <NEW_ALL_SEED_RUN_ROOT>
```

The all-run plan retains seeds 101,202,303,404,505, in that order, with all three primary methods.
Every run has its own method/seed directory; failed runs and reasons remain in orchestration.json.
No replacement seed, best-seed selection, implicit overwrite or retry occurs.
`all --smoke` runs only seed 101 on a synthetic 128-row engineering fixture.
Neither the original nor validated smoke outputs train on the 89,970-row artifact.
