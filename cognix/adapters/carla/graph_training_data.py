"""Read-only, hash-bound primary graph data and isolated engineering fixtures."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from cognix.adapters.carla import graph_fit_export as ge

ARTIFACT_SHA256 = "baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237"
SCHEMA_SHA256 = "5b678571d4ed112e5756e1364ff7966567a1338d87143ca120b3aede24d768fd"
MANIFEST_SHA256 = "9f8c971b72ac260b9f48b7f3ab45561d62535c22002d38b53cf4712536f6e81d"
SPLIT_SHA256 = "e95118b4149aa9d7e203f45679fb771a1123b97662dbe3548b9435ca8588b479"
SEEDS = (101, 202, 303, 404, 505)
METHODS = ("nograph", "standard_gat", "epistemic_gat")


@dataclass
class GraphDataset:
    arrays: dict
    schema: dict
    splits: dict
    data_kind: str = "frozen_graph_export"
    scientific_sha256: str = ARTIFACT_SHA256

    def __post_init__(self):
        ge.require(self.scientific_sha256 == ARTIFACT_SHA256, "graph artifact identity refusal")
        ge.require(self.schema["node_order"] == list(ge.NODE_ORDER)
                   and self.schema["feature_order"] == list(ge.FEATURE_ORDER)
                   and self.schema["adjacency"] == ge.ADJACENCY.tolist(), "node/feature/topology contract")
        ge.validate_nodes(self.arrays["node_features"])
        n = len(self.arrays["target_normal"])
        ge.require(self.arrays["node_features"].shape == (n, 3, 3), "graph shape")
        ge.require(set(self.arrays["target_normal"].tolist()) <= {0, 1}, "target_normal semantics")
        ge.require(len(set(self.arrays["row_key"].tolist())) == n, "duplicate row key")
        sid = self.scenarios
        for index, name in enumerate(sid):
            role = ge.frozen_split(str(name), self.splits)
            expected = 0 if role == "GRAPH_TRAIN" else 1
            ge.require(int(self.arrays["split"][index]) == expected, "scenario split mismatch")
        ge.require(set(sid[self.arrays["split"] == 1]) == set(self.splits["GRAPH_VALIDATION"]),
                   "all three frozen validation scenarios are required")
        ge.require(self.data_kind in ("frozen_graph_export", "synthetic_fixture"), "unknown dataset source")
        for array in self.arrays.values():
            array.setflags(write=False)

    @property
    def scenarios(self):
        dictionary = self.schema["scenario_dictionary"]
        return np.array([dictionary[int(i)]["scenario_id"] for i in self.arrays["scenario_index"]])

    @property
    def train_indices(self):
        return np.flatnonzero(self.arrays["split"] == 0)

    @property
    def validation_indices(self):
        return np.flatnonzero(self.arrays["split"] == 1)

    def batch(self, indices, device="cpu"):
        i = np.asarray(indices, dtype=np.int64)
        # Make a new model input. Never lower precision or mutate the scientific source.
        x = torch.tensor(self.arrays["node_features"][i], dtype=torch.float32, device=device)
        e = torch.tensor(self.arrays["node_features"][i, :, 1], dtype=torch.float64, device=device)
        y = torch.tensor(self.arrays["target_normal"][i], dtype=torch.float32, device=device)
        return x, e, y


def load_graph_dataset(artifact, preregistration):
    artifact, seal = Path(artifact), Path(preregistration)
    # No raw-data loader or TEST path exists in this module.
    for name, expected in (("protocol.json", ge.PROTOCOL_SHA256), ("split_manifest.json", SPLIT_SHA256)):
        ge.require(ge.sha256_file(seal / name) == expected, "frozen protocol/split hash refusal")
    for line in (seal / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        ge.require(ge.sha256_file(seal / name) == expected, "sealed preregistration file changed")
    protocol = json.loads((seal / "protocol.json").read_text())
    for relative, expected in protocol["bindings"]["generic_graph_source_sha256"].items():
        ge.require(ge.sha256_file(seal.parents[1] / relative) == expected,
                   "frozen generic graph source hash refusal")
    ge.require(ge.sha256_file(artifact / "manifest.json") == MANIFEST_SHA256, "artifact manifest hash refusal")
    ge.require(ge.sha256_file(artifact / "schema.json") == SCHEMA_SHA256, "artifact schema hash refusal")
    manifest = json.loads((artifact / "manifest.json").read_text())
    schema = json.loads((artifact / "schema.json").read_text())
    for name, record in manifest["files"].items():
        ge.require(ge.sha256_file(artifact / name) == record["sha256"], "artifact file hash refusal: " + name)
    with np.load(artifact / "graphs.npz", allow_pickle=False) as z:
        arrays = dict(z)
    ge.require(ge.scientific_content_hash(arrays, schema) == manifest["artifact_content_sha256"] == ARTIFACT_SHA256,
               "scientific graph hash refusal")
    ge.require(schema["common_provenance"]["protocol_sha256"] == ge.PROTOCOL_SHA256
               and schema["common_provenance"]["split_manifest_sha256"] == SPLIT_SHA256, "source protocol binding")
    ge.validate_artifact(arrays, schema)
    splits = json.loads((seal / "split_manifest.json").read_text())
    ge.require([s["scenario_id"] for s in schema["scenario_dictionary"]] == splits["sorted_FIT_ids"], "exact FIT whitelist")
    return GraphDataset(arrays, schema, splits)


def epoch_batches(dataset, seed, epoch_index, batch_size=256):
    ge.require(batch_size == 256 and epoch_index >= 0, "frozen batching specification")
    order = dataset.train_indices.copy()
    order = order[np.random.Generator(np.random.PCG64(int(seed) + int(epoch_index))).permutation(len(order))]
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def make_smoke_fixture(splits, pairs_per_scenario=16):
    """A small learnable synthetic fixture; never reported as scientific performance."""
    ge.require(1 <= pairs_per_scenario <= 64, "tiny synthetic fixture limit")
    scenarios = sorted([splits["GRAPH_TRAIN"][0]] + splits["GRAPH_VALIDATION"])
    nodes, targets, roles, sid_indices, ticks, pair_ids, row_keys, recipes = ([] for _ in range(8))
    for j, sid in enumerate(scenarios):
        role = int(sid in splits["GRAPH_VALIDATION"])
        for t in range(1, pairs_per_scenario + 1):
            pair = ge.pair_id(sid, t)
            for pseudo in (False, True):
                p = 0.05 if pseudo else 0.95
                nodes.append(np.tile([p, 0.0, 0.02], (3, 1)))
                targets.append(int(not pseudo)); roles.append(role); sid_indices.append(j); ticks.append(t)
                pair_ids.append(pair)
                row_keys.append(__import__("hashlib").sha256(ge.canonical_json({"fixture_pair": pair, "pseudo": pseudo})).hexdigest())
                recipes.append(t % 5 if pseudo else -1)
    arrays = {"node_features": np.array(nodes, dtype=np.float64), "target_normal": np.array(targets, dtype=np.uint8),
              "split": np.array(roles, dtype=np.uint8), "scenario_index": np.array(sid_indices, dtype=np.uint16),
              "tick": np.array(ticks, dtype=np.int32), "pair_id": np.array(pair_ids, dtype="U64"),
              "row_key": np.array(row_keys, dtype="U64"), "recipe_index": np.array(recipes, dtype=np.int8)}
    schema = {"node_order": list(ge.NODE_ORDER), "feature_order": list(ge.FEATURE_ORDER),
              "adjacency": ge.ADJACENCY.tolist(), "scenario_dictionary": [{"scenario_id": s} for s in scenarios],
              "common_provenance": {"protocol_sha256": ge.PROTOCOL_SHA256}}
    return GraphDataset(arrays, schema, splits, data_kind="synthetic_fixture")
