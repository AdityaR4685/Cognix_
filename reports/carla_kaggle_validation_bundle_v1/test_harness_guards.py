"""Engineering safeguards only; never imports/trains the scientific models."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("validation_guards", HERE / "validate_kaggle.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class HashGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = self.root / "source.py"
        self.payload.write_bytes(b"frozen source\n")
        self.record = hashlib.sha256(self.payload.read_bytes()).hexdigest()
        manifest = {"files": {"source.py": self.record}, "frozen_lock": {
            "trainer_identity": {"bundle_sha256": "9012174611be989f5305f5cbc40f05527831c9f704e49d8823c52fed88ccea3b"},
            "expected_hashes": {"protocol": "68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203",
            "graph_scientific": "baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237"}}}
        (self.root / "bundle_manifest.json").write_text(json.dumps(manifest))
        self.seal()

    def seal(self, extra=""):
        text = self.record + "  source.py\n" + harness.digest(self.root / "bundle_manifest.json") + "  bundle_manifest.json\n" + extra
        (self.root / "SHA256SUMS").write_text(text)
        (self.root / "SHA256SUMS.sha256").write_text(harness.digest(self.root / "SHA256SUMS") + "  SHA256SUMS\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_seal(self):
        manifest, seal, records = harness.verify_files(self.root)
        self.assertEqual(records["source.py"], self.record)

    def test_changed_source_refused(self):
        self.payload.write_bytes(b"changed\n")
        with self.assertRaisesRegex(RuntimeError, "file mismatch"):
            harness.verify_files(self.root)

    def test_changed_checksum_list_refused(self):
        with (self.root / "SHA256SUMS").open("a") as stream:
            stream.write("corrupt\n")
        with self.assertRaisesRegex(RuntimeError, "seal mismatch"):
            harness.verify_files(self.root)

    def test_missing_source_refused(self):
        self.payload.unlink()
        with self.assertRaisesRegex(RuntimeError, "file mismatch"):
            harness.verify_files(self.root)

    def test_path_escape_refused(self):
        self.seal("0" * 64 + "  ../outside.py\n")
        with self.assertRaisesRegex(RuntimeError, "unsafe manifest"):
            harness.verify_files(self.root)

    def test_changed_manifest_records_refused(self):
        manifest = json.loads((self.root / "bundle_manifest.json").read_text())
        manifest["files"]["source.py"] = "0" * 64
        (self.root / "bundle_manifest.json").write_text(json.dumps(manifest))
        self.seal()
        with self.assertRaisesRegex(RuntimeError, "manifest records differ"):
            harness.verify_files(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
