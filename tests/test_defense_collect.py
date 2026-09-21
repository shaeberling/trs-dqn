import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl.defense_collect import collect
from rl.defense_learning import sha256


class DefenseCollectorTests(unittest.TestCase):
    def bundle(self, root, score):
        bundle = root/"versions"/"frozen"
        bundle.mkdir(parents=True)
        game = dict(seed=10000, terminated=True, truncated=False, score=score,
                    missions_completed=0, highest_stage=1)
        (bundle/"model.safetensors").write_bytes(b"frozen weights")
        (bundle/"state.json").write_text("{}")
        evaluation = dict(games=[game], incomplete_games=0)
        (bundle/"evaluation.json").write_text(json.dumps(evaluation))
        (bundle/"verification.json").write_text(json.dumps(dict(
            verified=True, checkpoint_sha256=sha256(bundle/"model.safetensors"))))
        manifest = dict(rank=[0, 1, score], result=game,
                        hashes={p.name: sha256(p) for p in bundle.iterdir()})
        (bundle/"manifest.json").write_text(json.dumps(manifest))
        (root/"best").symlink_to(Path("versions")/"frozen")
        return bundle, evaluation

    def test_missing_first_validation_is_expected(self):
        with tempfile.TemporaryDirectory() as tmp, patch("rl.defense_collect.publish_best") as publish:
            self.assertIsNone(collect([Path(tmp)/"not-ready"], Path(tmp)/"out"))
            publish.assert_not_called()

    def test_strongest_immutable_source_is_reverified(self):
        with tempfile.TemporaryDirectory() as tmp, patch("rl.defense_collect.publish_best") as publish:
            root = Path(tmp)
            self.bundle(root/"one", 320)
            stronger, evaluation = self.bundle(root/"two", 400)
            publish.return_value = None
            collect([root/"one", root/"two"], root/"out")
            publish.assert_called_once_with(stronger.resolve()/"model.safetensors", evaluation,
                                            root/"out", log=None)

    def test_corrupt_bundle_never_reaches_publisher(self):
        with tempfile.TemporaryDirectory() as tmp, patch("rl.defense_collect.publish_best") as publish:
            root = Path(tmp)
            bundle, _ = self.bundle(root/"one", 400)
            (bundle/"model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                collect([root/"one"], root/"out")
            publish.assert_not_called()

    def test_sampling_probe_is_not_a_training_promotion(self):
        with tempfile.TemporaryDirectory() as tmp, patch("rl.defense_collect.publish_best") as publish:
            root = Path(tmp)
            bundle, _ = self.bundle(root/"one", 400)
            verification = json.loads((bundle/"verification.json").read_text())
            verification["temperature"] = .5
            (bundle/"verification.json").write_text(json.dumps(verification))
            manifest = json.loads((bundle/"manifest.json").read_text())
            manifest["hashes"]["verification.json"] = sha256(bundle/"verification.json")
            (bundle/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "original-policy"):
                collect([root/"one"], root/"out")
            publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
