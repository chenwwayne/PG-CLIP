import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from protocol import assert_target_data_isolation
from scripts.run_pgclip_protocol import ensure_shared_text_checkpoint, sha256


def write_metadata(path: Path, image_path: str) -> None:
    path.write_text(
        json.dumps({
            "image_path": image_path,
            "mask_path": "",
            "class_name": "panel",
            "label": 0,
        }) + "\n",
        encoding="utf-8",
    )


class ProtocolTest(unittest.TestCase):
    def test_shared_checkpoint_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "generic" / "text_adapter.pth"
            destination = root / "pg" / "text_adapter.pth"
            source.parent.mkdir()
            source.write_bytes(b"generic-text-checkpoint")
            digest = ensure_shared_text_checkpoint(source, destination)
            self.assertEqual(digest, sha256(destination))

            destination.write_bytes(b"different")
            with self.assertRaisesRegex(RuntimeError, "must reuse"):
                ensure_shared_text_checkpoint(source, destination)

    def test_target_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            target = root / "target.jsonl"
            write_metadata(source, "images/normal/a.png")
            write_metadata(target, "images/normal/a.png")
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                assert_target_data_isolation(source, target, root, root)

    def test_distinct_domain_roots_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            target = root / "target.jsonl"
            write_metadata(source, "images/normal/a.png")
            write_metadata(target, "images/normal/a.png")
            assert_target_data_isolation(source, target, root / "A", root / "B")

    def test_target_metadata_is_used_only_by_evaluation_commands(self):
        project = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(project / "scripts" / "run_pgclip_protocol.py"),
                "--data-root", str(project / "data"),
                "--direction", "active_to_cell",
                "--seed", "0", "--shot", "2", "--dry-run",
            ],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        )
        commands = [line for line in completed.stdout.splitlines() if line.startswith("$")]
        adaptation = [line for line in commands if "train.py" in line or "learn_apsf_weight.py" in line]
        evaluation = [line for line in commands if "test.py" in line]
        self.assertTrue(adaptation)
        self.assertTrue(evaluation)
        self.assertTrue(all("target_test.jsonl" not in line for line in adaptation))
        self.assertTrue(all("target_test.jsonl" in line for line in evaluation))


if __name__ == "__main__":
    unittest.main()
