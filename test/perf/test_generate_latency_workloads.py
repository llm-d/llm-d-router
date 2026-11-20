# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LatencyWorkloadTests(unittest.TestCase):
    def command(self, directory):
        return [
            sys.executable,
            str(Path(__file__).with_name("generate_latency_workloads.py")),
            "--phase",
            "schema",
            "--model",
            "local-model",
            "--tokenizer-path",
            str(directory),
            "--qps",
            "0.5",
            "1",
            "2",
            "--workers",
            "2",
            "--max-concurrency-per-worker",
            "4",
            "--clip-sigma",
            "4",
            "--output-dir",
            str(directory / "configs"),
            *[
                item
                for name in ("k8s", "load-prefix", "pred-before", "pred-after")
                for item in ("--target", f"{name}=http://{name}.invalid")
            ],
        ]

    def test_comparison_targets_use_identical_workload_and_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            subprocess.run(self.command(directory), check=True, capture_output=True)
            output = directory / "configs"
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "prepared-not-run")
            self.assertEqual(
                manifest["inference_perf_commit"],
                "e28d9a0bf5cefa743910b73057b3b686f0a94b98",
            )
            self.assertEqual(
                manifest["inference_perf_source_tree"],
                "84ff8848c53a22f0c30636466feeb904a881e262",
            )
            self.assertEqual(len(manifest["configs"]), 12)
            self.assertEqual(
                manifest["nominal_repeated_session_requests"]["A"], [0, 0, 0]
            )
            self.assertEqual(
                manifest["nominal_repeated_session_requests"]["C"], [0, 0, 0]
            )
            for scenario in ("A", "B", "C"):
                documents = [
                    json.loads(path.read_text())
                    for path in output.glob(f"schema-{scenario}-*.json")
                ]
                self.assertEqual(len(documents), 4)
                self.assertEqual(
                    len({document["server"]["base_url"] for document in documents}), 4
                )
                for document in documents:
                    self.assertEqual(document["data"], documents[0]["data"])
                    self.assertEqual(document["load"], documents[0]["load"])
                    shared = document["data"]["shared_prefix"]
                    self.assertTrue(shared["enable_multi_turn_chat"])
                    self.assertEqual(shared["output_len"]["std_dev"], 300)
                    self.assertEqual(shared["output_len"]["max"], 2200)
                    self.assertEqual(
                        [stage["duration"] for stage in document["load"]["stages"]],
                        [100, 100, 100],
                    )

    def test_invalid_qps_does_not_create_configurations(self):
        for qps in (["2", "1"], ["1", "1"], ["nan"], ["0"], ["0.001"]):
            with self.subTest(qps=qps), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                result = subprocess.run(
                    self.command(directory) + ["--qps", *qps], capture_output=True
                )
                self.assertEqual(result.returncode, 2)
                self.assertFalse((directory / "configs").exists())

    def test_repeated_sessions_are_counted_per_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            subprocess.run(
                self.command(directory) + ["--qps", "1", "61"],
                check=True,
                capture_output=True,
            )
            manifest = json.loads((directory / "configs/manifest.json").read_text())
            self.assertEqual(
                manifest["nominal_repeated_session_requests"]["A"], [0, 100]
            )
            self.assertEqual(
                manifest["nominal_repeated_session_requests"]["C"], [0, 5350]
            )


if __name__ == "__main__":
    unittest.main()
