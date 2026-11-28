#!/usr/bin/env python3
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

"""Generate Go messages from Router's canonical latency schema."""

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--proto-dir", default=root / "proto", type=Path)
    parser.add_argument("--protoc", default="protoc")
    parser.add_argument("--protoc-gen-go", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.proto_dir.resolve()
    schema = source / "common/protocol/latency.proto"
    fixtures = source / "testdata/wire.json"
    if not schema.is_file() or not fixtures.is_file():
        parser.error(
            "--proto-dir must contain Router's schema and wire fixtures"
        )
    version = subprocess.check_output([args.protoc, "--version"], text=True).strip()
    if version != "libprotoc 25.3":
        parser.error(f"protoc 25.3 is required, found {version}")
    package = "pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                args.protoc,
                f"--proto_path={source}",
                f"--plugin=protoc-gen-go={args.protoc_gen_go.resolve()}",
                f"--go_out={directory}",
                f"--go_opt=module=github.com/llm-d/llm-d-router/{package}",
                "common/protocol/latency.proto",
            ],
            check=True,
        )
        generated = (Path(directory) / "latency.pb.go").read_bytes()
    files = {
        "latency.pb.go": generated,
        "schema.sha256": (
            hashlib.sha256(schema.read_bytes()).hexdigest() + "\n"
        ).encode(),
        "testdata/wire.json": fixtures.read_bytes(),
    }
    for relative, data in files.items():
        destination = root / package / relative
        if args.check:
            if not destination.is_file() or destination.read_bytes() != data:
                parser.exit(
                    1,
                    f"{destination.relative_to(root)} differs from Router's schema or fixtures\n",
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)


if __name__ == "__main__":
    main()
