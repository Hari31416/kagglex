"""Runner entrypoint inside demo_module."""

import json
from pathlib import Path
import sys

from demo_module import __version__


def main() -> int:
    print(f"Executing inside demo_module version {__version__}")
    print(f"Python executable: {sys.executable}")

    # Verify data bundled via --include-data
    data_file = Path("data/input_sample.json")
    if not data_file.exists():
        # Also check /kaggle/working/data/input_sample.json
        data_file = Path("/kaggle/working/data/input_sample.json")

    print(f"Checking for bundled data file at: {data_file}")
    if not data_file.exists():
        print(f"ERROR: Expected data file not found at {data_file}")
        return 1

    content = json.loads(data_file.read_text(encoding="utf-8"))
    print(f"Successfully loaded bundled data: {content}")

    # Write output
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "module_output.json"
    result = {
        "module_version": __version__,
        "input_received": content,
        "status": "SUCCESS",
    }
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved output to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
