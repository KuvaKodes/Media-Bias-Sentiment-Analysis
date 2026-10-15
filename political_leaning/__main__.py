"""Command-line interface: inline text, UTF-8 file, or standard input."""

import argparse
import json
from pathlib import Path
import sys

from . import PoliticalLeaningClassifier


def main(argv=None):
    parser = argparse.ArgumentParser(description="Grade a passage's political leaning.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Passage to classify")
    source.add_argument("--file", type=Path, help="UTF-8 text file")
    source.add_argument("--stdin", action="store_true", help="Read standard input")
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument("--truncate", action="store_true", help="Allow truncation to 512 tokens")
    args = parser.parse_args(argv)
    try:
        text = args.text
        if args.file is not None:
            text = args.file.read_text(encoding="utf-8")
        elif args.stdin:
            text = sys.stdin.read()
        result = PoliticalLeaningClassifier(device=args.device).grade(
            text, truncate=args.truncate
        )
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
