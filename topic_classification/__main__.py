"""CLI for classifying inline text, files or standard input."""
import argparse
import json
from pathlib import Path
import sys

from . import TopicClassifier


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify a short text's general topic with GLiNER2.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--file", type=Path, help="UTF-8 text file")
    inputs.add_argument("--stdin", action="store_true")
    parser.add_argument("--topics", type=Path, help="JSON object mapping topic labels to descriptions")
    args = parser.parse_args(argv)
    try:
        topics = json.loads(args.topics.read_text(encoding="utf-8")) if args.topics else None
        text = args.text
        if args.file:
            text = args.file.read_text(encoding="utf-8")
        elif args.stdin:
            text = sys.stdin.read()
        result = TopicClassifier(topics=topics).classify(text)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
