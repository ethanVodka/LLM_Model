"""このリポジトリのソースを学習用JSONLへ変換するCLI。"""

import argparse
from pathlib import Path

from mini_llm.project_code_import import ProjectCodeConfig, import_project_code


def main() -> None:
    parser = argparse.ArgumentParser(prog="mini-llm-project-code")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/project_code_v1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/project_code_v1.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/data/project_code_v1.json"),
    )
    args = parser.parse_args()
    report = import_project_code(
        ProjectCodeConfig.from_yaml(args.config),
        args.output,
        args.report,
    )
    print(f"dataset={args.output}")
    print(f"records={report['record_count']}")
    print(f"characters={report['character_count']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
