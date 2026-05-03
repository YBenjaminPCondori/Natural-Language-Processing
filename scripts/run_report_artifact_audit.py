"""Generate the strict report.tex artifact audit outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "modules").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root containing pyproject.toml and modules/.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Optional path to the repo root.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modules.report_artifact_audit import build_report_artifact_audit

    summary = build_report_artifact_audit(project_root)
    print(
        json.dumps(
            {
                "ready_for_report_tex": summary["ready_for_report_tex"],
                "overall_readiness_score": summary["overall_readiness_score"],
                "stage_statuses": summary["stage_statuses"],
                "outputs": [
                    "outputs/data_audit.json",
                    "outputs/project_audit_report.md",
                    "outputs/project_audit_summary.json",
                ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
