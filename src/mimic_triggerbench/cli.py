from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from mimic_triggerbench.config import load_settings
from mimic_triggerbench.data_access.inventory import generate_inventory_report
from mimic_triggerbench.data_access.normalization_audit import (
    scan_normalization_coverage,
    write_normalization_coverage_report,
)
from mimic_triggerbench.data_access import (
    load_all_codebooks,
    load_mapping_ledger,
    mapping_ledger_path,
    reconcile_mapping_ledger,
)
from mimic_triggerbench.timeline import (
    build_all_timelines,
    write_timeline_parquet,
)
from mimic_triggerbench.feasibility import (
    run_feasibility_checkpoint,
    write_feasibility_reports,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mimic-triggerbench")
    sub = p.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="Scan for required MIMIC-IV tables and write an inventory report.")
    inv.add_argument(
        "--dotenv",
        default=".env",
        help="Path to .env file (default: .env).",
    )
    inv.add_argument(
        "--out",
        default="docs/data_inventory_generated.md",
        help="Output markdown report path (default: docs/data_inventory_generated.md).",
    )

    aud = sub.add_parser(
        "normalization-audit",
        help="Scan local MIMIC tables and write normalization coverage/unmapped-term report.",
    )
    aud.add_argument("--dotenv", default=".env", help="Path to .env file (default: .env).")
    aud.add_argument(
        "--out",
        default="docs/normalization_coverage_generated.md",
        help="Output markdown report path (default: docs/normalization_coverage_generated.md).",
    )
    aud.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Max rows to scan per table (default: 200000).",
    )
    aud.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K unmapped itemids/labels to list per table (default: 50).",
    )

    ledger = sub.add_parser(
        "mapping-ledger-check",
        help="Validate mapping ledger schema and reconcile mapped rows with runtime codebooks.",
    )
    ledger.add_argument(
        "--ledger",
        default=str(mapping_ledger_path()),
        help="Path to mapping_ledger.csv (default: project mapping ledger).",
    )

    tl = sub.add_parser(
        "build-timeline",
        help="Build canonical timelines from normalized MIMIC-IV events (Phase 4).",
    )
    tl.add_argument("--dotenv", default=".env", help="Path to .env file (default: .env).")
    tl.add_argument(
        "--out",
        default="output/timelines",
        help="Output directory for timeline Parquet files (default: output/timelines).",
    )
    tl.add_argument(
        "--max-stays",
        type=int,
        default=None,
        help="Limit number of ICU stays to process (default: all).",
    )
    tl.add_argument(
        "--stay-ids",
        type=str,
        default=None,
        help="Comma-separated list of stay_ids to process.",
    )
    tl.add_argument(
        "--partition",
        action="store_true",
        default=False,
        help="Partition output Parquet by stay_id.",
    )

    fc = sub.add_parser(
        "feasibility-check",
        help="Run action extraction feasibility checkpoint (Phase 3.5).",
    )
    fc.add_argument("--dotenv", default=".env", help="Path to .env file (default: .env).")
    fc.add_argument(
        "--out",
        default="output/feasibility",
        help="Output directory for feasibility reports (default: output/feasibility).",
    )
    fc.add_argument(
        "--max-stays",
        type=int,
        default=None,
        help="Limit number of ICU stays to process (default: all).",
    )
    fc.add_argument(
        "--review-n",
        type=int,
        default=25,
        help="Number of detections to sample per action family for review (default: 25).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    console = Console()
    args = build_parser().parse_args(argv)

    if args.command == "inventory":
        try:
            settings = load_settings(args.dotenv)
            out_path = Path(args.out)
            generate_inventory_report(settings, out_path)
            console.print(f"[green]Wrote inventory report to[/green] {out_path}")
            return 0
        except Exception as e:  # noqa: BLE001 - CLI boundary
            console.print(f"[red]Inventory failed:[/red] {e}")
            return 2

    if args.command == "normalization-audit":
        try:
            settings = load_settings(args.dotenv)
            out_path = Path(args.out)
            results = scan_normalization_coverage(
                settings,
                max_rows_per_table=args.max_rows,
                top_k=args.top_k,
            )
            write_normalization_coverage_report(
                results,
                out_path,
                settings=settings,
                max_rows_per_table=args.max_rows,
                top_k=args.top_k,
            )
            console.print(f"[green]Wrote normalization audit report to[/green] {out_path}")
            return 0
        except Exception as e:  # noqa: BLE001 - CLI boundary
            console.print(f"[red]Normalization audit failed:[/red] {e}")
            return 2

    if args.command == "mapping-ledger-check":
        try:
            ledger_rows = load_mapping_ledger(Path(args.ledger))
            codebooks = load_all_codebooks()
            rec = reconcile_mapping_ledger(codebooks, ledger_rows)
            console.print(f"Ledger rows: {rec.total_rows}")
            for decision, count in rec.decision_counts.items():
                console.print(f"- {decision}: {count}")
            if rec.ok:
                console.print("[green]Mapping ledger reconciliation passed.[/green]")
                return 0
            console.print("[red]Mapping ledger reconciliation failed.[/red]")
            for issue in rec.issue_messages:
                console.print(f"- {issue}")
            return 2
        except Exception as e:  # noqa: BLE001 - CLI boundary
            console.print(f"[red]Mapping ledger check failed:[/red] {e}")
            return 2

    if args.command == "build-timeline":
        try:
            settings = load_settings(args.dotenv)
            out_dir = Path(args.out)
            stay_ids = None
            if args.stay_ids:
                stay_ids = [int(s.strip()) for s in args.stay_ids.split(",")]
            timelines, stats = build_all_timelines(
                settings,
                stay_ids=stay_ids,
                max_stays=args.max_stays,
            )
            partition_cols = ["stay_id"] if args.partition else None
            write_timeline_parquet(timelines, out_dir, partition_cols=partition_cols)
            console.print(f"[green]Built timelines:[/green] {stats.summary()}")
            console.print(f"[green]Output written to[/green] {out_dir}")
            return 0
        except Exception as e:  # noqa: BLE001 - CLI boundary
            console.print(f"[red]Timeline build failed:[/red] {e}")
            return 2

    if args.command == "feasibility-check":
        try:
            settings = load_settings(args.dotenv)
            out_dir = Path(args.out)
            timelines, _stats = build_all_timelines(
                settings, max_stays=args.max_stays,
            )
            decisions, review_sets = run_feasibility_checkpoint(
                timelines, review_sample_n=args.review_n,
            )
            write_feasibility_reports(decisions, review_sets, out_dir)
            for d in decisions:
                gate = "[green]PASS[/green]" if d.passed else "[red]FAIL[/red]"
                console.print(
                    f"  {d.action_family}: {d.stats.detected_events} events, "
                    f"{d.stats.unique_stays} stays → {gate}"
                )
            console.print(f"[green]Reports written to[/green] {out_dir}")
            return 0
        except Exception as e:  # noqa: BLE001 - CLI boundary
            console.print(f"[red]Feasibility check failed:[/red] {e}")
            return 2

    console.print("[red]Unknown command[/red]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
