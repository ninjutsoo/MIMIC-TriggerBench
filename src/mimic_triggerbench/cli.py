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

    ep = sub.add_parser(
        "generate-episodes",
        help="Generate benchmark episodes from canonical timelines (Phase 5).",
    )
    ep.add_argument("--dotenv", default=".env", help="Path to .env file (default: .env).")
    ep.add_argument(
        "--timelines",
        default="output/timelines",
        help="Path to timeline Parquet file or directory (default: output/timelines).",
    )
    ep.add_argument(
        "--out",
        default="output/episodes",
        help="Output directory for episode files (default: output/episodes).",
    )
    ep.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names to generate episodes for (default: all).",
    )

    sp = sub.add_parser(
        "split-episodes",
        help="Apply deterministic patient-level train/val/test splits to episodes (Phase 6).",
    )
    sp.add_argument(
        "--episodes-dir",
        default="output/episodes",
        help="Directory containing episode Parquet files (default: output/episodes).",
    )
    sp.add_argument(
        "--out",
        default="output/splits",
        help="Output directory for split manifests and stats (default: output/splits).",
    )
    sp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42).",
    )
    sp.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of patients for train (default: 0.7).",
    )
    sp.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of patients for validation (default: 0.15).",
    )

    vs = sub.add_parser(
        "validate-schema",
        help="Validate a JSON file against the frozen BenchmarkOutput schema (Phase 6.5).",
    )
    vs.add_argument(
        "json_file",
        help="Path to a JSON file containing a BenchmarkOutput payload (or JSON-lines of payloads).",
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

    if args.command == "generate-episodes":
        try:
            import json as _json

            from mimic_triggerbench.labeling import (
                load_all_task_specs,
                generate_all_episodes,
                episodes_to_records,
            )
            from mimic_triggerbench.timeline.io import read_timeline_parquet
            import pandas as _pd

            timelines = read_timeline_parquet(Path(args.timelines))
            specs = load_all_task_specs()

            task_names = list(specs.keys())
            if args.tasks:
                task_names = [t.strip() for t in args.tasks.split(",")]

            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)

            for task_name in task_names:
                spec = specs[task_name]
                episodes = generate_all_episodes(spec, timelines)
                records = episodes_to_records(episodes)
                if records:
                    df = _pd.DataFrame.from_records(records)
                    pq_path = out_dir / f"episodes_{task_name}.parquet"
                    df.to_parquet(str(pq_path), index=False)
                    jsonl_path = out_dir / f"episodes_{task_name}.jsonl"
                    with jsonl_path.open("w", encoding="utf-8") as f:
                        for r in records:
                            f.write(_json.dumps(r, default=str) + "\n")
                    console.print(
                        f"  {task_name}: {len(episodes)} episodes "
                        f"({sum(1 for e in episodes if e.trigger_label)} pos, "
                        f"{sum(1 for e in episodes if not e.trigger_label)} neg)"
                    )
                else:
                    console.print(f"  {task_name}: 0 episodes")
            console.print(f"[green]Episodes written to[/green] {out_dir}")
            return 0
        except Exception as e:  # noqa: BLE001 - CLI boundary
            console.print(f"[red]Episode generation failed:[/red] {e}")
            return 2

    if args.command == "split-episodes":
        try:
            from mimic_triggerbench.splitting import (
                split_episodes_from_dir,
                write_split_manifests,
                write_split_stats,
            )

            out_dir = Path(args.out)
            episodes_dir = Path(args.episodes_dir)
            split_result = split_episodes_from_dir(
                episodes_dir,
                seed=args.seed,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
            )
            write_split_manifests(split_result, out_dir)
            write_split_stats(split_result, out_dir)
            for task, counts in split_result.per_task_counts.items():
                console.print(f"  {task}: {counts}")
            console.print(f"[green]Split manifests written to[/green] {out_dir}")
            return 0
        except Exception as e:  # noqa: BLE001 - CLI boundary
            console.print(f"[red]Split generation failed:[/red] {e}")
            return 2

    if args.command == "validate-schema":
        try:
            import json as _json

            from mimic_triggerbench.schemas import validate_benchmark_output

            json_path = Path(args.json_file)
            text = json_path.read_text(encoding="utf-8")
            payloads: list[dict] = []
            if text.strip().startswith("["):
                payloads = _json.loads(text)
            elif text.strip().startswith("{"):
                payloads = [_json.loads(text)]
            else:
                for line in text.strip().splitlines():
                    stripped = line.strip()
                    if stripped:
                        payloads.append(_json.loads(stripped))

            errors = 0
            for i, payload in enumerate(payloads):
                try:
                    validate_benchmark_output(payload)
                except Exception as ve:
                    console.print(f"[red]Payload {i} invalid:[/red] {ve}")
                    errors += 1

            if errors:
                console.print(f"[red]{errors}/{len(payloads)} payloads failed validation.[/red]")
                return 2
            console.print(f"[green]All {len(payloads)} payloads valid.[/green]")
            return 0
        except Exception as e:  # noqa: BLE001 - CLI boundary
            console.print(f"[red]Schema validation failed:[/red] {e}")
            return 2

    console.print("[red]Unknown command[/red]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
