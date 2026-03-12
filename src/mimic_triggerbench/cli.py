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

    console.print("[red]Unknown command[/red]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

