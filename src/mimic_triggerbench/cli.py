from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from mimic_triggerbench.config import load_settings
from mimic_triggerbench.data_access.inventory import generate_inventory_report


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

    console.print("[red]Unknown command[/red]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

