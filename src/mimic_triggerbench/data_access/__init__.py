"""
Data access layer for local offline MIMIC-IV and optional MIMIC-IV-Note data.

Phase 1 exposes:
- generate_inventory_report(...) – presence of required raw tables
- load_table_dataframe(...) – minimal helper to load a raw table as pandas
"""

from .inventory import generate_inventory_report
from .tables import load_table_dataframe

__all__ = ["generate_inventory_report", "load_table_dataframe"]

