"""
FinIntel AI - Run Script
========================
Just run:  python run.py

This will test all 5 sample files and show you the results clearly.
You can also test your own file:  python run.py myfile.csv
"""

import sys
import json
import os

# Add current folder to path so Python can find the ingestion module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.ingest import ingest_file

# ── Pretty print helpers ──────────────────────────────────────────────────────

def green(text):  return f"\033[92m{text}\033[0m"
def red(text):    return f"\033[91m{text}\033[0m"
def yellow(text): return f"\033[93m{text}\033[0m"
def bold(text):   return f"\033[1m{text}\033[0m"
def cyan(text):   return f"\033[96m{text}\033[0m"
def dim(text):    return f"\033[2m{text}\033[0m"

def print_separator(char="─", width=60):
    print(dim(char * width))

def print_header(title):
    print()
    print(bold(f"{'═' * 60}"))
    print(bold(f"  {title}"))
    print(bold(f"{'═' * 60}"))

def print_section(title):
    print()
    print(cyan(f"  ▸ {title}"))
    print_separator()

def fmt_money(val):
    if val is None: return dim("n/a")
    return f"${val:,.2f}"

def fmt_pct(val):
    if val is None: return dim("n/a")
    return f"{val:.1f}%"

# ── Main display function ─────────────────────────────────────────────────────

def display_result(filepath, result, show_all_records=False):

    filename = os.path.basename(filepath)
    print_header(f"FILE: {filename}")

    # Errors
    if result["errors"]:
        print()
        for err in result["errors"]:
            if isinstance(err, dict):
                print(red(f"  ⚠  {err.get('type')}: {err.get('details', err.get('detail', ''))}"))
            else:
                print(red(f"  ⚠  {err}"))

    # File info
    print_section("File info")
    fm = result["file_meta"]
    print(f"  Type        : {bold(fm.get('file_type', 'unknown'))}")
    print(f"  Rows found  : {fm.get('row_count', '?')}")
    if fm.get("encoding"):
        print(f"  Encoding    : {fm.get('encoding')}")
    if fm.get("delimiter"):
        print(f"  Delimiter   : {fm.get('delimiter')}")
    if fm.get("sheet"):
        print(f"  Sheet used  : {fm.get('sheet')}")

    # Column mapping
    print_section("Column mapping")
    cm = result["column_mapping"]
    confidence = cm.get("confidence", "?")
    conf_color = green if confidence == "HIGH" else (yellow if confidence == "PARTIAL" else red)
    print(f"  Confidence  : {conf_color(confidence)}")
    print()
    mapped = cm.get("mapped_fields", {})
    for canonical, raw in mapped.items():
        print(f"  {green('✓')}  {raw:<25} → {bold(canonical)}")
    if cm.get("missing_required"):
        for f in cm["missing_required"]:
            print(f"  {red('✗')}  (not found)               → {red(bold(f))} [REQUIRED]")
    if cm.get("missing_optional"):
        for f in cm["missing_optional"]:
            print(f"  {yellow('○')}  (not found)               → {yellow(f)} [optional]")
    if cm.get("unmapped_columns"):
        for col in cm["unmapped_columns"]:
            print(f"  {dim('–')}  {col:<25} → {dim('unmapped (kept as extra_)')}")

    # Summary
    print_section("Summary")
    s = result["summary"]
    print(f"  Total records : {s.get('total_records', 0)}")
    print(f"  Valid records : {green(str(s.get('valid_records', 0)))}")
    flagged = s.get("flagged_records", 0)
    if flagged:
        print(f"  Flagged       : {yellow(str(flagged))}")
    print(f"  Projects      : {', '.join(s.get('projects', [])) or dim('none')}")
    print(f"  Employees     : {', '.join(s.get('employees', [])) or dim('none')}")
    print()
    print(f"  Total revenue : {green(fmt_money(s.get('total_revenue')))}")
    print(f"  Total cost    : {fmt_money(s.get('total_cost'))}")
    print(f"  Total profit  : {green(fmt_money(s.get('total_profit')))}")
    print(f"  Avg margin    : {green(fmt_pct(s.get('avg_margin_pct')))}")

    # Validation flags
    flags = s.get("validation_flag_counts", {})
    if flags:
        print()
        print(f"  Validation flags found:")
        for flag, count in flags.items():
            color = red if "INVALID" in flag or "MISSING_DATE" in flag or "NEGATIVE" in flag else yellow
            print(f"    {color(flag)}: {count} record(s)")

    # Records
    records = result["records"]
    print_section(f"Records ({len(records)} total — showing first 3)")
    for i, rec in enumerate(records[:3]):
        print(f"\n  {bold(f'Record {i+1}:')}")
        fields_to_show = ["employee", "project", "date", "hours",
                          "billing_rate", "cost_rate",
                          "revenue", "cost", "profit", "margin_pct",
                          "quarter_label", "validation_flags", "is_valid"]
        for field in fields_to_show:
            val = rec.get(field)
            if val is None:
                display = dim("null")
            elif field == "validation_flags":
                display = red(str(val)) if val else green("[]")
            elif field == "is_valid":
                display = green("true") if val else red("false")
            elif field in ("revenue", "cost", "profit"):
                display = fmt_money(val) if val is not None else dim("null")
            elif field == "margin_pct":
                display = fmt_pct(val) if val is not None else dim("null")
            else:
                display = str(val)
            print(f"    {field:<18}: {display}")

    if len(records) > 3:
        print(f"\n  {dim(f'... and {len(records) - 3} more records')}")

    # Save to JSON
    out_path = filepath.replace(".csv", "_result.json").replace("sample_data/", "output/")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print()
    print(f"  {green('✓')} Full JSON saved to: {bold(out_path)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    # If user passes a file as argument: python run.py myfile.csv
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"\n{bold('Running FinIntel AI on:')} {filepath}")
        result = ingest_file(filepath)
        display_result(filepath, result)
        return

    # Otherwise run all 5 sample files
    sample_files = [
        ("sample_data/sample_1_consulting_clean.csv",  "Clean standard data"),
        ("sample_data/sample_2_messy_realworld.csv",   "Messy column names + bad data"),
        ("sample_data/sample_3_european_format.csv",   "European format (semicolons + dot-dates)"),
        ("sample_data/sample_4_days_format.csv",       "Days-based timesheet"),
        ("sample_data/sample_5_mixed_quality.csv",     "Mixed quality data"),
    ]

    print(bold("\n  FinIntel AI — Ingestion Engine Test Runner"))
    print(dim("  Testing all 5 sample files...\n"))

    passed = 0
    for rel_path, description in sample_files:
        filepath = os.path.join(base, rel_path)
        print(f"\n{yellow(f'[{description}]')}")
        try:
            result = ingest_file(filepath)
            display_result(filepath, result)
            passed += 1
        except Exception as e:
            print(red(f"  CRASHED: {e}"))

    print()
    print_separator("═")
    print(bold(f"\n  Done! {passed}/{len(sample_files)} files processed successfully."))
    print(dim("  JSON output files saved in: finintel_run/output/"))
    print(dim("  To test your own file: python run.py yourfile.csv\n"))


if __name__ == "__main__":
    main()
