from openpyxl import load_workbook
from datetime import datetime, date

HOURS_PER_DAY = 8.0


def clean(v):
    return str(v).strip() if v else ""


def is_date(v):
    return isinstance(v, (datetime, date))


def compute_financials(emp):
    billing_rate = emp.get("billing_rate")
    cost_rate = emp.get("cost_rate")

    actual_hours = emp.get("actual_hours", 0)
    billable_hours = emp.get("billable_hours", actual_hours)

    vacation_days = emp.get("vacation_days", 0)
    leave_hours = vacation_days * HOURS_PER_DAY

    revenue = billable_hours * billing_rate if billing_rate else None
    cost = (actual_hours + leave_hours) * cost_rate if cost_rate else None
    profit = revenue - cost if revenue and cost else None

    return {
        **emp,
        "revenue": revenue,
        "cost": cost,
        "profit": profit
    }


def parse_timesheet(filepath):
    wb = load_workbook(filepath, data_only=True)

    result = {
        "file": filepath,
        "sheets": {}
    }

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        rows = [list(r) for r in ws.iter_rows(values_only=True)]

        employees = {}

        # 🔥 STEP 1: Find header row (Proofpoint style)
        header_idx = None

        for i, row in enumerate(rows[:20]):
            row_vals = [clean(v).lower() for v in row]

            if "name" in row_vals and "project" in row_vals:
                header_idx = i
                break

        if header_idx is None:
            continue

        header = rows[header_idx]

        # 🔥 STEP 2: Identify columns
        name_idx = 0
        project_idx = 1

        date_cols = [
            j for j, v in enumerate(header)
            if is_date(v)
        ]

        # 🔥 STEP 3: Read employee rows
        for r in rows[header_idx + 2: header_idx + 50]:

            name = clean(r[name_idx])
            if not name:
                continue

            project = clean(r[project_idx])

            total_hours = 0

            for j in date_cols:
                if j < len(r) and r[j]:
                    try:
                        total_hours += float(r[j])
                    except:
                        pass

            if total_hours == 0:
                continue

            employees[name] = {
                "employee": name,
                "project": project,
                "actual_hours": total_hours,
                "billable_hours": total_hours,
                "working_days": round(total_hours / 8, 2),
                "vacation_days": 0,
                "billing_rate": 150,  # TEMP
                "cost_rate": 80       # TEMP
            }

        # 🔥 STEP 4: Financials
        enriched = [compute_financials(e) for e in employees.values()]

        result["sheets"][sheet_name] = {
            "template": "proofpoint",
            "employees": enriched
        }

    return result