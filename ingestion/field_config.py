"""
Field alias configuration for FinIntel AI.
Add new aliases here as you encounter new file formats — no code changes needed elsewhere.
"""

# Required fields and every known alias variation (all lowercase)
FIELD_ALIASES = {
    "employee": [
        "employee", "emp", "emp name", "employee name", "staff", "worker",
        "name", "consultant", "resource", "person", "team member", "associate",
        "full name", "staff name", "personnel", "user", "username"
    ],
    "project": [
        "project", "proj", "project name", "client", "account", "engagement",
        "matter", "job", "task", "work order", "project code", "proj name",
        "client name", "account name", "contract", "deal", "campaign"
    ],
    "date": [
        "date", "day", "work date", "entry date", "timesheet date", "period",
        "week ending", "week end", "billing date", "log date", "transaction date",
        "invoice date", "service date", "shift date"
    ],
    "hours": [
        "hours", "hrs", "hours worked", "time", "duration", "billable hours",
        "total hours", "time spent", "effort", "hours billed", "logged hours",
        "time logged", "quantity", "units", "days"  # days will be auto-converted ×8
    ],
    "billing_rate": [
        "billing_rate", "billing rate", "bill rate", "rate", "billable rate",
        "charge rate", "sell rate", "price", "hourly rate", "rate per hour",
        "client rate", "invoice rate", "br", "standard rate", "list rate"
    ],
    "cost_rate": [
        "cost_rate", "cost rate", "cost", "cr", "hourly cost", "internal rate",
        "pay rate", "salary rate", "loaded rate", "fully loaded rate",
        "employee cost", "resource cost", "blended rate", "overhead rate"
    ],
}

# Fields that are optional (won't fail validation if missing)
OPTIONAL_FIELDS = {"billing_rate", "cost_rate"}

# Required fields (will raise warning but still process)
REQUIRED_FIELDS = {"employee", "project", "date", "hours"}

# Date formats to try in order
DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y",
    "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y", "%Y.%m.%d",
    "%d %b %Y", "%b %d %Y", "%d %B %Y", "%B %d %Y",
    "%d-%b-%Y", "%b-%d-%Y", "%Y%m%d",
]
