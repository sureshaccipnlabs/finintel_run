from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PortfolioSummary:
    total_revenue: float = 0.0
    total_cost: float = 0.0
    total_profit: float = 0.0
    avg_margin_pct: float = 0.0
    total_employees: int = 0
    total_actual_hours: float = 0.0
    total_billable_hours: float = 0.0


@dataclass(slots=True)
class ProjectMetric:
    project: str
    revenue: float
    cost: float
    profit: float
    employee_count: int
    source_sheet: str
    source_file: str


@dataclass(slots=True)
class EmployeeMetric:
    employee: str
    project: str
    revenue: float
    cost: float
    profit: float
    margin_pct: float
    utilisation_pct: float
    actual_hours: float
    billable_hours: float
    leave_hours: float
    validation_flags: list[str] = field(default_factory=list)
    source_sheet: str = ""
    source_file: str = ""


@dataclass(slots=True)
class RiskInsight:
    employee: str
    issue: str
    source_sheet: str
    source_file: str


@dataclass(slots=True)
class RankingEntry:
    name: str
    value: float


@dataclass(slots=True)
class KnowledgeBase:
    summary: PortfolioSummary
    projects: list[ProjectMetric]
    employees: list[EmployeeMetric]
    risks: list[RiskInsight]
    rankings: dict[str, RankingEntry]
    metadata: dict[str, Any] = field(default_factory=dict)
