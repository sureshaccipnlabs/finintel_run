from models import EmployeeMetric
from models import KnowledgeBase
from models import PortfolioSummary
from models import ProjectMetric
from models import RankingEntry
from models import RiskInsight


class KnowledgeBuilder:

    def build(self, payloads: list[dict]) -> KnowledgeBase:
        projects: list[ProjectMetric] = []
        employees: list[EmployeeMetric] = []
        risks: list[RiskInsight] = []
        source_files: list[str] = []

        summary_totals = {
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "total_profit": 0.0,
            "avg_margin_pct": 0.0,
            "total_employees": 0,
            "total_actual_hours": 0.0,
            "total_billable_hours": 0.0,
        }
        margin_count = 0

        for payload in payloads:
            source_file = payload.get("__source_file__", "unknown.json")
            source_files.append(source_file)

            for result in payload.get("results", []):
                result_body = result.get("result", {})
                overall_summary = result_body.get("overall_summary", {})

                summary_totals["total_revenue"] += float(overall_summary.get("total_revenue", 0.0))
                summary_totals["total_cost"] += float(overall_summary.get("total_cost", 0.0))
                summary_totals["total_profit"] += float(overall_summary.get("total_profit", 0.0))
                if "avg_margin_pct" in overall_summary:
                    summary_totals["avg_margin_pct"] += float(overall_summary.get("avg_margin_pct", 0.0))
                    margin_count += 1

                for sheet_name, sheet_data in result_body.get("sheets", {}).items():
                    sheet_summary = sheet_data.get("summary", {})
                    summary_totals["total_employees"] += int(sheet_summary.get("total_employees", 0))
                    summary_totals["total_actual_hours"] += float(sheet_summary.get("total_actual_hours", 0.0))
                    summary_totals["total_billable_hours"] += float(sheet_summary.get("total_billable_hours", 0.0))

                    for project_name, project_data in sheet_data.get("projects", {}).items():
                        projects.append(
                            ProjectMetric(
                                project=project_name,
                                revenue=float(project_data.get("revenue", 0.0)),
                                cost=float(project_data.get("cost", 0.0)),
                                profit=float(project_data.get("profit", 0.0)),
                                employee_count=int(project_data.get("employees", 0)),
                                source_sheet=sheet_name,
                                source_file=source_file,
                            )
                        )

                    for employee_data in sheet_data.get("employees", []):
                        employees.append(
                            EmployeeMetric(
                                employee=employee_data.get("employee", "Unknown"),
                                project=employee_data.get("project", "Unknown"),
                                revenue=float(employee_data.get("revenue", 0.0)),
                                cost=float(employee_data.get("cost", 0.0)),
                                profit=float(employee_data.get("profit", 0.0)),
                                margin_pct=float(employee_data.get("margin_pct", 0.0)),
                                utilisation_pct=float(employee_data.get("utilisation_pct", 0.0)),
                                actual_hours=float(employee_data.get("actual_hours", 0.0)),
                                billable_hours=float(employee_data.get("billable_hours", 0.0)),
                                leave_hours=float(employee_data.get("leave_hours", 0.0)),
                                validation_flags=list(employee_data.get("validation_flags", [])),
                                source_sheet=sheet_name,
                                source_file=source_file,
                            )
                        )

                    for risk_data in sheet_data.get("risks", []):
                        risks.append(
                            RiskInsight(
                                employee=risk_data.get("employee", "Unknown"),
                                issue=risk_data.get("issue", "Unknown"),
                                source_sheet=sheet_name,
                                source_file=source_file,
                            )
                        )

        summary = PortfolioSummary(
            total_revenue=summary_totals["total_revenue"],
            total_cost=summary_totals["total_cost"],
            total_profit=summary_totals["total_profit"],
            avg_margin_pct=(summary_totals["avg_margin_pct"] / margin_count) if margin_count else 0.0,
            total_employees=summary_totals["total_employees"],
            total_actual_hours=summary_totals["total_actual_hours"],
            total_billable_hours=summary_totals["total_billable_hours"],
        )

        employee_indexes = self._build_employee_indexes(employees)

        return KnowledgeBase(
            summary=summary,
            projects=projects,
            employees=employees,
            risks=risks,
            rankings=self._compute_rankings(projects=projects, employees=employees),
            metadata={
                "source_files": source_files,
                "employee_index": employee_indexes["employee_index"],
                "employee_token_index": employee_indexes["employee_token_index"],
            },
        )

    def _compute_rankings(
        self,
        projects: list[ProjectMetric],
        employees: list[EmployeeMetric],
    ) -> dict[str, RankingEntry]:
        rankings: dict[str, RankingEntry] = {}

        if employees:
            top_profit_employee = max(employees, key=lambda item: item.profit)
            most_utilized_employee = max(employees, key=lambda item: item.utilisation_pct)
            rankings["top_profit_employee"] = RankingEntry(
                name=top_profit_employee.employee,
                value=top_profit_employee.profit,
            )
            rankings["most_utilized_employee"] = RankingEntry(
                name=most_utilized_employee.employee,
                value=most_utilized_employee.utilisation_pct,
            )

        if projects:
            top_profit_project = max(projects, key=lambda item: item.profit)
            top_margin_project = max(
                projects,
                key=lambda item: (item.profit / item.revenue) if item.revenue else 0.0,
            )
            rankings["top_profit_project"] = RankingEntry(
                name=top_profit_project.project,
                value=top_profit_project.profit,
            )
            rankings["top_margin_project"] = RankingEntry(
                name=top_margin_project.project,
                value=(top_margin_project.profit / top_margin_project.revenue * 100)
                if top_margin_project.revenue else 0.0,
            )

        return rankings

    def _build_employee_indexes(self, employees: list[EmployeeMetric]) -> dict[str, dict[str, list[EmployeeMetric]]]:
        employee_index: dict[str, list[EmployeeMetric]] = {}
        employee_token_index: dict[str, list[EmployeeMetric]] = {}

        for employee in employees:
            normalized_name = self._normalize_text(employee.employee)
            employee_index.setdefault(normalized_name, []).append(employee)

            for token in normalized_name.split():
                if len(token) > 2:
                    employee_token_index.setdefault(token, []).append(employee)

        return {
            "employee_index": employee_index,
            "employee_token_index": employee_token_index,
        }

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower()
        for char in ["'", "?", ".", ",", "!", ":", ";", "(", ")", "-", "_"]:
            normalized = normalized.replace(char, " ")
        return " ".join(normalized.split())
