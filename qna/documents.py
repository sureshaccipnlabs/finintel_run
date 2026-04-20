from langchain_core.documents import Document

from models import KnowledgeBase


class DocumentFactory:

    def build(self, knowledge: KnowledgeBase) -> list[Document]:
        documents: list[Document] = []

        documents.append(
            Document(
                page_content=(
                    "Portfolio Summary\n\n"
                    f"Total Revenue: {knowledge.summary.total_revenue}\n"
                    f"Total Cost: {knowledge.summary.total_cost}\n"
                    f"Total Profit: {knowledge.summary.total_profit}\n"
                    f"Average Margin: {round(knowledge.summary.avg_margin_pct, 2)}\n"
                    f"Total Employees: {knowledge.summary.total_employees}\n"
                    f"Total Actual Hours: {knowledge.summary.total_actual_hours}\n"
                    f"Total Billable Hours: {knowledge.summary.total_billable_hours}\n"
                ),
                metadata={"type": "summary"},
            )
        )

        for project in knowledge.projects:
            margin_pct = round((project.profit / project.revenue * 100), 2) if project.revenue else 0.0
            documents.append(
                Document(
                    page_content=(
                        "Project Metrics\n\n"
                        f"Project Name: {project.project}\n"
                        f"Revenue: {project.revenue}\n"
                        f"Cost: {project.cost}\n"
                        f"Profit: {project.profit}\n"
                        f"Margin Percentage: {margin_pct}\n"
                        f"Employee Count: {project.employee_count}\n"
                        f"Sheet: {project.source_sheet}\n"
                        f"Source File: {project.source_file}\n"
                    ),
                    metadata={
                        "type": "project",
                        "project": project.project,
                        "profit": project.profit,
                        "source_sheet": project.source_sheet,
                    },
                )
            )

        for employee in knowledge.employees:
            documents.append(
                Document(
                    page_content=(
                        "Employee Performance\n\n"
                        f"Employee: {employee.employee}\n"
                        f"Project: {employee.project}\n"
                        f"Revenue: {employee.revenue}\n"
                        f"Cost: {employee.cost}\n"
                        f"Profit: {employee.profit}\n"
                        f"Margin: {employee.margin_pct}\n"
                        f"Utilization: {employee.utilisation_pct}\n"
                        f"Actual Hours: {employee.actual_hours}\n"
                        f"Billable Hours: {employee.billable_hours}\n"
                        f"Leave Hours: {employee.leave_hours}\n"
                        f"Validation Flags: {employee.validation_flags}\n"
                        f"Sheet: {employee.source_sheet}\n"
                        f"Source File: {employee.source_file}\n"
                    ),
                    metadata={
                        "type": "employee",
                        "employee": employee.employee,
                        "project": employee.project,
                        "profit": employee.profit,
                    },
                )
            )

        for risk in knowledge.risks:
            documents.append(
                Document(
                    page_content=(
                        "Risk Insight\n\n"
                        f"Employee: {risk.employee}\n"
                        f"Issue: {risk.issue}\n"
                        f"Sheet: {risk.source_sheet}\n"
                        f"Source File: {risk.source_file}\n"
                    ),
                    metadata={"type": "risk", "employee": risk.employee},
                )
            )

        for ranking_name, ranking in knowledge.rankings.items():
            documents.append(
                Document(
                    page_content=(
                        "Ranking Insight\n\n"
                        f"Ranking: {ranking_name}\n"
                        f"Name: {ranking.name}\n"
                        f"Value: {ranking.value}\n"
                    ),
                    metadata={"type": "ranking", "ranking": ranking_name},
                )
            )

        return documents
