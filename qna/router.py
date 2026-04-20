from models import KnowledgeBase


class RuleBasedRouter:

    def answer(self, question: str, knowledge: KnowledgeBase) -> str | None:
        normalized = question.lower()
        rankings = knowledge.rankings
        employee_metric_answer = self._answer_employee_metric(question=question, knowledge=knowledge)
        if employee_metric_answer:
            return employee_metric_answer

        if "highest profit" in normalized and "employee" in normalized:
            ranking = rankings.get("top_profit_employee")
            if ranking:
                return (
                    "Top employee by profit\n\n"
                    f"Employee: {ranking.name}\n"
                    f"Profit: {ranking.value}"
                )

        if "highest profit" in normalized and "project" in normalized:
            ranking = rankings.get("top_profit_project")
            if ranking:
                return (
                    "Top project by profit\n\n"
                    f"Project: {ranking.name}\n"
                    f"Profit: {ranking.value}"
                )

        if "highest margin" in normalized or "top margin project" in normalized:
            ranking = rankings.get("top_margin_project")
            if ranking:
                return (
                    "Top project by margin\n\n"
                    f"Project: {ranking.name}\n"
                    f"Margin: {round(ranking.value, 2)}%"
                )

        if "most utilized" in normalized or "highest utilization" in normalized:
            ranking = rankings.get("most_utilized_employee")
            if ranking:
                return (
                    "Most utilized employee\n\n"
                    f"Employee: {ranking.name}\n"
                    f"Utilization: {round(ranking.value, 2)}%"
                )

        if "portfolio summary" in normalized or "overall summary" in normalized:
            summary = knowledge.summary
            return (
                "Portfolio Summary\n\n"
                f"Total Revenue: {summary.total_revenue}\n"
                f"Total Cost: {summary.total_cost}\n"
                f"Total Profit: {summary.total_profit}\n"
                f"Average Margin: {round(summary.avg_margin_pct, 2)}%"
            )

        return None

    def _answer_employee_metric(self, question: str, knowledge: KnowledgeBase) -> str | None:
        normalized = self._normalize_text(question)
        employee = self._find_employee(question=question, knowledge=knowledge)
        if employee is None:
            return None

        if any(term in normalized for term in ["margin pct", "margin percentage", "margin percent", "profit percentage", "profit pct", "margin"]):
            return (
                "Employee Metric\n\n"
                f"Employee: {employee.employee}\n"
                f"Margin Percentage: {round(employee.margin_pct, 2)}%"
            )

        if "utilization" in normalized or "utilisation" in normalized or "utilized" in normalized or "utilised" in normalized:
            return (
                "Employee Metric\n\n"
                f"Employee: {employee.employee}\n"
                f"Utilization: {round(employee.utilisation_pct, 2)}%"
            )

        if "profit" in normalized:
            return (
                "Employee Metric\n\n"
                f"Employee: {employee.employee}\n"
                f"Profit: {employee.profit}"
            )

        if "revenue" in normalized:
            return (
                "Employee Metric\n\n"
                f"Employee: {employee.employee}\n"
                f"Revenue: {employee.revenue}"
            )

        if "cost" in normalized:
            return (
                "Employee Metric\n\n"
                f"Employee: {employee.employee}\n"
                f"Cost: {employee.cost}"
            )

        if "billable" in normalized and "hour" in normalized:
            return (
                "Employee Metric\n\n"
                f"Employee: {employee.employee}\n"
                f"Billable Hours: {employee.billable_hours}"
            )

        if "actual" in normalized and "hour" in normalized:
            return (
                "Employee Metric\n\n"
                f"Employee: {employee.employee}\n"
                f"Actual Hours: {employee.actual_hours}"
            )

        if "leave" in normalized and "hour" in normalized:
            return (
                "Employee Metric\n\n"
                f"Employee: {employee.employee}\n"
                f"Leave Hours: {employee.leave_hours}"
            )

        return None

    def _find_employee(self, question: str, knowledge: KnowledgeBase):
        normalized_question = self._normalize_text(question)
        exact_match = self._find_employee_from_index(normalized_question=normalized_question, knowledge=knowledge)
        if exact_match is not None:
            return exact_match

        best_match = None
        best_score = 0

        for employee in knowledge.employees:
            employee_name = self._normalize_text(employee.employee)
            if employee_name in normalized_question:
                score = len(employee_name.split())
                if score > best_score:
                    best_match = employee
                    best_score = score
                    continue

            tokens = [token for token in employee_name.split() if token]
            if not tokens:
                continue

            first_token = tokens[0]
            if len(first_token) > 2 and first_token in normalized_question and best_score < 1:
                best_match = employee
                best_score = 1

            matched_tokens = 0
            for token in tokens:
                if len(token) > 2 and token in normalized_question:
                    matched_tokens += 1

            if matched_tokens >= 2 and matched_tokens > best_score:
                best_match = employee
                best_score = matched_tokens

        return best_match

    def _find_employee_from_index(self, normalized_question: str, knowledge: KnowledgeBase):
        metadata = knowledge.metadata or {}
        employee_index = metadata.get("employee_index", {})
        employee_token_index = metadata.get("employee_token_index", {})

        for employee_name, employees in employee_index.items():
            if employee_name in normalized_question and employees:
                return employees[0]

        matched_employees = []
        for token in normalized_question.split():
            if len(token) > 2 and token in employee_token_index:
                matched_employees.extend(employee_token_index[token])

        if not matched_employees:
            return None

        scores = {}
        for employee in matched_employees:
            employee_key = id(employee)
            scores[employee_key] = scores.get(employee_key, 0) + 1

        best_employee = None
        best_score = 0
        for employee in matched_employees:
            score = scores[id(employee)]
            if score > best_score:
                best_employee = employee
                best_score = score

        return best_employee

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower()
        for char in ["'", "?", ".", ",", "!", ":", ";", "(", ")", "-", "_"]:
            normalized = normalized.replace(char, " ")
        return " ".join(normalized.split())
