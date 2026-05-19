# FinIntel AI — LLM Prompt Reference

All prompts are defined as module-level constants and called via `_llm_generate()`.
Each entry lists: **file**, **purpose**, **inputs**, **expected output**, and the **exact prompt**.

---

## 1. Q&A Engine — Main Answer Prompt

| Field | Value |
|---|---|
| **Constant** | `_QA_PROMPT` |
| **File** | `ingestion/qa_engine.py` |
| **Purpose** | Primary prompt that answers every business question from pre-computed context. Returns structured JSON consumed by the frontend. |
| **Inputs** | `{context}` — 20-line analytics slice summary; `{question}` — normalised user question |
| **Output format** | JSON: `{"summary", "visual_type", "columns", "data"}` |
| **Timeout** | 120 s |

### Prompt

```
You are FinIntel AI. Answer from this data:

{context}
---
Question: {question}

Return ONLY valid JSON. No text before or after the JSON.

EXAMPLES:

Q: "What is total revenue?"
{"summary": "Total revenue is $88,960.", "visual_type": "metric", "columns": [], "data": [{"label": "Total Revenue", "value": 88960}]}

Q: "Give me overall summary"
{"summary": "Total Revenue: $88,960, Cost: $77,216, Profit: $11,744, Margin: 13.2%", "visual_type": "metric", ...}

Q: "Top 3 employees by profit"
{"summary": "Top 3 employees by profit.", "visual_type": "table", "columns": ["employee", "profit"], "data": [...]}

Q: "Compare Project Alpha vs Project Beta"
{"summary": "Comparison of ALPHA vs BETA.", "visual_type": "table", "columns": ["metric", "ALPHA", "BETA"], "data": [...]}

RULES:
1.  "metric"  → totals, counts, averages, summaries (label/value pairs)
2.  "table"   → lists, rankings, multi-column breakdowns
3.  "text"    → only if no data found
4.  Lowercase keys: employee, project, month, revenue, cost, profit, utilization, hours, attendance
5.  Numbers must be raw numerics — no $ or % strings
6.  summary must be a specific answer, not a generic heading
7.  Return ONLY the JSON object, nothing else
8.  "last month" = month marked LAST MONTH in context
9.  highest/top/best → sort DESCENDING; lowest/bottom/worst → sort ASCENDING
10. "not generating", "zero" → filter for 0 or negative values
11. "below X%" → only values < X;  "above X%" → only values > X
12. "per employee/project/hour" → compute average
13. "compare A vs B" → side-by-side table, EXACT project name as column header
14. "YTD" → sum Jan to last available month
15. Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
16. "top N in Project X" → filter by project, then rank
17. Project utilization → use AvgUtilisation from PROJECTS section only
18. "which project/employee has highest/lowest X" → return ONLY the single winner as metric
19. Compare numerics exactly: 96.6 > 96.0 > 95.0
ANTI-HALLUCINATION (CRITICAL):
20. Use exact numbers from GROUNDING FACTS section; never re-derive them
21. Entity absent from data → {"summary": "No data found.", "visual_type": "text", "columns": [], "data": []}
22. Project names are CASE-SENSITIVE and must match PROJECTS section exactly
23. Never mix data across projects
24. Monthly breakdowns → only use months in MonthlyBreakdown for that project
25. For designation questions → filter by Designation field; skip employees with empty Designation
26. Q quarter without year → use most recent year in data
27. "this year" / "YTD" → only months from current calendar year
28. NEVER echo or rephrase the question as the answer
```

---

## 2. Question Normaliser

| Field | Value |
|---|---|
| **Constant** | `_norm_prompt` (inline) |
| **File** | `ingestion/qa_engine.py` — `_normalize_question()` |
| **Purpose** | Corrects spelling and grammar in the user's question before it enters the main pipeline. |
| **Input** | `{q}` — raw user question |
| **Output** | Plain text — corrected question only |
| **Timeout** | 30 s |
| **Safety guard** | If response is empty or > 2× the original length it is discarded and the raw question is used |

### Prompt

```
Correct any spelling or grammar mistakes in the following question
and return ONLY the corrected question text, nothing else.

Question: {q}
```

---

## 3. Forecast — Intent Classifier

| Field | Value |
|---|---|
| **Constant** | Inline f-string in `_llm_classify_question()` |
| **File** | `ingestion/forecast_.py` |
| **Purpose** | Determines whether the user's question is asking for a future forecast or querying historical data. Also extracts forecast parameters (scope, targets, horizon). |
| **Input** | `{question}`, known project/employee names |
| **Output** | JSON with `is_forecast`, `metrics`, `scope`, `targets`, `horizon_type`, `horizon_value`, `explicit_periods` |
| **Timeout** | 30 s |

### Prompt

```
You are a financial data assistant. Determine if this question is asking for a
FUTURE FORECAST or just querying EXISTING/HISTORICAL data.

Question: "{question}"

CRITICAL: is_forecast=true ONLY if the question explicitly asks about FUTURE
predictions using words like:
- "forecast", "predict", "projection", "estimate", "expected"
- "next month", "next quarter", "next 3 months", "upcoming"
- Future dates beyond current data (e.g., "July 2026", "Q3 2026")

is_forecast=false for:
- Current or past data: "total revenue", "show employees", "top performers"
- Comparisons: "compare", "difference between"
- Lists/rankings: "top 5", "highest", "lowest", "list all"
- Aggregations: "total", "average", "sum", "count"

EXAMPLES:
- "What is the total revenue?"          → is_forecast=false
- "Top 5 employees by utilization"      → is_forecast=false
- "Forecast revenue for next quarter"   → is_forecast=true
- "Predict profit for July 2026"        → is_forecast=true

Return ONLY valid JSON:
{
  "is_forecast": true | false,
  "metrics": ["revenue", "profit", "leaves", ...],
  "scope": "overall" | "project" | "employee",
  "targets": [...],
  "horizon_type": "months" | "quarters" | "explicit_months" | "explicit_quarter",
  "horizon_value": <int | null>,
  "explicit_periods": ["July 2026", ...]
}
```

---

## 4. Forecast — Method Adviser

| Field | Value |
|---|---|
| **Function** | `_llm_advise_settings()` |
| **File** | `ingestion/forecast_.py` |
| **Purpose** | Recommends the best forecasting algorithm (`lin` / `holt` / `sma`) and damping/adjustment parameters based on the most recent 6 months of data. |
| **Input** | Last-6-month snapshot of revenue, cost, profit, headcount, utilization, hours, leave days; raw user question |
| **Output** | JSON: `{method, params, damping, adjustments}` |
| **Timeout** | 60 s |

### Prompt (structure)

```
You are a forecasting assistant. Based on the last months, advise the method
and adjustments.
Return ONLY JSON with keys: method, params, damping, adjustments.
- method:      { metric: 'lin' | 'holt' | 'sma' }
- params:      { metric: {alpha?, beta?, k?} }
- damping:     { metric: 0–1 }  // 0=no damping, 0.3=30% month-over-month damping
- adjustments: [ {label, metric, type: 'percent'|'absolute', value} ]

Recent monthly snapshot (last up to 6):
- {month}: revenue=…, cost=…, profit=…, utilization=…, …

User question: {question}
Now respond with strict JSON only.
```

---

## 5. AI Insights — Risk & Scorecard Narrative

| Field | Value |
|---|---|
| **Constant** | `_INSIGHT_PROMPT` |
| **File** | `ingestion/risk_engine.py` |
| **Purpose** | Generates exactly 3 actionable management insights from the dataset summary, top 10 risks, and employee scorecards. |
| **Inputs** | `{summary}` — overall KPIs + per-project stats; `{risk_text}` — top 10 risks; `{scorecard_text}` — top 8 employee scorecards |
| **Output** | Plain text — 3 insight paragraphs separated by blank lines |
| **Timeout** | 180 s |

### Prompt

```
You are FinIntel AI, a senior HR and financial analytics advisor.

Dataset summary:
{summary}

Top 10 risks detected:
{risk_text}

Employee scorecards:
{scorecard_text}

Output exactly 3 insights. Rules:
- Start the FIRST word of insight 1 directly — no intro, no preamble, no title
- No numbering, no bullet points, no labels
- Each insight is 1-2 sentences
- Include a specific number from the data
- Tell management what to DO
- Separate insights with a blank line
- Do NOT write anything before the first insight
```

---

## 6. Text Parser — Timesheet Extraction

| Field | Value |
|---|---|
| **Constant** | `_EXTRACT_PROMPT` |
| **File** | `ingestion/text_parser.py` |
| **Purpose** | Extracts structured employee timesheet records from raw unstructured text (e.g., pasted PDF or email content). |
| **Input** | `{text}` — raw text block |
| **Output** | JSON array of employee record objects |
| **Timeout** | 120 s |
| **Guard** | Only called if `looks_like_timesheet_text()` heuristic passes |

### Prompt

```
You are a data extraction assistant. Extract ALL employee timesheet/financial
records from the text below.

TEXT:
{text}

For EACH employee entry found, extract:
- employee:      person's name
- project:       project or client name
- month:         month and year (e.g. "MARCH'26" or "March 2026")
- actual_hours:  hours worked (number)
- billing_rate:  billing rate per hour (number)
- cost_rate:     cost rate per hour (number)
- vacation_days: leave/vacation days (0 if not mentioned)

Return ONLY a JSON array of objects. Example:
[
  {"employee": "John Smith", "project": "Alpha", "month": "MARCH'26",
   "actual_hours": 160, "billing_rate": 150, "cost_rate": 85, "vacation_days": 0}
]

If no records can be extracted, return an empty array: []
Extract ALL records you can find. Do NOT make up data that isn't in the text.
```

---

## 7. AI Mapper — Column Header Mapping

| Field | Value |
|---|---|
| **Constant** | `_COL_PROMPT` |
| **File** | `ingestion/ai_mapper.py` |
| **Purpose** | Maps arbitrary Excel column headers to standard semantic field names. Used as a fallback when deterministic rule matching fails. |
| **Input** | `{header_list}` — numbered list of column header strings |
| **Output** | JSON: `{field_name: column_index, ...}` |
| **Timeout** | Default (60 s) |

### Prompt

```
You are a data analyst. Given column headers from a timesheet Excel file,
map each to a semantic field.

Column headers (index → value):
{header_list}

Fields:
- name:           employee/person name
- project:        project, client, account, engagement
- billing_rate:   billing rate per hour (NOT a dollar total/amount)
- cost_rate:      cost/CTC rate per hour
- actual_hours:   actual hours worked
- billable_hours: final billable hours (NOT the same as max/approved hours)
- max_hours:      maximum/approved/expected/budgeted hours
- leaves:         leave days, vacation, PTO
- working_days:   working days

Return ONLY a JSON: {"name": <idx>, "project": <idx>, ...}
Use null if a field has no matching column.
```

---

## 8. AI Mapper — Full Sheet Structure Analysis

| Field | Value |
|---|---|
| **Constant** | `_SHEET_PROMPT` |
| **File** | `ingestion/ai_mapper.py` |
| **Purpose** | Detects the full structural layout of an Excel sheet: which row is the header, where data starts/ends, and the project name if it lives in sheet metadata rather than a column. |
| **Input** | `{sheet_name}`, `{rows_text}` — first 35 rows as text |
| **Output** | JSON: `{header_row, columns: {...}, project_name, data_start, data_end}` |
| **Timeout** | 90 s |

### Prompt

```
You are a data analyst. Analyze the following rows from a timesheet Excel sheet
and determine its structure.

Sheet name: "{sheet_name}"

Rows (row_index: [cell values]):
{rows_text}

Determine:
1. header_row:    Row index containing the main data column headers
2. columns:       Semantic-field → 0-based column index mapping
                  (name, project, billing_rate, cost_rate, actual_hours,
                   billable_hours, max_hours, leaves, working_days)
3. project_name:  If "project" column is null, project name from sheet metadata
4. data_start:    Row index where employee data rows begin
5. data_end:      Row index of the last employee data row

Return ONLY a JSON object:
{"header_row": <int>, "columns": {"name": <int>, "project": <int|null>, ...},
 "project_name": <string|null>, "data_start": <int>, "data_end": <int>}
```

---

## Prompt Interaction Diagram

```
User Question
     │
     ▼
[2] Normaliser       ← fixes typos before anything else
     │
     ▼
[3] Forecast Classifier
     ├── is_forecast=true ──▶ [4] Method Adviser ──▶ Deterministic engine
     │
     └── is_forecast=false
              │
              ▼
         Rule-based engine (no LLM)
              │  returns None?
              ▼
         [1] Main Q&A Prompt  ◀── pre-computed 20-line context slice
              │
              ▼
         _validate_answer()  ◀── numeric hallucination check
              │  fails?
              ▼
         Rule-based fallback (no LLM)

Excel / Text Upload
     │
     ├──▶ [8] Sheet Structure  ──▶ [7] Column Mapping  ──▶ ingest pipeline
     │
     └──▶ [6] Text Extraction

Dashboard / Risks page
     └──▶ [5] AI Insights
```

---

## Key Response Shape (all Q&A responses)

```json
{
  "summary":     "Human-readable answer sentence",
  "visual_type": "metric | table | text",
  "columns":     ["col1", "col2"],
  "data":        [{"col1": ..., "col2": ...}],
  "sources": {
    "total_records": 240,
    "time_range": "ALL",
    "forecast": false
  }
}
```
