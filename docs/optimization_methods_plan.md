# Optimization Methods Expansion Plan

This document proposes a production-oriented path to add four optimization capabilities to BESSLab:

1. Structured sensitivity analysis
2. Brute-force design-space search
3. Differential Evolution (DE)
4. Combined top-down + bottom-up optimization

The approach is intentionally incremental so each phase can be validated before introducing more computational complexity.

## Current-state fit in this repository

BESSLab already has building blocks that align with this expansion:

- KPI-producing simulation core in `services/simulation_core.py`
- Existing sweep utilities in `utils/sweeps.py`
- Existing sensitivity and scenario UI pages in `pages/06_Sensitivity_Stress_Test.py` and `pages/05_Multi_Scenario_Batch.py`
- Main app KPI/economics outputs in `app.py` (LCOE/LCOS/NPV/IRR and dispatch diagnostics)

That means we can implement optimization as a thin orchestration layer around an existing deterministic simulation function, rather than rewriting model internals.

---

## 1) Structured sensitivity analysis (simple and explainable)

### Goal
Support one-way and two-way sensitivity runs on selected levers (e.g., CAPEX, tariff, RTE, availability, degradation, WESM prices), then visualize impact on KPIs such as NPV, PIRR, EIRR, shortfall, and compliance.

### Suggested implementation

- Add a dedicated service module:
  - `services/analysis_sensitivity.py`
- Define typed payloads that keep assumptions explicit:
  - `SensitivityParameter(name: str, base_value: float, values: list[float], unit: str)`
  - `SensitivityRequest(parameters: list[SensitivityParameter], kpis: list[str])`
- Reuse the current simulation/economics entrypoint for each sampled point.
- Return normalized long-form results DataFrame with explicit units and scenario IDs.

### UI / API integration

- Extend existing sensitivity page to support:
  - one-way tornado (already conceptually present)
  - two-way heatmap matrix
- Add CSV export containing both sampled inputs and computed KPIs.

### Guardrails

- Reject hidden business assumptions (e.g., inferred tariff escalation) unless user supplied.
- Display baseline values and delta definitions (% vs absolute) directly in chart subtitles.

---

## 2) Brute-force optimization (exhaustive search)

### Goal
Exhaustively evaluate every variable combination in bounded ranges for low-dimensional design spaces.

### Suggested implementation

- Add a search service:
  - `services/analysis_bruteforce.py`
- Input schema:
  - `SearchVariable(name: str, values: list[float], unit: str)`
  - `BruteForceRequest(variables: list[SearchVariable], objective: str, constraints: dict[str, float])`
- Build Cartesian product with `itertools.product`.
- Evaluate each candidate through the same deterministic simulation wrapper.
- Store outputs in a tabular result set (candidate inputs + KPIs + feasibility flags).

### Performance strategy

- Hard warning if total combinations exceed a configurable threshold.
- Optional chunked execution to avoid memory spikes.
- Optional multiprocessing (future) behind a toggle to avoid deployment instability.

### Visualization

- 2 variables: heatmap
- 3 variables: faceted heatmaps or 3D scatter
- >3 variables: parallel-coordinates + filtered ranking table

### Guardrails

- Explicitly show how many combinations were skipped or clipped.
- Avoid hiding high-dimensional uncertainty behind a single 2D plot.

---

## 3) Differential Evolution (global optimizer for complex spaces)

### Goal
Find near-global optima with fewer simulations than brute force for non-linear/non-differentiable objective surfaces.

### Suggested implementation

- Add an optimizer service:
  - `services/analysis_de.py`
- Prefer SciPy DE if already available; otherwise keep a small pure-Python fallback implementation.
- Input schema:
  - `DEVariable(name: str, lower: float, upper: float, unit: str, is_integer: bool = False)`
  - `DERequest(variables: list[DEVariable], objective: str, constraints: dict[str, float], seed: int)`
- Objective function should:
  - run simulation
  - apply penalties for constraint violations
  - return scalar value for minimization

### Reproducibility and trust

- Persist random seed and algorithm hyperparameters in outputs.
- Return convergence history and top-N candidate solutions (not just the winner).
- Include a "replay best candidate" action in UI for auditability.

### Guardrails

- Enforce bounded search domains for every decision variable.
- Mark all penalty formulations as configurable placeholders unless agreed with stakeholders.

---

## 4) Top-down + bottom-up optimization (bankable realism)

### Goal
Couple design decisions (size, commercial assumptions) with an internal dispatch optimizer that decides charge/discharge/bids per timestep.

### Suggested architecture

- Keep top-down optimizer as orchestrator (sensitivity/brute-force/DE).
- Add bottom-up dispatch solver interface:
  - `services/dispatch_optimizer.py`
  - `optimize_dispatch(inputs: DispatchInputs) -> DispatchDecisionSeries`
- Inject dispatch optimizer into simulation run so each candidate sizing/commercial tuple is evaluated with optimized operations.

### Progressive rollout

1. Start with single-market arbitrage heuristic with clear constraints.
2. Add co-optimized arbitrage + ancillary objective weights.
3. Introduce degradation-aware dispatch penalty terms.

### Guardrails

- Every dispatch run should log constraint binding events (SOC floor/ceiling, power limits, contractual requirements).
- Separate revenue decomposition by market to prevent black-box conclusions.

---

## Recommended phased delivery

### Phase 1 (fast, low-risk)

- Unify a single simulation evaluation wrapper used by all analysis modes.
- Expand current sensitivity workflow to robust one-way/two-way tables and exports.
- Deliver stakeholder-friendly visuals and baseline comparison cards.

### Phase 2 (mid complexity)

- Add brute-force engine with capped combinations + chunked execution.
- Add ranking and filtering UI for feasible candidate review.

### Phase 3 (advanced optimization)

- Add DE module with reproducible seeds, constraints, and convergence diagnostics.
- Benchmark DE vs brute-force on curated scenarios.

### Phase 4 (integrated optimization)

- Add internal dispatch optimizer interface and initial arbitrage implementation.
- Wire top-down search to bottom-up optimized dispatch.

---

## Suggested acceptance tests

- Deterministic replay test: same seed and inputs produce same best candidate.
- Constraint enforcement test: infeasible candidates always penalized or filtered.
- Consistency test: brute-force optimum is matched/approached by DE on small benchmark grids.
- Units regression test: objective comparisons do not mix MW/MWh or USD/PHP silently.
- Performance smoke test: run completes under configured limit for a known scenario size.

---

## Practical notes for BESS feasibility workflows

- Keep all design variables user-configurable with documented units (MW, MWh, USD/MWh, %).
- Treat tariff models, bid strategies, and market-priority rules as explicit configuration—not hard-coded assumptions.
- Always export both input decisions and output KPIs for bankability and peer review.
