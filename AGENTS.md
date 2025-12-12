# Agent Guidelines for BESSLab

These instructions apply to all files in this repository unless a more specific `AGENTS.md` is added in a subdirectory.

## Coding style and quality
- Favor clarity, maintainability, and correctness over brevity or cleverness.
- Use type hints for new or modified Python functions and keep function boundaries small and focused.
- Prefer vectorized pandas/numpy operations for data work; avoid unnecessary Python loops over rows.
- Add concise docstrings or comments that explain the "why" behind non-obvious logic or decisions.
- Preserve existing file structure, naming, and formatting patterns when making changes.
- Avoid adding new dependencies unless necessary; reuse standard library and existing packages.

## Testing and validation
- When adding functionality, include or update simple tests or usage examples where feasible.
- Note potential edge cases, failure modes, or performance concerns in comments or TODOs when they are not addressed.

## BESS integration and feasibility studies
- Keep assumptions about BESS sizing, operations, and feasibility explicit; never infer business rules without marking them as placeholders.
- Prefer parameterized functions and configurations so BESS scenarios are easy to vary (e.g., capacities, efficiencies, tariffs).
- Clearly separate data ingestion, transformation, and analytics stages to support reproducibility.
- Document unit conventions (kW/kWh, USD, time zones) whenever they appear in code or comments.

## Pull requests and documentation
- Summarize substantive changes clearly in PR descriptions and ensure commit messages are descriptive.
- If adding new user-facing behaviors, include short notes or examples demonstrating usage.
