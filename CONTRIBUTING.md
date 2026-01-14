# Contributing to BESSLab

Thanks for your interest in contributing to **BESSLab**. Contributions of all kinds are welcome—bug reports, documentation improvements, feature proposals, code changes, and test coverage.

By participating in this project, you agree to follow this document and the project’s Code of Conduct (if provided).

---

## 1) License and contribution terms (GNU/GPL)

BESSLab is released under a **GNU/GPL** license (see `LICENSE`).

**Important:** Any contribution you submit (code, documentation, tests, or other material) will be incorporated into the project and distributed under the same GNU/GPL license terms.

By submitting a contribution, you confirm that:
- You have the right to submit it (it is your original work or you have permission).
- You agree that it may be redistributed and modified under the project’s GNU/GPL license.
- You are not knowingly contributing code that is incompatible with GNU/GPL obligations.

If your organization requires a Contributor License Agreement (CLA) or additional paperwork, open an issue before submitting a PR.

---

## 2) Ways to contribute

### A) Report bugs
Please open an issue and include:
- What you expected to happen vs. what happened
- Steps to reproduce (minimal example if possible)
- Screenshots/logs (if applicable)
- Environment details (OS, Python version, app version/commit)
- Any relevant input data structure (redacted if sensitive)

### B) Request features / improvements
Open an issue with:
- The user problem you are solving (not just the solution)
- Proposed approach and alternatives considered
- Impact on existing workflows (compatibility, performance, UX)
- Any standard/regulatory basis if relevant (e.g., IEC/IEEE assumptions)

### C) Contribute code
Pick an issue (or open one) and submit a Pull Request.

---

## 3) Development setup (typical Python app)

> If your repo has a `README.md` setup section, follow that first. This section is the default flow.

1. **Fork** the repository and clone your fork.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
