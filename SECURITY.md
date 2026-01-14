# Security Policy

## Supported Versions

Security updates are provided for the latest version on the default branch.

If you are using an older release, upgrade to the latest commit/version before reporting an issue unless the vulnerability prevents upgrading.

---

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues, discussions, or social media.

Instead, use one of the following private reporting options:

### Preferred: GitHub Private Vulnerability Reporting
1. Go to the repository’s **Security** tab.
2. Click **Report a vulnerability**.
3. Provide the information requested below.

This method keeps the report private and enables coordinated disclosure.

### Alternative: Maintainer Contact
If GitHub Private Vulnerability Reporting is not available for this repository, contact the maintainers privately:
- Open a **draft pull request** that contains **no exploit code** and request a private discussion, or
- Use another private channel listed in the repository (if any).

---

## What to Include in Your Report

To help us triage quickly, include:
- A clear description of the vulnerability and potential impact
- Affected component(s) and versions/commit hash
- Steps to reproduce (prefer a minimal proof of concept)
- Any logs, screenshots, or stack traces that help explain the issue
- Whether you have a suggested fix or mitigation
- Your preferred credit name (if you want acknowledgement)

---

## Coordinated Disclosure

We aim to:
- Acknowledge receipt of your report within **7 days**
- Provide an initial assessment and next steps within **14 days**
- Coordinate a fix and public disclosure timeline based on severity and complexity

We may request additional details to reproduce the issue reliably.

---

## Scope

### In scope
- Vulnerabilities in application code, authentication/authorization (if applicable), and data handling
- Dependency vulnerabilities that materially affect BESSLab users
- Misconfigurations or insecure defaults in shipped configuration

### Out of scope
- Vulnerabilities requiring physical access to a user’s device
- Social engineering attacks
- Denial-of-service reports that rely on unrealistic traffic volumes
- Issues in third-party services outside this repository’s control

---

## Security Best Practices for Contributors

- Do not commit credentials, API keys, tokens, or private endpoints.
- Avoid including real operational/commercial data in issues, PRs, or test fixtures.
- Prefer secure-by-default configuration and input validation.
- Keep dependencies minimal and up to date.

---

## Credit

We appreciate responsible disclosure and will provide credit in release notes/advisories upon request, unless you prefer to remain anonymous.
