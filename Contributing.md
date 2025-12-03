# Contributing

Thank you for considering contributing to this repository. This document explains the preferred workflow, coding standards, and testing expectations.

Getting started
1. Fork the repository and create a descriptive branch:
   git checkout -b feature/<short-description>

2. Implement your change with clear commits. Use Conventional Commits style if possible.

3. Run tests and linters locally:
   pip install -r requirements-dev.txt
   pre-commit install
   pre-commit run --all-files
   pytest -q

4. Open a pull request describing:
   - The problem you are solving
   - The design decisions and trade-offs
   - How to reproduce and test changes

Coding standards
- Follow Black formatting and isort imports.
- Keep functions short and documented with docstrings.
- Add unit tests for behavior changes or data pipeline updates.

Testing
- Unit tests for model shapes and data pipeline are required for changes that modify training/inference behavior.
- For experimental scripts, provide smoke tests that run in <30s on CPU.

Licensing & authorship
- Contributions are accepted under the repository LICENSE.
- Include relevant citations for borrowed code and models.

Security
- Do not commit secrets, API keys or private data.
- If you discover security issues, open a private issue and mark it sensitive.
