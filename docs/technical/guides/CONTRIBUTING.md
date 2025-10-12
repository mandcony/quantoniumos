# Contributing to QuantoniumOS

We welcome contributions! This guide outlines the process for contributing to the project to ensure a smooth and effective workflow.

## Branching Strategy

We use a simple feature-branching workflow:

1.  **`main` branch is protected.** Direct pushes to `main` are not allowed.
2.  For any new feature or bugfix, create a new branch from the latest `main`.
3.  Use a descriptive branch name, e.g., `feature/new-crypto-algorithm` or `fix/ui-rendering-bug`.

## Making Changes

1.  **Fork the repository** to your own GitHub account.
2.  **Clone your fork** to your local machine.
3.  **Create a new branch** for your changes.
4.  **Make your code changes.** Please adhere to the existing coding style and patterns.
5.  **Run the validation suite** to ensure your changes have not broken existing functionality:
    ```bash
    python tests/comprehensive_validation_suite.py
    ```
6.  **Commit your changes** with a clear and descriptive commit message.

## Submitting a Pull Request (PR)

1.  **Push your branch** to your fork on GitHub.
2.  **Open a Pull Request** from your branch to the `main` branch of the original repository.
3.  **Fill out the PR template:**
    *   Provide a clear title for your PR.
    *   In the description, explain the "what" and "why" of your changes.
    *   If your PR resolves an existing issue, link it using `Closes #issue-number`.
4.  **The team will review your PR.** Be prepared to answer questions or make further changes based on feedback.

## Coding Standards

-   **Python:** Follow PEP 8 guidelines. Use a linter like `flake8` or `black` to maintain consistent formatting.
-   **C:** Follow a consistent style, similar to the existing code in `src/assembly/kernel/`.
-   **Documentation:** If you add a new feature, update the relevant documentation in the `/docs` directory. If you change behavior, ensure the documentation is updated to reflect it.
