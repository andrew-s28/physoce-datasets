---
title: Contributing
---

## Guide to Contributing

Do you have a publicly available dataset that you've got a great download script for? Do you have specialist knowledge required for calculating derived data products? Please consider [opening an issue](https://github.com/andrew-s28/physoce-datasets/issues) or [submitting a pull request](https://github.com/andrew-s28/physoce-datasets/pulls)! We welcome any and all contributions to this project.

If you'd like to contribute to the code or documentation, please refer to the developer instructions below:

1. Fork the repo on GitHub using the "Fork" button in the top right of the [repository home page](https://github.com/andrew-s28/physoce-datasets).
2. Clone your fork to your local machine:

    ```bash
    git clone https://github.com/your-username-here/physoce-datasets.git
    cd physoce-datasets
    ```

3. Setup the development environment by installing development and documentation dependencies and installing pre-commit hooks:

    ```bash
    uv sync --group dev --group docs
    pre-commit install
    ```

    If you're only updating code, you don't need the docs group. If you're only updating docs, you *do* need the dev group.

4. Create a new branch with a helpful name:

    ```bash
    git checkout -b your-great-new-feature
    ```

5. Make your code changes, stage them via `git add`, and commit them with `git commit -m 'here's my great code changes'`.
6. Push your changes to your fork using:

    ```bash
    git push -u origin your-great-new-feature
    ```

7. Open a pull request in the [upstream repository](https://github.com/andrew-s28/physoce-datasets/pulls).

Thanks so much for contributing to open source code!

## AI Contribution Policy

The core developers have made use of modern large language model (LLM) auto-complete and other minor LLM assistance in the development of this project. LLMs can be great for documentation and boiler plate as well as configuration such as GitHub actions, but they are not substitues for a deep understanding of functional code that you are submitting in a pull request. For this reason, **all pull requests that utilize a significant amount of LLM assistance, defined in this case as going above and beyond basic auto-complete and documentation, must include a statement of what code was written exclusively or predominantly by AI**. PRs that do not adhere to this policy may be closed without review, under the sole judgement of project maintainers.
