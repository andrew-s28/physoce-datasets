---
title: Contributing
---

## Guide to Contributing

Do you have a publicly available dataset that you've got a great download script for? Do you have specialist knowledge required for calculating derived data products? Please consider [opening an issue](https://github.com/andrew-s28/physoce-datasets/issues) or [submitting a pull request](https://github.com/andrew-s28/physoce-datasets/pulls)! We welcome any and all contributions to this project.

If you'd like to contribute to the code or documentation, please refer to the developer instructions below:

1. [Create a fork of the repository](https://github.com/andrew-s28/physoce-datasets/fork).
2. Clone your fork to your local machine:

    ```bash
    git clone https://github.com/your-username-here/physoce-datasets.git
    cd physoce-datasets
    ```

3. Setup the development environment by installing development and documentation dependencies and installing pre-commit hooks:

    ```bash
    uv sync --dev
    pre-commit install
    ```

4. Create a new branch with a helpful name:

    ```bash
    git checkout -b your-great-new-feature
    ```

5. Make your code changes, stage them via `git add`, and commit them with `git commit -m 'here's my great code changes'`.
6. Push your changes to your fork using:

    ```bash
    git push -u origin your-great-new-feature
    ```

7. [Open a pull request in the upstream repository](https://github.com/andrew-s28/physoce-datasets/compare).

Thanks so much for contributing to open source code!

## AI Contribution Policy

We share the same [AI Usage Policy as xarray](https://docs.xarray.dev/en/stable/contribute/ai-policy.html). In short, this allows developers to use AI tools as a part of their development workflow, but requires that contributors understand all submitted code and take full responosiblity for their changes.
