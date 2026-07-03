---
title: Installation
---

## uv

`uv` is the preferred way to manage this package. Please refer to the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Once uv is installed, you can initialize a virtual environment with `uv sync` or you can run any commands directly with `uv run` and `uv` will handle the `venv` creation and activation auto-magically.

```bash
# add physoce-datasets to your project folder, equivalent to a pip install
uv add physoce-datasets
# you can now run physoce-datasets from your project folder
uv run physoce-datasets --help
```

Once you've added `physoce-datasets` to your project, you can import any of the download classes in your scripts:

```python
# note the import uses underscore in place of dash
from physoce_datasets import copernicus_marine, era5, ooi_ea
```

Alternatively, you can run the command line interface from anywhere using [uv tools](https://docs.astral.sh/uv/guides/tools/):

```bash
# run physoce-datasets from the command line anywhere!
uvx physoce-datasets --help
```

## pip

Of course, you can also use the classic `pip`, but you have to handle creating and activating the `venv` yourself:

```bash
python -m venv .venv
source .venv/bin/activate
pip install physoce-datasets
```
