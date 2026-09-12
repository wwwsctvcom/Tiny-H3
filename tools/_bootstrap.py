"""Make ``import tiny_h3`` work straight from a source checkout.

The package sources live flat under ``src/`` while keeping the import name
``tiny_h3`` (wired up in pyproject via ``package-dir``).  An editable install
(``pip install -e .``) provides the mapping; this shim replicates it so the
CLI tools also run without any installation.
"""

import importlib.util
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def bootstrap() -> None:
    if importlib.util.find_spec("tiny_h3") is not None:
        return
    spec = importlib.util.spec_from_file_location(
        "tiny_h3", os.path.join(_SRC, "__init__.py"), submodule_search_locations=[_SRC]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["tiny_h3"] = module
    spec.loader.exec_module(module)


bootstrap()
