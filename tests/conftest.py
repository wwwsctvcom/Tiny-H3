"""Make the project importable in tests without an editable install.

The sources live flat under ``src/`` while the import name stays ``tiny_h3``
(same mapping pyproject wires up via ``package-dir``).
"""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

if importlib.util.find_spec("tiny_h3") is None:
    spec = importlib.util.spec_from_file_location(
        "tiny_h3", os.path.join(SRC, "__init__.py"), submodule_search_locations=[SRC]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["tiny_h3"] = module
    spec.loader.exec_module(module)
