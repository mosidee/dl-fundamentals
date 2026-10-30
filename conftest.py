"""Lets tests in each module folder import `common` and pick exercise vs solution via IMPL."""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_impl(module_dir: str):
    """Import `exercise` (default) or `solution` (IMPL=solution) from the given module folder."""
    name = os.environ.get("IMPL", "exercise")
    folder = ROOT / module_dir
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    # Each module folder registers its own copy; drop a stale one from another folder.
    for key in [k for k in sys.modules if k in ("exercise", "solution")]:
        mod = sys.modules[key]
        if getattr(mod, "__file__", "") and not str(mod.__file__).startswith(str(folder)):
            del sys.modules[key]
    return importlib.import_module(name)
