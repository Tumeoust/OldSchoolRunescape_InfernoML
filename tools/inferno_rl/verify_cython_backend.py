from __future__ import annotations

import argparse
import importlib
import sys


def _is_compiled(module) -> bool:
    module_file = getattr(module, "__file__", "") or ""
    return module_file.lower().endswith((".pyd", ".so"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether the real tools.inferno_rl import path is loading "
            "compiled simulator modules."
        )
    )
    parser.add_argument(
        "--require-compiled",
        action="store_true",
        help="Exit with a non-zero code when geometry/pathfinding/forecast_fast are not compiled.",
    )
    args = parser.parse_args()

    geometry = importlib.import_module("tools.inferno_rl.simulator.geometry")
    pathfinding = importlib.import_module("tools.inferno_rl.simulator.pathfinding")
    try:
        forecast_fast = importlib.import_module("tools.inferno_rl.simulator.forecast_fast")
    except ModuleNotFoundError:
        forecast_fast = None

    modules = {
        "geometry": geometry,
        "pathfinding": pathfinding,
        "forecast_fast": forecast_fast,
    }

    compiled = True
    for name, module in modules.items():
        module_file = getattr(module, "__file__", "<missing>") if module is not None else "<missing>"
        is_compiled = module is not None and _is_compiled(module)
        compiled = compiled and is_compiled
        if module is None:
            backend = "unavailable"
        else:
            backend = "compiled" if is_compiled else "pure-python"
        print(f"{name}: {backend} -> {module_file}")

    if args.require_compiled and not compiled:
        print(
            "Expected compiled simulator modules on the tools.inferno_rl import path.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
