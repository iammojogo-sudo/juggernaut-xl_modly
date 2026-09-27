#!/usr/bin/env python3
"""add_preview - install the "Preview" node into Modly (manual fallback).

Prefer letting setup.py do this automatically. Use this script when you want
to apply the patch by hand, or when the auto-apply did not run.

Run by double-clicking add_preview.bat (uses the extension venv and asks
Windows for permission). Close Modly first - the app.asar cannot be swapped
while Modly is running.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _modly_running():
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process -Name Modly -ErrorAction SilentlyContinue | "
             "Select-Object -First 1 -ExpandProperty Id"],
            capture_output=True, text=True, timeout=30)
        return bool((r.stdout or "").strip())
    except Exception:  # noqa: BLE001
        return False


def main():
    try:
        import image_preview_patch as P
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot load image_preview_patch.py ({exc})")
        return 1

    print("Juggernaut XL - install Preview node")
    print("=" * 40)
    if _modly_running():
        print("Modly is running. Close it completely, then run this again.")
        return 2

    app = P.find_app_asar()
    if not app:
        print("ERROR: could not find Modly's app.asar.")
        return 1
    print(f"Modly app.asar: {app}")

    try:
        state = P.patch_state(app)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not read Modly's app.asar ({exc})")
        return 1
    if not P.needs_patch(app):
        print("The Preview node and the parameter sliders are ALREADY installed.")
        return 0
    missing = [k for k, v in state.items() if not v]
    print(f"Missing features: {', '.join(missing)}")

    work = HERE / "_preview_stage"
    try:
        staged, checks = P.stage(app, str(work))
    except P.PatchError as exc:
        print("This Modly build does not match the patch (version changed).")
        print("Nothing was changed. A port of the patch is needed for this build.")
        print(f"  - {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: staging failed ({exc}). Nothing was changed.")
        return 1
    for c in checks:
        print(f"  check: {c}")

    print("Windows will ask permission now. Click Yes.")
    out = str(HERE / "_preview_install.log")
    P.run_elevated_wait(
        sys.executable,
        [str(HERE / "image_preview_patch.py"), "finalize",
         "--app", app, "--staged", str(staged)],
        out,
    )
    print("Done. Reopen Modly - the node is called Preview.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
