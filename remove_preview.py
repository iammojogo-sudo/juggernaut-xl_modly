#!/usr/bin/env python3
"""remove_preview - restore Modly to the state before the Preview node.

Run by double-clicking remove_preview.bat. This restores the snapshot taken
when the patch was applied, including its hash record, so any other mods you
had before installing Preview are preserved. Close Modly first.
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

    print("Juggernaut XL - remove Preview node (restore pre-install state)")
    print("=" * 40)
    if _modly_running():
        print("Modly is running. Close it completely, then run this again.")
        return 2

    app = P.find_app_asar()
    if not app:
        print("ERROR: could not find Modly's app.asar.")
        return 1
    print(f"Modly app.asar: {app}")

    if not P._backup_path(app).is_file() or not P._record_path(app).is_file():
        print("Nothing to remove - no Preview snapshot found.")
        return 0

    print("Windows will ask permission now. Click Yes.")
    out = str(HERE / "_preview_remove.log")
    P.run_elevated_wait(
        sys.executable,
        [str(HERE / "image_preview_patch.py"), "restore", "--app", app],
        out,
    )
    print("Done. Reopen Modly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
