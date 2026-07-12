# deps.py
from __future__ import annotations
import os
import sys
import subprocess
import importlib
from typing import Dict, Tuple

from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog, QApplication
from qgis.PyQt.QtCore import Qt


def _python_for_pip() -> str:
    """
    Return a Python interpreter path suitable for 'python -m pip ...'.
    On Windows/QGIS, sys.executable may be qgis-bin.exe; try nearby python.exe.
    """
    exe = sys.executable
    base = os.path.basename(exe).lower()
    if os.name == "nt" and not base.startswith("python"):
        # try sibling python.exe
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(cand):
            return cand
        # try common bundled locations under QGIS root
        root = os.path.dirname(os.path.dirname(exe))
        for sub in (
            os.path.join("apps", "Python311", "python.exe"),
            os.path.join("apps", "Python310", "python.exe"),
            os.path.join("apps", "Python39",  "python.exe"),
            os.path.join("bin", "python3.exe"),
        ):
            cand2 = os.path.join(root, sub)
            if os.path.exists(cand2):
                return cand2
    return exe


def _import_ok(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def ensure_modules(
    required: Dict[str, str],
    parent_widget,
    *,
    ask_user: bool = True,
    title: str = "ArchaeoAstroInsight: Missing Python packages",
) -> Tuple[bool, str]:
    """
    Ensure modules in `required` (module -> pip package) are importable.
    If missing, optionally prompt and attempt installation via pip.

    Returns (ok, message). If ok=False, message explains why.
    """
    missing = [m for m in required if not _import_ok(m)]
    if not missing:
        return True, "All dependencies present."

    pkgs = [required[m] for m in missing]
    if ask_user:
        msg = (
            "This feature needs extra Python packages:\n\n"
            + "  • " + "\n  • ".join(pkgs)
            + "\n\nInstall them now into this QGIS Python?"
        )
        resp = QMessageBox.question(
            parent_widget, title, msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if resp != QMessageBox.Yes:
            return False, "User declined installation."

    # Run pip with a small modal progress dialog
    py = _python_for_pip()
    dlg = QProgressDialog("Installing dependencies…", "Cancel", 0, 0, parent_widget)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setAutoClose(False)
    dlg.setMinimumDuration(0)
    dlg.show()
    QApplication.processEvents()

    try:
        cmd = [py, "-m", "pip", "install", "--upgrade", "--disable-pip-version-check"] + pkgs
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # keep UI responsive
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            QApplication.processEvents()
            if dlg.wasCanceled():
                proc.terminate()
                return False, "Installation cancelled."
        ok = (proc.wait() == 0)
        if not ok:
            return False, "pip failed to install required packages."
    except Exception as e:
        return False, f"pip invocation failed: {e}"
    finally:
        dlg.close()

    # Re-check imports
    still = [m for m in required if not _import_ok(m)]
    if still:
        return False, "Installed, but imports still failing (try restarting QGIS)."
    return True, "Installed dependencies."
