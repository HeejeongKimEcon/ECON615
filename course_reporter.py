"""
Course reporter: lets the instructor and TA see, during class, which exercises
you have run and any error messages, so we can help you faster.

What is sent after each cell you run: your name, the lecture, the cell number,
whether it ran without problems, the error or warning message and, only when
there is a problem, the code of that cell. Nothing else is sent.

Turned on by the first cell of each lecture notebook:

    import course_reporter; course_reporter.start("lecture01")
"""

import datetime
import difflib
import json
import os
import threading
import urllib.request
import warnings
from pathlib import Path

# The instructor's Power Automate webhook address (set once for the course)
ENDPOINT = "https://default5569f185d22f4e139850ce5b1abcd2.e8.environment.api.powerplatform.com:443/powerautomate/automations/direct/cu/17/workflows/c0fcf838b979441bbcad5964e4d90a94/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=OdvFEYQZ-AObmqaYyoFUNsXRbx-X7z-yeMD5H8GASB8"

NAME_FILE = Path.home() / ".course_name"
MAX_CODE = 3000  # characters of a problem cell's code that are sent


def _not_finite(x):
    try:
        import numpy as np
        return x is not None and not np.all(np.isfinite(np.asarray(x, dtype=float)))
    except Exception:
        return False


# Lecture-specific checks for mistakes that do not raise an error:
# (exercise cell number, test on the student's variables, message)
CHECKS = {
    "lecture01": [
        (5, lambda ns: ns.get("it") == 999, "Hit the iteration cap: value function did not converge"),
        (5, lambda ns: _not_finite(ns.get("v")), "NaN or Inf in the value function"),
    ],
}


def exercise_cells(notebook_path):
    """(id, code) of the exercise cells, in order: code cells other than this setup cell."""
    nb = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    cells = [(c.get("id"), "".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    return [(i, src) for i, src in cells if "course_reporter" not in src]


def _send(report):
    if not ENDPOINT:
        return
    body = json.dumps({"type": "message", "attachments": [], "report": report}).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10).close()
    except Exception:
        pass  # never disturb the student's work


class _Reporter:
    def __init__(self, shell, lecture, name):
        self.shell, self.lecture, self.name = shell, lecture, name
        self.user = os.environ.get("JUPYTERHUB_USER", "local")
        self.seq = 0
        self.code, self.cell_id, self.caught = "", None, []
        self._showwarning = warnings.showwarning

    def report(self, cell, status, message="", code=""):
        self.seq += 1
        report = {"v": 1, "user": self.user, "name": self.name, "lecture": self.lecture,
                  "cell": cell, "status": status, "message": message[:500], "code": code[:MAX_CODE],
                  "seq": self.seq,
                  "time": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        threading.Thread(target=_send, args=(report,), daemon=True).start()

    def cell_number(self):
        try:
            cells = exercise_cells(f"{self.lecture}.ipynb")
        except (OSError, ValueError, KeyError):
            return None  # the notebook was renamed or moved
        ids = [i for i, _ in cells]
        if self.cell_id in ids:
            return ids.index(self.cell_id) + 1
        # Kernels that do not say which cell is running: take the saved cell whose
        # code is most similar (the saved copy can lag behind the student's typing)
        scores = [difflib.SequenceMatcher(None, self.code, src).ratio() for _, src in cells]
        best = max(range(len(scores)), key=scores.__getitem__, default=None)
        return best + 1 if best is not None and scores[best] > 0.6 else None

    def pre(self, info):
        self.code, self.cell_id, self.caught = info.raw_cell, getattr(info, "cell_id", None), []
        self._showwarning = original = warnings.showwarning

        def record(message, category, *args, **kwargs):
            self.caught.append(f"{category.__name__}: {message}")
            return original(message, category, *args, **kwargs)  # student still sees it
        warnings.showwarning = record

    def post(self, result):
        warnings.showwarning = self._showwarning
        if "course_reporter" in self.code or not any(
                line.strip() and not line.strip().startswith("#") for line in self.code.splitlines()):
            return  # the setup cell, or an exercise still holding only its "# TODO" placeholder
        cell = self.cell_number()
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            self.report(cell, "err", f"{type(error).__name__}: {error}", self.code)
            return
        flags = []
        for n, test, message in CHECKS.get(self.lecture, []):
            try:
                if n == cell and test(self.shell.user_ns):
                    flags.append(message)
            except Exception:
                pass
        flags += list(dict.fromkeys(self.caught))  # warnings, without repeats
        if flags:
            self.report(cell, "warn", " · ".join(flags[:2]), self.code)
        else:
            self.report(cell, "ok")


_active = None


def _ask_name():
    name = os.environ.get("COURSE_NAME")
    if not name and NAME_FILE.exists():
        name = NAME_FILE.read_text(encoding="utf-8").strip()
    if not name:
        name = input("Your name as on the class list (asked only once): ").strip()
        NAME_FILE.write_text(name, encoding="utf-8")
    return name


def start(lecture):
    """Turn on reporting for this notebook (run it again any time; it does not duplicate)."""
    global _active
    from IPython import get_ipython
    shell = get_ipython()
    if shell is None:
        print("The course reporter only works inside Jupyter.")
        return
    if _active is not None:
        shell.events.unregister("pre_run_cell", _active.pre)
        shell.events.unregister("post_run_cell", _active.post)
    _active = _Reporter(shell, lecture, _ask_name())
    shell.events.register("pre_run_cell", _active.pre)
    shell.events.register("post_run_cell", _active.post)
    _active.report(None, "start")
    print(f"Course reporter is on for {lecture} ({_active.name}). "
          "Your instructor sees which cells you ran and any error messages.")
