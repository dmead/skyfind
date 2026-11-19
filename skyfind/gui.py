"""Watch a solve: the same picture as the replay GIFs — trails lit up as
they're fitted and named as they're identified, the object list growing,
the candidate sites on the world map closing in.

`LiveView` follows a work folder's `events.jsonl`, so it shows a solve
running anywhere (the workbench, another terminal) and replays a finished
one; the workbench (`workbench.py`) embeds it. `window` is the bare view in
a window of its own (`skyfind watch`). Needs the `gui` extra (PySide6).
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow,
                               QSizePolicy)

from .replay import Renderer, Story, scenes


_POOL = ThreadPoolExecutor(1)


def _load(wd, from_events: bool):
    """(story, scenes) for a work folder: its event log, or its results."""
    story = Story.from_events(wd) if from_events else Story.from_workdir(wd)
    return story, (scenes(story) if story is not None else [])


class LiveView(QLabel):
    """The replay picture for one work folder, advancing a frame per tick
    and re-reading the event log about once a second."""

    def __init__(self, wd=None, *, fps: int = 10, width: int = 1280,
                 parent=None):
        super().__init__(parent)
        self.fps, self.width_px = fps, width
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #121416; color: #9aa;")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / fps))
        self.set_workdir(wd)

    def set_workdir(self, wd) -> None:
        self.wd = wd
        self.loading = None                  # a reload in the background
        self.story = self.render = None
        self.frames: list = []
        self.i = 0
        self.stamp = None
        self.polls = 0
        self.frame = None
        self.setPixmap(QPixmap())
        self.setText("no solve yet: the picture appears when one starts"
                     if wd is not None else "open a clip")

    def progress(self) -> tuple[int, int, bool]:
        """(tracks fitted, tracks to fit, solved)."""
        s = self.story
        if s is None:
            return 0, 0, False
        return len(s.tracks), s.expected, s.solution is not None

    def reload(self) -> None:
        """Look for news in the event log; reading it and laying out the
        replay (seconds for a big solve) happen off the window's thread."""
        wd = self.wd
        if wd is None:
            return
        if self.loading is not None:
            if self.loading.done():
                self._take(self.loading)
                self.loading = None
            return
        p = wd.path("events.jsonl")
        stamp = p.stat().st_mtime_ns if p.exists() else None
        if stamp is None and wd.has("solution.json") and self.story is None:
            stamp = "done"
        if stamp is None or stamp == self.stamp:
            return
        self.stamp = stamp
        self.loading = _POOL.submit(_load, wd, p.exists())

    def _take(self, fut) -> None:
        try:
            story, frames = fut.result()
        except (OSError, ValueError, KeyError):
            return                    # half-written log: next time
        if story is None or fut is not self.loading:
            return
        if self.render is None or story.solution is not None and \
                (self.story is None or self.story.solution is None):
            self.render = Renderer(story, width=self.width_px)
        else:
            self.render.story = story
        self.story = story
        self.frames = frames

    def restart(self) -> None:
        """Play from the beginning again."""
        self.i = 0

    def tick(self) -> None:
        self.polls += 1
        if self.polls % max(1, self.fps) == 1 or (
                self.loading is not None and self.loading.done()):
            self.reload()
        if not self.frames or self.i >= len(self.frames):
            return
        img = self.render.render(self.frames[self.i])
        self.i += 1
        h, w = img.shape[:2]
        q = QImage(img.data, w, h, img.strides[0],
                   QImage.Format_BGR888).copy()
        self.frame = QPixmap.fromImage(q)
        self._show()

    def _show(self) -> None:
        if self.frame is not None:
            self.setPixmap(self.frame.scaled(self.size(), Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._show()


def window(wd, *, solve_args: list[str] | None = None, fps: int = 10,
           width: int = 1280):
    """The bare live view in a window (not shown yet); with `solve_args`
    it starts `skyfind solve` too. `run` shows it and runs the loop."""
    app = QApplication.instance() or QApplication(sys.argv)

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"skyfind — {wd.clip.name}")
            self.view = LiveView(wd, fps=fps, width=width)
            self.setCentralWidget(self.view)
            self.resize(width, int(width * 0.75))
            self.proc = None
            if solve_args is not None:
                self.proc = subprocess.Popen(
                    [sys.executable, "-m", "skyfind", "solve", str(wd.clip),
                     "--workdir", str(wd.dir), *solve_args])

        # the view's state, as the window's (tests, callers)
        story = property(lambda self: self.view.story)
        frames = property(lambda self: self.view.frames)
        label = property(lambda self: self.view)

        def tick(self):
            self.view.tick()

        def closeEvent(self, ev):
            self.view.timer.stop()
            super().closeEvent(ev)

    return app, Window()


def run(wd, *, solve_args: list[str] | None = None, fps: int = 10,
        width: int = 1280) -> int:
    app, w = window(wd, solve_args=solve_args, fps=fps, width=width)
    w.show()
    code = app.exec()
    if w.proc is not None and w.proc.poll() is None:
        print("the solve is still running in the background "
              f"(pid {w.proc.pid}); it keeps writing to {wd.dir}")
    return code
