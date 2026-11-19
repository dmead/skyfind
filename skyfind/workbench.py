"""The skyfind workbench: the whole workflow in one window — pick a clip,
find its trails, choose where the orbital elements come from, say roughly
when (or let planets or comets), solve while the picture fills in, then
write the report and a preview.

Each step runs `python -m skyfind <step>` as a child process, so the window
stays live, the log shows exactly what the command line would, and Stop
ends the step's whole process tree (the solve's workers too). The window
computes nothing itself; it assembles the commands the CLI already takes.

    skyfind gui [FOLDER or CLIP]        (or gui.cmd in the repository)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QDate, QDateTime, QProcess, QProcessEnvironment, \
    QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QCursor, QDesktopServices, QPixmap, QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup,
                               QCheckBox, QDateEdit, QDateTimeEdit, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QRadioButton,
                               QScrollArea, QSizePolicy, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget)

from . import sources as src
from .gui import LiveView
from .pipeline import Workdir
from .sources import SPACETRACK, Source

VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
UTC_FMT = "yyyy-MM-dd HH:mm"
DAY_FMT = "yyyy-MM-dd"
SCAN_LINE = re.compile(r"scan: night (\d+) of (\d+) \(([\d-]+)\)")
# the report's pictures, as tabs beside the live view
REPORT = (("trails", "_satellites.jpg"), ("close-up", "_satellites_closeup.jpg"),
          ("site map", "_site_map.png"))


# ================================================================ clips
def workdir_of(clip: Path) -> Path:
    return clip.with_name(clip.stem + ".skyfind")


def list_clips(folder: Path) -> list[Path]:
    """The videos in a folder (skyfind writes nothing next to them: its
    output is in each clip's work folder)."""
    if not folder or not folder.is_dir():
        return []
    return [p for p in sorted(folder.iterdir(), key=lambda p: p.name.lower())
            if p.suffix.lower() in VIDEO and p.is_file()]


def clip_status(clip: Path) -> tuple[str, str]:
    """(stage, one line): stage is new, trails or solved."""
    wd = workdir_of(clip)
    sol = wd / "solution.json"
    if sol.exists():
        try:
            s = json.loads(sol.read_text())
            ns, ew = ("N" if s["lat"] >= 0 else "S"), \
                ("E" if s["lon"] >= 0 else "W")
            conf = sum(o.get("conf") == "confirmed" for o in s["objects"])
            from .report import took
            return "solved", (f"{abs(s['lat']):.3f}{ns} {abs(s['lon']):.3f}"
                              f"{ew}  {s['frame0_utc'][:16]} UTC  "
                              f"{len(s['objects'])} objects ({conf} confirmed)"
                              + (f"  in {took(s['elapsed_s'])}"
                                 if "elapsed_s" in s else ""))
        except (OSError, ValueError, KeyError):
            return "solved", "solved"
    if (wd / "segments.json").exists():
        try:
            n = sum(len(t["frames"]) >= 3
                    for t in json.loads((wd / "tracks.json").read_text()))
            return "trails", f"trails found: {n} tracks over 3+ frames"
        except (OSError, ValueError):
            return "trails", "trails found"
    return "new", "not started"


def clip_date(clip: Path, source: bool = False):
    """The clip's date once known (given, or dated from planets or comets);
    with `source`, (date, "given" | "planets" | "comets" | ...)."""
    f = workdir_of(clip) / "date.json"
    try:
        d = json.loads(f.read_text())
        t = datetime.strptime(d["utc"][:16], "%Y-%m-%d %H:%M").replace(
            tzinfo=timezone.utc)
    except (OSError, ValueError, KeyError):
        return (None, None) if source else None
    return (t, d.get("source")) if source else t


def step_args(step: str, clip: Path, extra: list[str]) -> list[str]:
    return ["-m", "skyfind", step, str(clip), "--workdir",
            str(workdir_of(clip)), *extra]


class ImageView(QLabel):
    """A picture scaled to fit, or a line saying what will be here."""

    def __init__(self, empty: str):
        super().__init__()
        self.empty, self.src = empty, None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background: #121416; color: #9aa;")
        self.set_path(None)

    def set_path(self, path: Path | None) -> None:
        self.src = QPixmap(str(path)) if path and path.exists() else None
        if self.src is None or self.src.isNull():
            self.src = None
            self.setPixmap(QPixmap())
            self.setText(self.empty)
        self._fit()

    def _fit(self) -> None:
        if self.src is not None:
            self.setPixmap(self.src.scaled(self.size(), Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit()


# ================================================================ sources
class SourcesDialog(QDialog):
    """Which orbital elements to use, in which order, and the Space-Track
    account. Later sources replace earlier ones object by object."""

    def __init__(self, parent, sources: list[Source], user: str,
                 password: str | None, when: datetime | None):
        super().__init__(parent)
        self.setWindowTitle("Orbital elements")
        self.when = when
        self.resize(640, 560)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Tick the sources to use. Order matters: a source lower in the "
            "list is preferred for the objects it has, wherever its elements "
            "are current for the clip's night — put the general catalogue "
            "first, then the better sources for some objects (Starlink/"
            "OneWeb supplemental orbits, classified objects). Stale files do "
            "no harm. Drag to reorder."))
        lay.itemAt(0).widget().setWordWrap(True)
        self.list = QListWidget(self)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setAlternatingRowColors(True)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (("Add files…", self._add), ("Remove", self._remove),
                         ("Up", lambda: self._move(-1)),
                         ("Down", lambda: self._move(1))):
            b = QPushButton(text, self)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        acct = QGroupBox("Space-Track account (for the download)", self)
        form = QFormLayout(acct)
        self.user = QLineEdit(user, acct)
        self.pw = QLineEdit(password or "", acct)
        self.pw.setEchoMode(QLineEdit.Password)
        self.remember = QCheckBox("Remember the password in Windows "
                                  "Credential Manager", acct)
        self.remember.setChecked(bool(password) and src.can_remember())
        self.remember.setEnabled(src.can_remember())
        form.addRow("user name", self.user)
        form.addRow("password", self.pw)
        form.addRow("", self.remember)
        note = QLabel(
            "Space-Track has no API keys: skyfind signs in with your "
            "account the way the website does, downloads the night's "
            "elements once and keeps them. The password is only ever passed "
            "to the solve, never written to a file. "
            f"<a href='{src.signup_url()}'>Make a free account</a>.", acct)
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        form.addRow(note)
        lay.addWidget(acct)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        for s in sources:
            self._item(s)

    def _describe(self, path: str) -> str:
        if path == SPACETRACK:
            day = f" ({self.when:%Y-%m-%d})" if self.when else ""
            return ("Space-Track — the clip's night" + day +
                    "\n  the full catalogue's history; downloaded once with "
                    "your account, then cached")
        p = Path(path)
        if not p.exists():
            return f"{p.name}\n  missing: {p}"
        try:
            s = src.cached_summary(p)
        except OSError as e:
            return f"{p.name}\n  can't read: {e}"
        badge = ""
        if self.when is not None and s.sets:
            badge = ("   ✓ covers the clip's date" if s.covers(self.when)
                     else f"   ⚠ not near the clip's date {self.when:%Y-%m-%d}")
        return f"{p.name}\n  {s.text()}{badge}"

    def _item(self, s: Source) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            it = QListWidgetItem(self._describe(s.path))
        finally:
            QApplication.restoreOverrideCursor()
        it.setData(Qt.UserRole, s.path)
        it.setToolTip(s.path)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Checked if s.enabled else Qt.Unchecked)
        self.list.addItem(it)

    def _add(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Element files", "", "TLE text (*.3le *.tle *.txt);;"
                                       "all files (*)")
        have = set(self.paths())
        for f in files:
            if f not in have:
                self._item(Source(f, True))

    def _remove(self) -> None:
        for it in self.list.selectedItems():
            if it.data(Qt.UserRole) != SPACETRACK:
                self.list.takeItem(self.list.row(it))

    def _move(self, d: int) -> None:
        r = self.list.currentRow()
        if r < 0 or not 0 <= r + d < self.list.count():
            return
        it = self.list.takeItem(r)
        self.list.insertItem(r + d, it)
        self.list.setCurrentRow(r + d)

    def paths(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole)
                for i in range(self.list.count())]

    def result_sources(self) -> list[Source]:
        return [Source(self.list.item(i).data(Qt.UserRole),
                       self.list.item(i).checkState() == Qt.Checked)
                for i in range(self.list.count())]


# ================================================================ window
class Workbench(QMainWindow):
    def __init__(self, start: Path | None = None, settings=None):
        super().__init__()
        self.setWindowTitle("skyfind — where and when, from the satellites")
        self.settings = settings or QSettings("skyfind", "workbench")
        self.sources, self.st_user = src.load()
        self.st_password = src.password(self.st_user)   # keyring, if kept
        self.clip: Path | None = None
        self.proc: QProcess | None = None
        self.queue: list[tuple[str, list[str]]] = []
        self.step = ""
        self._build()
        folder = self.settings.value("folder", "")
        if start is not None and start.is_file():
            self.set_folder(start.parent)
            self.select_clip(start)
        else:
            self.set_folder(start if start is not None else
                            Path(folder) if folder else None)
        self._place()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(500)

    # ------------------------------------------------------------ placement
    def _place(self) -> None:
        """The last size and place when that is still wholly on one screen;
        else most of the screen under the pointer, centred. (Screens with
        different scaling side by side can leave a window straddling them,
        half off both.)"""
        g = self.settings.value("geometry")
        if g is not None and self.restoreGeometry(g) and self._on_one_screen():
            return
        scr = QApplication.screenAt(QCursor.pos()) or \
            QApplication.primaryScreen()
        a = scr.availableGeometry()
        w = min(1500, int(a.width() * 0.92))
        h = min(950, int(a.height() * 0.9))
        self.setGeometry(a.x() + (a.width() - w) // 2,
                         a.y() + (a.height() - h) // 2, w, h)

    def _on_one_screen(self) -> bool:
        f = self.geometry()
        return any(s.availableGeometry().contains(f)
                   for s in QApplication.screens())

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        if not getattr(self, "_fitted", False):
            self._fitted = True
            QTimer.singleShot(0, self.fit_to_screen)

    def fit_to_screen(self) -> None:
        """Once shown: pull the window wholly inside the screen it landed
        on. Windows rescales a window that lands on a screen with other
        scaling, which can push it past that screen's edge."""
        if self.isMaximized() or self.isFullScreen():
            return
        a = self.screen().availableGeometry()
        f, g = self.frameGeometry(), self.geometry()
        fw, fh = f.width() - g.width(), f.height() - g.height()
        w = min(g.width(), a.width() - fw)
        h = min(g.height(), a.height() - fh)
        x = min(max(f.x(), a.x()), a.x() + a.width() - (w + fw))
        y = min(max(f.y(), a.y()), a.y() + a.height() - (h + fh))
        if (w, h) != (g.width(), g.height()):
            self.resize(w, h)
        if (x, y) != (f.x(), f.y()):
            self.move(x, y)

    # ------------------------------------------------------------ layout
    def _build(self) -> None:
        panel = QWidget()
        lv = QVBoxLayout(panel)

        g1 = QGroupBox("① clip")
        v = QVBoxLayout(g1)
        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        b = QPushButton("Folder…")
        b.clicked.connect(self._pick_folder)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(b)
        v.addLayout(row)
        self.clips = QListWidget()
        self.clips.setMinimumHeight(170)
        self.clips.currentItemChanged.connect(self._clip_changed)
        v.addWidget(self.clips)
        b = QPushButton("Open a clip from anywhere…")
        b.clicked.connect(self._pick_clip)
        v.addWidget(b)
        self.clip_info = QLabel("")
        self.clip_info.setWordWrap(True)
        v.addWidget(self.clip_info)
        lv.addWidget(g1)

        g2 = QGroupBox("② trails")
        v = QVBoxLayout(g2)
        row = QHBoxLayout()
        self.trails_btn = QPushButton("Find trails")
        self.trails_btn.clicked.connect(self.find_trails)
        row.addWidget(self.trails_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.trails_info = QLabel("")
        v.addWidget(self.trails_info)
        lv.addWidget(g2)

        g3 = QGroupBox("③ orbital elements")
        v = QHBoxLayout(g3)
        self.sources_info = QLabel("")
        self.sources_info.setWordWrap(True)
        b = QPushButton("Sources…")
        b.clicked.connect(self.edit_sources)
        v.addWidget(self.sources_info, 1)
        v.addWidget(b)
        lv.addWidget(g3)

        g4 = QGroupBox("④ when")
        f = QFormLayout(g4)
        self.by_planets = QRadioButton("date it from planets or comets in "
                                       "view")
        self.given = QRadioButton("around (UTC)")
        self.span = QRadioButton("some night from")
        self.span.setToolTip("only roughly known: every night in between is "
                             "searched on its own (a few minutes a night), "
                             "then the best gets the full solve")
        grp = QButtonGroup(g4)
        for b in (self.by_planets, self.given, self.span):
            grp.addButton(b)
        today = QDate.currentDate()             # not Qt's 2000-01-01
        self.span_from, self.span_to = QDateEdit(today), QDateEdit(today)
        for d in (self.span_from, self.span_to):
            d.setDisplayFormat(DAY_FMT)
            d.setCalendarPopup(True)
        self.when = QDateTimeEdit(QDateTime(today, QDateTime.currentDateTime()
                                            .time()))
        self.when.setDisplayFormat(UTC_FMT)
        self.when.setCalendarPopup(True)
        self.window_h = QDoubleSpinBox()
        self.window_h.setRange(0.25, 12.0)
        self.window_h.setSingleStep(0.5)
        self.window_h.setSuffix(" h")
        self.window_h.setToolTip("how far either side to search: 12 h "
                                 "covers every longitude")
        f.addRow(self.by_planets)
        row = QHBoxLayout()
        row.addWidget(self.given)
        row.addWidget(self.when, 1)
        f.addRow(row)
        f.addRow("search ±", self.window_h)
        row = QHBoxLayout()
        row.addWidget(self.span)
        row.addWidget(self.span_from, 1)
        row.addWidget(QLabel("to"))
        row.addWidget(self.span_to, 1)
        f.addRow(row)
        for w in (self.by_planets, self.given, self.span):
            w.toggled.connect(self._when_changed)
        self.when.dateTimeChanged.connect(self._when_changed)
        self.window_h.valueChanged.connect(self._when_changed)
        for d in (self.span_from, self.span_to):
            d.dateChanged.connect(self._when_changed)
        lv.addWidget(g4)

        g5 = QGroupBox("⑤ solve")
        v = QVBoxLayout(g5)
        row = QHBoxLayout()
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(lambda: self.start([self._solve()]))
        row.addWidget(self.solve_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.free_site = QCheckBox("ignore the skyline: find the site from "
                                   "the satellites alone")
        self.free_site.setToolTip(
            "a ragged treeline can put the skyline's latitude degrees off, "
            "and the search only looks a few degrees around it; try this "
            "when no site stands out (it runs on the GPU when there is one)")
        self.free_site.toggled.connect(self._free_changed)
        v.addWidget(self.free_site)
        self.solve_info = QLabel("")
        self.solve_info.setWordWrap(True)
        v.addWidget(self.solve_info)
        lv.addWidget(g5)

        g6 = QGroupBox("⑥ results")
        v = QHBoxLayout(g6)
        self.report_btn = QPushButton("Report")
        self.report_btn.clicked.connect(lambda: self.start([self._report()]))
        self.video_btn = QPushButton("Preview video")
        self.video_btn.clicked.connect(lambda: self.start([self._video()]))
        self.gif_btn = QPushButton("GIF")
        self.gif_btn.clicked.connect(lambda: self.start([self._gif()]))
        self.open_btn = QPushButton("Open folder")
        self.open_btn.clicked.connect(self._open_folder)
        for b in (self.report_btn, self.video_btn, self.gif_btn,
                  self.open_btn):
            v.addWidget(b)
        lv.addWidget(g6)

        run = QGroupBox("run")
        v = QVBoxLayout(run)
        row = QHBoxLayout()
        self.all_btn = QPushButton("Run every step")
        self.all_btn.setToolTip("trails (if not found yet), solve, report "
                                "and the preview video")
        self.all_btn.clicked.connect(self.run_all)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        row.addWidget(self.all_btn, 1)
        row.addWidget(self.stop_btn)
        v.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        v.addWidget(self.bar)
        self.step_info = QLabel("idle")
        v.addWidget(self.step_info)
        lv.addWidget(run)
        lv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)

        self.view = LiveView()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.view, "solve")
        self.pictures = {}
        for name, _ in REPORT:
            self.pictures[name] = ImageView(f"the {name} appears here after "
                                            "⑥ Report")
            self.tabs.addTab(self.pictures[name], name)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setPlaceholderText("each step's output appears here, as "
                                    "the command line would show it")
        self.log.setStyleSheet("font-family: Consolas, monospace; "
                               "font-size: 9pt;")
        right = QSplitter(Qt.Vertical)
        right.addWidget(self.tabs)
        right.addWidget(self.log)
        right.setSizes([700, 220])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([450, 1050])
        self.setCentralWidget(split)
        self._show_sources()
        self._update_buttons()

    # ------------------------------------------------------------ clips
    def set_folder(self, folder: Path | None) -> None:
        self.folder = folder
        self.folder_edit.setText(str(folder) if folder else "")
        if folder:
            self.settings.setValue("folder", str(folder))
        self.refresh_clips()

    def refresh_clips(self) -> None:
        cur = self.clip
        self.clips.blockSignals(True)
        self.clips.clear()
        for c in list_clips(self.folder) if self.folder else []:
            stage, text = clip_status(c)
            mark = {"solved": "●", "trails": "◐", "new": "○"}[stage]
            it = QListWidgetItem(f"{mark} {c.name}\n    {text}")
            it.setData(Qt.UserRole, str(c))
            self.clips.addItem(it)
            if cur is not None and c == cur:
                self.clips.setCurrentItem(it)
        self.clips.blockSignals(False)

    def select_clip(self, clip: Path) -> None:
        for i in range(self.clips.count()):
            if Path(self.clips.item(i).data(Qt.UserRole)) == clip:
                self.clips.setCurrentRow(i)
                return
        self._set_clip(clip)                 # not in the listed folder

    def _clip_changed(self, it, _prev=None) -> None:
        if it is not None:
            self._set_clip(Path(it.data(Qt.UserRole)))

    def _set_clip(self, clip: Path) -> None:
        self.clip = clip
        wd = workdir_of(clip)
        self.clip_info.setText(f"{clip}\nwork folder: {wd.name}")
        self.view.set_workdir(Workdir(clip) if wd.exists() else None)
        self._show_pictures()
        self._load_when()
        self._load_free()
        self._show_status()
        self._show_sources()
        self._update_buttons()

    def _show_pictures(self) -> None:
        for name, suffix in REPORT:
            self.pictures[name].set_path(
                workdir_of(self.clip) / f"{self.clip.stem}{suffix}"
                if self.clip else None)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder of clips",
                                             str(self.folder or ""))
        if d:
            self.set_folder(Path(d))

    def _pick_clip(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "Clip", str(self.folder or ""),
            "videos (" + " ".join(f"*{e}" for e in sorted(VIDEO)) + ")")
        if f:
            p = Path(f)
            if p.parent != self.folder:
                self.set_folder(p.parent)
            self.select_clip(p)

    def _show_status(self) -> None:
        if self.clip is None:
            for w in (self.trails_info, self.solve_info):
                w.setText("")
            return
        stage, text = clip_status(self.clip)
        self.trails_info.setText(text if stage == "trails" else
                                 "done" if stage == "solved" else
                                 "not yet: cleans the clip and finds the "
                                 "trail segments")
        self.solve_info.setText(text if stage == "solved" else "")

    # ------------------------------------------------------------ when
    def _when_key(self) -> str:
        return "when/" + hashlib.sha1(str(self.clip).encode()).hexdigest()

    def _load_when(self) -> None:
        d = self.settings.value(self._when_key())
        given, source = clip_date(self.clip, source=True)
        f = t = ""
        if d:
            mode, utc, win, *span = d.split("|")
            f, t = span if len(span) == 2 else ("", "")
        elif given is not None:
            # a date the planets or comets gave stays theirs: re-solving
            # "around" it would record it as given (and lose what dated it)
            mode = "given" if source == "given" else "planets"
            utc, win = f"{given:%Y-%m-%d %H:%M}", "12"
        else:
            mode, utc, win = "planets", "", "12"
        ws = (self.by_planets, self.given, self.span, self.when,
              self.window_h, self.span_from, self.span_to)
        for w in ws:
            w.blockSignals(True)
        self.by_planets.setChecked(mode == "planets")
        self.given.setChecked(mode == "given")
        self.span.setChecked(mode == "span")
        if utc:
            self.when.setDateTime(QDateTime.fromString(utc, UTC_FMT))
        self.window_h.setValue(float(win))
        if f and t:
            self.span_from.setDate(QDate.fromString(f, DAY_FMT))
            self.span_to.setDate(QDate.fromString(t, DAY_FMT))
        for w in ws:
            w.blockSignals(False)
        self._enable_when()

    def _enable_when(self) -> None:
        self.when.setEnabled(self.given.isChecked())
        self.window_h.setEnabled(not self.span.isChecked())
        for d in (self.span_from, self.span_to):
            d.setEnabled(self.span.isChecked())

    def _when_changed(self, *_) -> None:
        self._enable_when()
        if self.clip is not None:
            mode = "given" if self.given.isChecked() else \
                "span" if self.span.isChecked() else "planets"
            self.settings.setValue(self._when_key(), "|".join((
                mode, self.when.dateTime().toString(UTC_FMT),
                f"{self.window_h.value():g}",
                self.span_from.date().toString(DAY_FMT),
                self.span_to.date().toString(DAY_FMT))))

    def when_args(self) -> list[str]:
        if self.span.isChecked():
            return ["--from", self.span_from.date().toString(DAY_FMT),
                    "--to", self.span_to.date().toString(DAY_FMT)]
        if not self.given.isChecked():
            return ["--window", f"{self.window_h.value():g}"]
        return ["--around", self.when.dateTime().toString(UTC_FMT),
                "--window", f"{self.window_h.value():g}"]

    def _span(self) -> tuple[datetime, datetime]:
        def day(d):
            return datetime.strptime(d.date().toString(DAY_FMT),
                                     "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return day(self.span_from), day(self.span_to)

    def _clip_when(self) -> datetime | None:
        if self.span.isChecked():
            a, b = self._span()
            return a + (b - a) / 2
        if self.given.isChecked():
            return datetime.strptime(self.when.dateTime().toString(UTC_FMT),
                                     "%Y-%m-%d %H:%M").replace(
                                         tzinfo=timezone.utc)
        return clip_date(self.clip) if self.clip else None

    # ------------------------------------------------------------ sources
    def _show_sources(self) -> None:
        on = [s for s in self.sources if s.enabled]
        names = ["Space-Track" if s.is_spacetrack else Path(s.path).name
                 for s in on]
        text = ", ".join(names) if names else \
            "none ticked: the solve looks for a cached Space-Track download"
        if any(s.is_spacetrack for s in on) and not self.st_user:
            text += "  (needs your Space-Track account)"
        self.sources_info.setText(text)

    def edit_sources(self) -> bool:
        dlg = SourcesDialog(self, self.sources, self.st_user,
                            self.st_password, self._clip_when())
        if dlg.exec() != QDialog.Accepted:
            return False
        self.sources = dlg.result_sources()
        user, pw = dlg.user.text().strip(), dlg.pw.text()
        if user != self.st_user:
            src.forget(self.st_user)
        self.st_user, self.st_password = user, pw or None
        if dlg.remember.isChecked() and user and pw:
            src.remember(user, pw)
        elif not dlg.remember.isChecked():
            src.forget(user)
        src.save(self.sources, self.st_user)
        self._show_sources()
        return True

    def _needs_account(self) -> bool:
        on = [s for s in self.sources if s.enabled]
        return any(s.is_spacetrack for s in on) or not on

    # ------------------------------------------------------------ steps
    def _trails(self, reclean: bool = False):
        return ("trails", step_args("trails", self.clip,
                                    ["--reclean"] if reclean else []))

    def find_trails(self) -> None:
        """Trails already found: start over, or just extract them again
        from what the work folder has."""
        reclean = False
        if clip_status(self.clip)[0] != "new":
            box = QMessageBox(self)
            box.setWindowTitle("Find trails")
            box.setText("This clip's trails were found before.")
            box.setInformativeText(
                "Start over tracks the sky and takes the trails out again "
                "(minutes; worth it after a skyfind update). Extract again "
                "reuses that work and only finds the segments.")
            over = box.addButton("Start over", QMessageBox.AcceptRole)
            again = box.addButton("Extract again", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() not in (over, again):
                return
            reclean = box.clickedButton() is over
        self.start([self._trails(reclean)])

    def _solve(self):
        step = "scan" if self.span.isChecked() else "solve"
        free = ["--free-site"] if self.free_site.isChecked() else []
        return (step, step_args(step, self.clip, src.solve_args(self.sources)
                                + self.when_args() + free))

    def _free_key(self) -> str:
        return "free/" + hashlib.sha1(str(self.clip).encode()).hexdigest()

    def _load_free(self) -> None:
        on = self.settings.value(self._free_key(), False)
        self.free_site.blockSignals(True)
        self.free_site.setChecked(on in (True, "true", "1"))
        self.free_site.blockSignals(False)

    def _free_changed(self, on: bool) -> None:
        if self.clip is not None:
            self.settings.setValue(self._free_key(), bool(on))

    def _report(self):
        return ("report", step_args("report", self.clip, []))

    def _video(self):
        out = workdir_of(self.clip) / f"{self.clip.stem}_solve.mp4"
        return ("preview video", step_args(
            "replay", self.clip, ["--no-gif", "--mp4", str(out),
                                  "--width", "1280"]))

    def _gif(self):
        return ("GIF", step_args("replay", self.clip, ["--width", "960"]))

    def run_all(self) -> None:
        steps = []
        if clip_status(self.clip)[0] == "new":
            steps.append(self._trails())
        steps += [self._solve(), self._report(), self._video()]
        self.start(steps)

    def start(self, steps: list) -> None:
        if self.clip is None or self.proc is not None:
            return
        if any(n == "scan" for n, _ in steps) and not self._scan_sources():
            return
        steps = [self._solve() if n in ("solve", "scan") else (n, a)
                 for n, a in steps]
        if any(n in ("solve", "scan") for n, _ in steps) and \
                self._needs_account() and not self.st_password and \
                not src.has_env_account():
            if not self._ask_account():
                return
        self.queue = list(steps)
        self._next()

    def _scan_sources(self) -> bool:
        """A span of nights needs elements for every one of them: when the
        ticked files don't cover the span, offer Space-Track's."""
        a, b = self._span()
        if b < a:
            QMessageBox.warning(self, "When", "The last night is before the "
                                "first.")
            return False
        on = [s for s in self.sources if s.enabled]
        if any(s.is_spacetrack for s in on):
            return True
        files = [Path(s.path) for s in on if Path(s.path).exists()]
        if any(src.cached_summary(f).covers(a) and
               src.cached_summary(f).covers(b) for f in files):
            return True
        r = QMessageBox.question(
            self, "Orbital elements",
            f"None of the ticked sources covers {a:%Y-%m-%d} to "
            f"{b:%Y-%m-%d}. Use Space-Track's elements for those nights "
            "(downloaded a week at a time with your account, then kept)?",
            QMessageBox.Yes | QMessageBox.Cancel)
        if r != QMessageBox.Yes:
            return False
        for s in self.sources:
            if s.is_spacetrack:
                s.enabled = True
        src.save(self.sources, self.st_user)
        self._show_sources()
        return True

    def _ask_account(self) -> bool:
        r = QMessageBox.question(
            self, "Space-Track",
            "The solve may need to download the night's orbital elements "
            "from Space-Track, which needs your account. Add it now?\n\n"
            "(No: go on anyway — fine when they were downloaded before.)",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Yes:
            return self.edit_sources()
        return True

    def _env(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        for k, v in src.environment(self.st_user, self.st_password).items():
            env.insert(k, v)
        return env

    def _next(self) -> None:
        if not self.queue:
            self.step = ""
            self._finish_all()
            return
        self.step, args = self.queue.pop(0)
        self.scan_at = None
        if self.step in ("solve", "scan") or \
                not workdir_of(self.clip).exists():
            self.view.set_workdir(Workdir(self.clip))
        self.proc = QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setArguments(args)
        self.proc.setProcessEnvironment(self._env())
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self._log(f"\n$ skyfind {' '.join(_quote(a) for a in args[2:])}\n")
        self.proc.start()
        self._update_buttons()

    def _read(self) -> None:
        raw = bytes(self.proc.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        # progress bars rewrite their line: keep the last state of each
        text = "\n".join(ln.split("\r")[-1]
                         for ln in raw.replace("\r\n", "\n").split("\n"))
        for m in SCAN_LINE.finditer(text):
            self.scan_at = (int(m.group(1)), int(m.group(2)), m.group(3))
        if "full solve on" in text:
            self.scan_at = None                 # the picture takes over
        self._log(text)

    def _log(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    def _finished(self, code, _status) -> None:
        self.proc = None
        ok = code == 0
        self._log(f"[{self.step} {'done' if ok else f'failed (exit {code})'}]\n")
        self.refresh_clips()
        self._show_status()
        self._show_pictures()
        if ok and self.step == "report":
            self.tabs.setCurrentIndex(1)
        if not ok:
            self.queue = []
            self.step_info.setText(f"{self.step} failed — see the log")
            tail = self.log.toPlainText()[-2000:]
            if self.step == "solve" and "no site stands out" in tail:
                self.step_info.setText("no site stood out (chance agreements "
                                       "only): most likely the wrong night — "
                                       "try some night from … to in ④")
            if "refused the sign-in" in tail:
                self.step_info.setText("Space-Track refused the sign-in: "
                                       "check the user name and password in "
                                       "③ Sources…")
            if self.step == "scan" and "stands out" in tail:
                self.step_info.setText("no night stood out: check the dates "
                                       "in ④ and the sources in ③")
            if self.step == "solve" and "date the clip from planets" in tail:
                self.step_info.setText(
                    "no planets or comets to date it by: in ④ give the date "
                    "(around) "
                    "or the nights it could be (some night from … to), then "
                    "solve again")
                self.given.setFocus()
            if self.step == "solve" and "The Moon is in view" in tail:
                self.step_info.setText(
                    "no planets or comets, but the Moon: it stood there once "
                    "a lunar month (see the log). In ④ give a span (some "
                    "night from … to): only the Moon's nights get tried")
                self.span.setFocus()
        if ok:
            self._next()
        else:
            self._update_buttons()

    def _finish_all(self) -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        if not self.step_info.text().endswith("see the log"):
            self.step_info.setText("done")
        self.view.restart()
        self._update_buttons()

    def stop(self) -> None:
        if self.proc is None:
            return
        self.queue = []
        _kill_tree(self.proc)
        self._log("[stopped]\n")

    # ------------------------------------------------------------ state
    def _poll(self) -> None:
        if self.proc is None:
            return
        if self.step == "scan" and self.scan_at is not None:
            i, n, night = self.scan_at
            self.bar.setRange(0, n)
            self.bar.setValue(i - 1)
            self.step_info.setText(f"scan: night {i} of {n} ({night})")
        elif self.step in ("solve", "scan"):
            done, n, solved = self.view.progress()
            if solved:
                self.bar.setRange(0, 0)
                self.step_info.setText("solved: identifying every trail…")
            elif n:
                self.bar.setRange(0, n)
                self.bar.setValue(done)
                self.step_info.setText(f"solve: {done} of {n} tracks fitted")
            else:
                self.bar.setRange(0, 0)
                self.step_info.setText("solve: stars, camera, orbits…")
        else:
            self.bar.setRange(0, 0)
            self.step_info.setText(f"{self.step}…")

    def _update_buttons(self) -> None:
        busy = self.proc is not None
        stage = clip_status(self.clip)[0] if self.clip else None
        have = self.clip is not None
        self.trails_btn.setEnabled(have and not busy)
        self.solve_btn.setEnabled(have and not busy and stage != "new")
        for b in (self.report_btn, self.video_btn, self.gif_btn):
            b.setEnabled(have and not busy and stage == "solved")
        self.open_btn.setEnabled(have and workdir_of(self.clip).exists())
        self.all_btn.setEnabled(have and not busy)
        self.stop_btn.setEnabled(busy)
        self.clips.setEnabled(not busy)

    def _open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(workdir_of(self.clip))))

    def closeEvent(self, ev) -> None:
        if self.proc is not None:
            r = QMessageBox.question(
                self, "skyfind", f"{self.step} is still running. Stop it and "
                "close?", QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                ev.ignore()
                return
            self.stop()
        self.settings.setValue("geometry", self.saveGeometry())
        self.timer.stop()
        self.view.timer.stop()
        super().closeEvent(ev)


def _quote(a: str) -> str:
    return f'"{a}"' if " " in a else a


def _kill_tree(proc: QProcess) -> None:
    """End a step and everything it started (the solve's worker pool)."""
    pid = proc.processId()
    if sys.platform == "win32" and pid:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True)
    else:
        proc.kill()


def run(start: Path | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("skyfind")
    w = Workbench(start)
    w.show()
    return app.exec()
