"""The workbench lists a folder's clips with their progress, assembles each
step's command from what's chosen, runs it as a child process, and keeps the
Space-Track password out of every file (offscreen Qt)."""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDateTime, QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from skyfind import sources, workbench  # noqa: E402
from skyfind.sources import SPACETRACK, Source  # noqa: E402

from .synth import constellation  # noqa: E402


def _clip(path):
    from remove_satellites.core.video import FrameWriter
    with FrameWriter(path, 320, 180, 24, crf=20) as fw:
        for _ in range(3):
            fw.write(np.full((180, 320, 3), 30, np.uint8))
    return path


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYFIND_DATA", str(tmp_path / "data"))
    d = tmp_path / "clips"
    d.mkdir()
    a, b = _clip(d / "alpha.mp4"), _clip(d / "beta.mov")
    wa = workbench.workdir_of(a)
    wa.mkdir()
    (wa / "segments.json").write_text("{}")
    (wa / "tracks.json").write_text(json.dumps([{"frames": [1, 2, 3]},
                                                {"frames": [7]}]))
    wb = workbench.workdir_of(b)
    wb.mkdir()
    (wb / "solution.json").write_text(json.dumps(dict(
        lat=34.748, lon=-119.112, frame0_utc="2026-07-20 04:37:25",
        objects=[{"conf": "confirmed"}, {"conf": "probable"}])))
    return d


@pytest.fixture
def window(folder, tmp_path):
    QApplication.instance() or QApplication([])
    ini = tmp_path / "workbench.ini"
    w = workbench.Workbench(folder, settings=QSettings(str(ini),
                                                      QSettings.IniFormat))
    w.ini = ini
    yield w
    w.timer.stop()
    w.view.timer.stop()


def _row(w, name):
    for i in range(w.clips.count()):
        if w.clips.item(i).text().split("\n")[0].endswith(name):
            return i
    raise KeyError(name)


def test_lists_the_clips_with_their_progress(window, folder):
    w = window
    assert w.clips.count() == 2
    texts = "\n".join(w.clips.item(i).text() for i in range(2))
    assert "trails found: 1 tracks" in texts
    assert "34.748N 119.112W" in texts and "2 objects (1 confirmed)" in texts
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    assert w.solve_btn.isEnabled() and not w.report_btn.isEnabled()
    w.clips.setCurrentRow(_row(w, "beta.mov"))
    assert w.report_btn.isEnabled() and w.video_btn.isEnabled()


def test_the_solve_command_follows_the_sources_and_the_date(window, folder):
    w = window
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    w.sources = [Source(SPACETRACK, True), Source("cat.3le"),
                 Source("sup.3le", False), Source("classified.tle")]
    w.given.setChecked(True)
    w.when.setDateTime(QDateTime.fromString("2026-07-20 00:00",
                                            workbench.UTC_FMT))
    w.window_h.setValue(12)
    name, args = w._solve()
    clip = folder / "alpha.mp4"
    assert name == "solve" and args == [
        "-m", "skyfind", "solve", str(clip), "--workdir",
        str(workbench.workdir_of(clip)), "--tle", "cat.3le", "--tle",
        "classified.tle", "--spacetrack", "--around", "2026-07-20 00:00",
        "--window", "12"]
    w.by_planets.setChecked(True)
    assert "--around" not in w._solve()[1]


def test_the_date_is_remembered_per_clip(window):
    w = window
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    w.given.setChecked(True)
    w.when.setDateTime(QDateTime.fromString("2024-10-02 06:00",
                                            workbench.UTC_FMT))
    w.clips.setCurrentRow(_row(w, "beta.mov"))
    assert w.by_planets.isChecked()
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    assert w.given.isChecked()
    assert w.when.dateTime().toString(workbench.UTC_FMT) == "2024-10-02 06:00"


def test_a_date_the_sky_gave_isn_t_passed_back_as_given(window, folder):
    """A clip dated by a comet opens on "planets or comets", so solving
    again doesn't turn the comet's date into a typed one."""
    w = window
    wb = workbench.workdir_of(folder / "beta.mov")
    (wb / "date.json").write_text(json.dumps(dict(
        source="comets", utc="2025-02-02 16:32", tt=2460709.19)))
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    w.clips.setCurrentRow(_row(w, "beta.mov"))
    assert w.by_planets.isChecked() and "--around" not in w._solve()[1]
    (wb / "date.json").write_text(json.dumps(dict(
        source="given", utc="2025-02-02 16:32", tt=2460709.19)))
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    w.clips.setCurrentRow(_row(w, "beta.mov"))
    assert w.given.isChecked()


def test_the_password_only_reaches_the_solve_s_environment(window, tmp_path):
    w = window
    w.st_user, w.st_password = "someone", "hunter2-not-in-files"
    env = w._env()
    assert env.value("SPACETRACK_USER") == "someone"
    assert env.value("SPACETRACK_PASSWORD") == "hunter2-not-in-files"
    sources.save(w.sources, w.st_user)
    w.settings.sync()
    for f in (tmp_path / "data").rglob("*"):
        if f.is_file():
            assert "hunter2" not in f.read_text(errors="ignore"), f
    assert "hunter2" not in w.ini.read_text(errors="ignore")


def test_the_sources_dialog_describes_and_reorders(tmp_path):
    QApplication.instance() or QApplication([])
    epoch = datetime(2025, 2, 14, tzinfo=timezone.utc)
    els, _ = constellation(epoch)
    f = tmp_path / "cat.3le"
    f.write_text("".join(f"{e.name}\n{e.line1}\n{e.line2}\n" for e in els))
    dlg = workbench.SourcesDialog(None, [Source(SPACETRACK, False),
                                         Source(str(f))], "", None, epoch)
    assert dlg.list.count() == 2
    assert "objects" in dlg.list.item(1).text()
    assert "covers the clip's date" in dlg.list.item(1).text()
    dlg.list.setCurrentRow(1)
    dlg._move(-1)
    got = dlg.result_sources()
    assert [s.path for s in got] == [str(f), SPACETRACK]
    assert [s.enabled for s in got] == [True, False]


def test_a_step_runs_as_a_child_process_and_reports_failure(window, folder):
    w = window
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    # a report with no solution fails: the window says so and stays usable
    w.start([w._report()])
    assert w.proc is not None and not w.trails_btn.isEnabled()
    app = QApplication.instance()
    t0 = time.time()
    while w.proc is not None and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.05)
    assert w.proc is None
    log = w.log.toPlainText()
    assert "$ skyfind report" in log and "[report failed" in log
    assert w.trails_btn.isEnabled() and "failed" in w.step_info.text()


def test_finding_trails_again_asks_whether_to_start_over(window, folder,
                                                        monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    w = window
    started = []
    monkeypatch.setattr(w, "start", lambda steps: started.append(steps))
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))          # trails found

    def answer(label):
        def exec_(box):
            for b in box.buttons():
                if b.text() == label:
                    box._clicked = b
            return 0
        return exec_

    monkeypatch.setattr(QMessageBox, "exec", answer("Start over"))
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda box: getattr(box, "_clicked", None))
    w.find_trails()
    assert started[-1][0][1][-1] == "--reclean"
    monkeypatch.setattr(QMessageBox, "exec", answer("Extract again"))
    w.find_trails()
    assert "--reclean" not in started[-1][0][1]
    assert all("--fresh" not in a and "--cleaned" not in a
               for s in started for _, a in s)


def test_the_window_opens_wholly_on_one_screen(folder, tmp_path):
    """A saved place off every screen (a monitor unplugged, two scalings
    side by side) isn't restored: the window comes up on a screen."""
    QApplication.instance() or QApplication([])
    ini = QSettings(str(tmp_path / "w.ini"), QSettings.IniFormat)
    w = workbench.Workbench(folder, settings=ini)
    w.setGeometry(-3000, -40, 3000, 1900)
    ini.setValue("geometry", w.saveGeometry())
    w.timer.stop()
    w.view.timer.stop()
    w2 = workbench.Workbench(folder, settings=ini)
    assert w2._on_one_screen()
    w2.timer.stop()
    w2.view.timer.stop()


def test_some_night_in_a_month_becomes_a_scan(window, folder):
    from PySide6.QtCore import QDate
    w = window
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    w.sources = [Source(SPACETRACK, True)]
    w.span.setChecked(True)
    w.span_from.setDate(QDate(2026, 3, 1))
    w.span_to.setDate(QDate(2026, 3, 31))
    name, args = w._solve()
    assert name == "scan"
    assert args[2] == "scan" and args[-4:] == ["--from", "2026-03-01",
                                               "--to", "2026-03-31"]
    assert "--window" not in args and "--around" not in args
    assert not w.window_h.isEnabled() and w.span_from.isEnabled()
    # remembered with the clip
    w.clips.setCurrentRow(_row(w, "beta.mov"))
    assert not w.span.isChecked()
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    assert w.span.isChecked()
    assert w.span_to.date() == QDate(2026, 3, 31)
    # progress comes from the scan's log lines
    m = workbench.SCAN_LINE.search("scan: night 7 of 31 (2026-03-07)\n")
    assert m.groups() == ("7", "31", "2026-03-07")


def test_a_window_past_its_screen_s_edge_is_pulled_back(window):
    w = window
    a = w.screen().availableGeometry()
    w.show()
    w.setGeometry(a.x() + a.width() // 2, a.y() + 40, a.width(), a.height())
    w.fit_to_screen()
    f = w.frameGeometry()
    # (the test screen is narrower than the window can shrink to)
    widest = max(a.width(), w.minimumSizeHint().width())
    assert a.contains(f.topLeft()) and f.width() <= widest + 20
    assert f.bottom() <= a.bottom() + 1
    w.hide()


def test_dates_start_at_today_not_2000(window):
    from PySide6.QtCore import QDate
    assert window.span_from.date() == QDate.currentDate()
    assert window.when.date() == QDate.currentDate()


def test_ignoring_the_skyline_is_passed_on_and_remembered(window):
    w = window
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    assert "--free-site" not in w._solve()[1]
    w.free_site.setChecked(True)
    assert w._solve()[1][-1] == "--free-site"
    w.clips.setCurrentRow(_row(w, "beta.mov"))
    assert not w.free_site.isChecked()
    w.clips.setCurrentRow(_row(w, "alpha.mp4"))
    assert w.free_site.isChecked()
