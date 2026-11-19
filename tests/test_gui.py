"""The live window follows a work folder's event log (offscreen Qt)."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from skyfind import gui, pipeline  # noqa: E402

from .test_replay import _story  # noqa: E402


def _folder(tmp_path):
    from remove_satellites.core.video import FrameWriter
    clip = tmp_path / "night.mp4"
    with FrameWriter(clip, 640, 360, 24, crf=20) as fw:
        for _ in range(3):
            fw.write(np.full((360, 640, 3), 30, np.uint8))
    wd = pipeline.Workdir(clip)
    wd.write("clip.json", dict(clip=str(clip), cleaned=str(clip), n_frames=3,
                               width=640, height=360, fps=24.0))
    return wd


def _events(story, n_tracks, solved):
    out = [dict(type="start", n_tracks=len(story.tracks), lat0=47.0,
                window_h=12, title="night")]
    for i, t in enumerate(story.tracks[:n_tracks]):
        out.append(dict(type="track", i=i, **t))
    if solved:
        out += [dict(type="solution", **story.solution),
                dict(type="ids", ids=story.ids, objects=story.objects),
                dict(type="done")]
    return "".join(json.dumps(e) + "\n" for e in out)


def _until(w, done, timeout=30.0):
    """Tick the window until `done()` (the replay loads in the background)."""
    import time
    t0 = time.time()
    while not done() and time.time() - t0 < timeout:
        w.tick()
        time.sleep(0.02)
    assert done()


def test_window_follows_a_running_solve(tmp_path):
    wd = _folder(tmp_path)
    story = _story()
    wd.path("events.jsonl").write_text(_events(story, 2, False))
    app, w = gui.window(wd, fps=10, width=800)
    _until(w, lambda: w.story is not None and w.frames)     # loads aside
    for _ in range(25):
        w.tick()
    assert w.story is not None and len(w.story.tracks) == 2
    assert not w.story.done and w.label.pixmap() is not None
    before = len(w.frames)
    # the solve moves on: more tracks, then the answer
    wd.path("events.jsonl").write_text(_events(story, 6, True))
    os.utime(wd.path("events.jsonl"), None)
    _until(w, lambda: w.story.done)
    for _ in range(25):
        w.tick()
    assert w.story.done and w.story.solution is not None
    assert len(w.frames) > before
    w.close()


def test_a_moved_work_folder_finds_its_cleaned_copy(tmp_path):
    wd = _folder(tmp_path)
    (wd.path("cleaned.mp4")).write_bytes(b"")
    wd.write("clip.json", dict(clip="X:/gone/night.mp4", cleaned="cleaned.mp4",
                               n_frames=3, width=640, height=360, fps=24.0))
    assert wd.info["cleaned"] == str(wd.path("cleaned.mp4"))
    wd.write("clip.json", dict(clip="X:/gone/night.mp4",
                               cleaned="X:/gone/night.skyfind/cleaned.mp4",
                               n_frames=3, width=640, height=360, fps=24.0))
    assert wd.info["cleaned"] == str(wd.path("cleaned.mp4"))
