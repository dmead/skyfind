"""The CLI's commands calling one another."""

import inspect

from skyfind import cli


def test_run_gives_every_option(monkeypatch, tmp_path):
    """`run` calls `trails`, `solve` and `report` as functions: an option
    it leaves out arrives as typer's OptionInfo, not its default (a truthy
    one: it once turned on --free-site)."""
    real = {f: inspect.signature(getattr(cli, f)).parameters
            for f in ("trails", "solve", "report")}
    got = {}
    for f in real:
        monkeypatch.setattr(cli, f, lambda *a, _f=f, **k: got.update(
            {_f: (a, k)}))
    cli.run(tmp_path / "night.mp4", tle=None, around=None, window=12.0,
            workers=None, workdir=None)
    for f, params in real.items():
        a, k = got[f]
        assert set(list(params)[:len(a)]) | set(k) == set(params), f
