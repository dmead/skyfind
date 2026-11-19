"""Re-solve the clips with known answers (scratch copies of their work
folders) and compare with the reference results. Space-Track downloads use
the account saved in the workbench (the password goes from the keyring to
the solve's environment, nowhere else).

    python scripts/regress.py [NAME ...]
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

def cases() -> dict:
    """name -> (clip, scratch work folder, solve options, reference), from
    `$SKYFIND_DATA/regress.json` (untracked: the clips are on loan)::

        {"cases": {"NAME": {"clip": "...", "workdir": "...", "options": [],
                            "reference": {"lat": .., "lon": .., "frame0":
                            "HH:MM:SS", "objects": .., "confirmed": ..}}}}
    """
    from skyfind import data
    f = data.data_dir() / "regress.json"
    if not f.exists():
        sys.exit(f"no regression cases: write them to {f} (see this "
                 "script's cases())")
    c = json.loads(f.read_text(encoding="utf-8"))["cases"]
    return {k: (v["clip"], v["workdir"], v.get("options", []), v["reference"])
            for k, v in c.items()}


def main():
    from skyfind import sources
    CASES = cases()
    names = sys.argv[1:] or list(CASES)
    _, user = sources.load()
    env = dict(os.environ, **sources.environment(user, sources.password(user)))
    py = str(Path(sys.executable))
    rows = []
    for name in names:
        clip, wd, opts, ref = CASES[name]
        log = Path(wd) / "solve.log"
        tic = time.time()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call([py, "-u", "-m", "skyfind", "solve", clip,
                                  "--workdir", wd, *opts], stdout=f,
                                 stderr=subprocess.STDOUT, env=env)
        dt = time.time() - tic
        if rc or not (Path(wd) / "solution.json").exists():
            rows.append(f"{name}: FAILED (exit {rc}) after {dt:.0f} s; see {log}")
            print(rows[-1], flush=True)
            continue
        s = json.loads((Path(wd) / "solution.json").read_text())
        conf = sum(o["conf"] == "confirmed" for o in s["objects"])
        km = math.hypot((s["lat"] - ref["lat"]) * 111.2,
                        (s["lon"] - ref["lon"]) * 111.2
                        * math.cos(math.radians(ref["lat"])))
        rows.append(f"{name}: {s['lat']:.4f} {s['lon']:.4f} ({km:.1f} km from "
                    f"reference), frame 0 {s['frame0_utc'][11:19]} (ref "
                    f"{ref['frame0']}), {len(s['objects'])} objects / {conf} "
                    f"confirmed (ref {ref['objects']}/{ref['confirmed']}), "
                    f"{dt:.0f} s")
        print(rows[-1], flush=True)
    print("\n".join(["", "summary:"] + rows))


if __name__ == "__main__":
    main()
