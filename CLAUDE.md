# skyfind — where and when a timelapse was shot, and which satellites crossed it

Blind solve from the footage alone: stars give the lens, the camera's
orientation and the frame interval; the skyline gives "up"; planets date the
night; satellite trails matched against orbital elements give the site
(latitude, longitude, height), the absolute clock and the exposure, and
name every trail. The image engine (video IO, sky tracking, foreground mask,
the trail cleaner) is **remove-satellites**, pinned in `pyproject.toml`.

**remove-satellites is a library here, never a user-facing dependency**: no
option, log line, UI or doc asks the user to run it or refers to its files.
skyfind never reads or writes next to the clip (no `_no-satellites.mp4`, no
`<clip>.track.npz`): the sky track, mask and cleaned copy all live in the
work folder (`Workdir.track()`, `cleaned.mp4`). The e2e test checks the
clip's folder holds nothing else.

## Environment

- **Containers** (`Dockerfile`, `docker-compose.yml`): skyfind's Python
  in Linux images, so nothing about this machine is in the commands or
  the docs. `docker compose run --rm skyfind <command>` with the
  reference data at /data, clips under /clips (work folders next to
  them) and scratch work folders under /work; this machine's folders are
  in the untracked `.env` (`.env.example`), the GPU through
  `docker-compose.gpu.yml`, the engine at the commit uv.lock pins,
  fetched from GitHub (`docker-compose.engine.yml`: the checkout beside
  this repo instead, its uncommitted work included). Space-Track's login
  is passed through from the environment, never written. Target `phone`
  is the phone app's stack with its switches: `docker compose run --rm
  phone`. The workbench GUI stays on the host. Worker processes are
  spawned, never forked, on every platform (`skyfind/__init__.py`): a
  forked child can't use the GPU its parent opened.
- **No system Python for the project** (on the host). Everything runs
  through uv (on Windows, when it isn't on PATH, winget's
  `"$LOCALAPPDATA/Microsoft/WinGet/Links/uv.exe"`).
- **Before any uv command**: `. scripts/py.sh`: it sources the untracked
  `scripts/py.local.sh` when there is one (a machine's uv cache, managed
  Pythons and `SKYFIND_DATA`, e.g. off a full system drive). Machine paths, phone
  models and the test clips' names live only in untracked files (`*.local.*`,
  `android/local.sh`, `$SKYFIND_DATA/regress.json`), never in git.
- Tests: `uv run pytest -q` (synthetic only, ~2 min; `-m "not slow"` skips
  the rendered-clip end-to-end run). Date tests skip without DE421 in the
  data dir.
- **Reference data never goes in git**: Hipparcos, DE421 and Natural Earth
  map layers download into `$SKYFIND_DATA` on first use. TLEs come from the
  user (`--tle`) or Space-Track with *their* `SPACETRACK_USER`/`PASSWORD`;
  when missing, `TLEsNeeded` says how — never invent or reuse credentials.
  Space-Track data must not be committed (their terms forbid redistribution):
  tests use made-up elements from `tests/synth.py`.
- **Blind means blind**: no location or date priors from file names or
  guesses. Search time over +-12 h (every longitude) around the planets'
  date; take latitude / sidereal time starts only from the data. The one
  exception is a clip the phone app recorded itself: its GPS fix and frame
  times are measurements, kept beside the clip, and `pipeline.run_known`
  names the trails there without a search (the user's choice, 2026-09-30).

## Design invariants (locked by tests — change test + code together)

- **Camera model** (`camera.py`): one orientation, one lens, and the sky
  turning about a *fitted* axis — the pole of date (precession moves it
  ~0.14 deg from J2000 by 2024: pixels over a night). Exact against
  skyfield's Earth rotation (`test_camera`).
- **The stars fix the observer, the clock moves the satellites**
  (`site.py`, `solve.Problem.dirs`): the site's GCRS position at each
  frame's mid-exposure comes from the camera's zenith (stars + skyline) and
  (lat, dlst, h); the clock offset T only shifts the satellites. Moving the
  observer by T produced a phantom offshore solution once.
- **Trail ends are predicted as the Earth-fixed camera recorded them**: a
  point `off` s from mid-exposure is rotated by `off` s of sky rotation
  before comparing (`Problem.dirs`). Without it the exposure fits ~0.3 s
  short.
- **Absolute finite-difference steps** (`solve.fdjac`, `STEPS`): the float32
  orbit table is noisy below cm / us; default relative steps never move.
- **SGP4 gets UTC Julian dates** (`orbits.sgp4_times`), as skyfield does.
- **One object fits many (site, time) pairs** (once per orbit for LEO): per-
  track fits keep every good fit; only agreement between *distinct* objects
  (`cluster`) identifies the site.
- **Candidate pruning is loose on purpose**: rate within x1.5, direction
  within 40 deg, 3 deg end tolerance over a +-4 deg lat / +-6 deg dlst site
  grid — the skyline is only good to a degree or two.
- **Ends cut short** (frame edge, skyline: flagged by `trails.Extractor`;
  Earth's shadow: found by the joint fit) don't enter the fits as exposure
  boundaries.
- **Plate solve hashes quads in their own tangent plane** over a ladder of
  focal lengths and verifies by growing a rotation + f fit outwards: wide
  lenses stretch off-axis quads too much for pixel-space hashing. A solution
  must beat chance by 6 sigma + 8 matches given the detection density
  (compressed video yields thousands of noise detections; raw counts
  accepted false solves).
- **Only detections that move with the sky** feed the plate solve and camera
  fit (`pipeline._moves_with_sky`: each must reappear in the next sampled
  frame where remove-satellites' track carries it). Floodlit structures and
  foliage the mask missed outshine stars otherwise.
- **Refraction** is a fixed warp in camera coordinates about the up vector
  (the camera is Earth-fixed); density fitted from the stars. It applies
  whenever the skyline gives an up, `--free-site` included (the site search
  ignores the skyline's latitude, not its up). Without it low stars pull the
  frame interval ~0.3% long.
- **Up** comes from the skyline, or from the vanishing point of vertical
  structures (towers, poles, rockets) when the skyline fit is weak
  (`horizon.verticals`).
- **A camera may turn** (a motion-control rig: dolly, jib, pan/tilt head;
  two real clips): `Camera.turn` is each frame's rotation from the
  Earth-fixed pose at k0; refraction and up stay in that pose; a frame's
  own rays go through `frame_to_gcrs` (the free search's sky rays too).
  - Detected after the fixed fit (`pipeline._camera_turns`): the stars
    followed frame by frame (`calibrate.track`) miss the best fixed camera,
    its axis pinned to the pole, by > 1 deg (a free axis soaks a steady pan
    up: a dolly clip 0.6 free, 6.8 pinned; tripods <= 0.45).
  - One lens and a per-sampled-frame orientation (`calibrate.follow`, from
    several starts: the tracker's clean pairs, and wide first matches at a
    few focal lengths; the one putting most catalogue stars on detections
    wins: a wide lens's fit has more than one basin). Then every frame its
    own orientation from its own stars (`calibrate.refine_turn`: the rig
    jitters 0.25 deg between samples).
  - The interval: the stars can't split the Earth's turn from the
    camera's, the land can (`motion.interval`: the far skyline kept still),
    but only to a few percent (the rig's travel; smooth hills pin a
    sideways shift poorly: the synthetic rig's came out 3% short), so the
    solve refits the longest anchors at intervals +-8% about it, in 0.5%,
    0.1% and 0.02% steps, keeping the most agreeing objects
    (`pipeline._interval_search`; `Problem.with_interval` only moves the
    turn about the pole; workers reload a re-saved problem.json). The
    count plateaus (the synthetic rig: 16 objects from 29.54 to 30.27 s),
    so the joint fit's own anchors, refitted at intervals about it (tenths,
    hundredths, fiftieths of a percent; the anchors' middle frame keeps
    its time and site), pick it by their residuals, and the joint fit is
    redone there (`pipeline._refine_interval`; synthetic: 30.029 ->
    29.9993 s, frame 0 16 s -> 0.4 s off).
  - The trails are taken out again with the sky's motion from the camera
    (`motion.sky_track`: remove-satellites' tracker takes a turn for bad
    or dropped frames), `pipeline._reclean`. The cleaned copy keeps one
    land mask over the land's whole sweep (`motion.land_mask`), but the
    segments are found against each frame's own land
    (`motion.FrameLand`, `turning_frames`, `run_trails(turning=)`): of
    the union, frame k gets back as sky what shows stars the sky's motion
    holds (a near tree slides by parallax, so the land's own motion can't
    say where it is; dark ground shows no stars and stays land), less the
    land seen in frames k-3..k+3. The dolly clip: 38% of the frame masked before,
    32-34% at night now. Pieces of a trail the land split are rejoined
    (`trails.join_pieces`: collinear, facing ends cut, the gap on land).
    Clutter is judged in the ground's frame, plus segments lying along
    (not across) the frame's border (`_ground_clutter`).
  - Up from the ridge in frame k0 found as the land's motion is
    (`pipeline._moving_horizon`, a fifth of the columns on a line: ridges
    are ragged); refraction refit with only pose and density free.
  - The exposure turn adds the rig's own (`Problem.exposure_turn`,
    clip-averaged, CPU and GPU alike).
  - Sparse clips (a long interval carries satellites across in one or two
    frames): under `SPARSE_ANCHORS` 3-frame tracks, 2-frame ones anchor too
    (fixed cameras included).
  Tripods keep their exact old arithmetic (`tests/test_moving.py`);
  `tests/synth.py` renders a rig (`pan`, `tilt`: land and skyline move,
  texture fixed to the ground) and `test_a_rig_end_to_end` solves it.
- **A camera may follow the stars** (a tracking mount: the stars stand
  still in the frame). It is the turning
  camera above with the turn known: about the pole, at the sidereal rate.
  - Found after the fixed fit (`pipeline._tracking`): the fit's interval
    (the sky's turn) comes out a fraction of a second or negative, and at
    its word the trails would cross faster than any satellite
    (`SAT_MAX_RATE`, 3 deg/s; a tripod's real-time clip looks the same to
    the stars, but its satellites creep).
  - `Camera.tracking`: the fixed fit re-expressed as an Earth-fixed pose
    at k0 plus each frame's turn; every pixel keeps its place among the
    stars (the mount's own drift stays in the turn); this year's pole (the
    fit's axis is noise when nothing turns).
  - Every frame is a capture (`run_trails`, `STILL_PX`): steps that don't
    move aren't repeats when none moves. remove-satellites' tracker has
    the same three assumptions undone: no rejecting steps by their motion
    when all are still; "movers" fitted only when most move together
    (else it's noise and the odd satellite); and when leaving out the
    camera-fixed features leaves nothing to follow, they were the stars.
  - The interval can't come from the stars: `--interval`, else the
    satellites over every interval their rates allow
    (`TRACKED_RATES` 0.08-1.6 deg/s against the trails' median travel per
    frame; `_interval_search(wide=)`, a percent apart: the count of
    agreeing objects holds over about that), then refined as a rig's. The
    first real clip: 1.4-27 s searched in 11 min, 5 objects at 7.01
    s, refined to 7.031 s (6 s exposures a second apart); 45 objects
    against 14 at the next site, 342 named.
  - No skyline is read (any land sweeps through the frame): the site is
    free. Clutter is fixed to the frame, as a tripod's.
  - `tests/test_tracked.py`; `test_a_tracking_mount_end_to_end` (the
    synthetic sky held still: 20 s found blind in 4.6-92 s, the site and
    clock exact, every segment named).
- **Slow movers are named by position** (`pipeline._slow`, after the
  identification): an object creeping a few pixels a frame (a spent stage
  near the top of a transfer orbit, a satellite a few thousand km up)
  leaves segments under `MIN_LEN`, lights one spot frame after frame
  (clutter, by that test), and its trail is mostly its own blur, longer
  than its travel, so its ends time nothing. Its track is long, though: of
  the tracks of `SLOW_FRAMES`+ frames at `SLOW_RATE` (30-900 arcsec/s
  against the stars) not already named, the object within `SLOW_TOL`
  (0.25 deg, median over the frames' mid-exposures) at the solved site and
  clock names it, unless the next best is within half again. IDs carry
  `how: "position"`; such an object is confirmed only over 10+ frames
  under 0.15 deg. A tracked clip: 21 tracks, 15 objects (Falcon 9,
  H-2A, Thor Agena stages, a Breeze-M tank). The replay ends with a
  chapter for them (`replay.slow_movers`, phase "slow": everything else
  greyed, each picked out in white with its rate), and rings them on the
  footage.
- **Only what a sunlit satellite could be** (`shadow.py`,
  `pipeline._sunlit`, the scan night by night): along each trail's line of
  sight the Sun lights only some heights, and an orbit there crosses the
  sky at most so fast; a trail faster than every sunlit orbit (a segment
  at its length over the whole interval, a track at its drift) is left
  out of the anchors and the identification (`in_shadow` in the solution,
  a `shadow` event; the replay rings them red). Loose on purpose: the
  skyline's site grid, the Sun +-1 day, the umbra, x1.1 circular speed
  + 0.5 km/s, x1.25 to spare. Free sites aren't tested. A 4K clip: two
  approach-path aircraft tracks and 24 segments out, and a chance
  identification weakened to probable (GONETS-M on an aircraft at 6 deg,
  where nothing below ~5000 km was lit: 90 confirmed, was 91); the
  synthetic sky (lit satellites only) keeps every trail.
- **Colour is judged against the stars' white** (`pipeline._white`,
  `_chroma`, white.json from `_setup`): a grade tints satellites and stars
  alike, so a trail's channel spread is taken after dividing out the
  stars' tint, not against grey (`trails.Extractor`'s `chroma`). White at
  the stars' scale, which is a trail's: 4:2:0 video keeps colour at half
  resolution, so points and thin lines lose theirs alike (a 4K clip's
  stars come out exactly grey, its trails near-neutral: unchanged). A
  warm-graded clip: 31 more segments, 23 -> 30 objects, and the site from
  7 km off the truth to 1; another +2 objects.
- **Frames are not always captures**: a clip re-timed to another frame rate
  drops frames; remove-satellites' `Track.clock` gives each frame's capture
  index and `Camera.s(k)` / `Problem.tk(k)` use it for all sky angles and
  times.
- **Trail ends at half maximum** along the trail (`trails._half_max_ends`),
  not where a threshold runs out (bright trails measured long).
- **Error bars** are the larger of formal and leave-one-anchor-out; height is
  reported only when the trails constrain it (else 1 +- 1 km, folded in).
- **Confidence tiers + null rate**: IDs are confirmed (several frames, or
  < 0.12 deg and centred < 3 s) or probable; the null rate (clock +-5/10 min)
  says how many chance matches to expect. Starlink trains alias along-track.
- **Parallel = serial**: the solver's worker pool splits work into small
  jobs over one shared orbit table; results are identical to the serial
  path (`test_parallel_matches_serial`, `..._propagation_matches_serial`:
  the TEME->GCRS rotation is written term by term so any split of the table
  into blocks gives the same bits).
- **One pool, started first; the table in shared memory** (`pipeline._build`,
  `orbits._Table`): the `solve.Workers` start before dating and elements
  (they import meanwhile), fill the orbit table in (time x object) blocks
  in shared memory (never on disk; `orbits.NoRoom` falls back to .npy files
  on a disk with room), and load the problem on their first job
  (`solve._problem`, `Workers.attach`). Each finished block is copied to
  the GPU at once (`gpu.Upload`). Propagation is CPU-bound near the
  14900K's limit (efficiency cores, hyperthreads): SGP4 on a GeForce's
  FP64 wouldn't beat it.
- **Element files are indexed once** (`tles.Catalog`): columns (object,
  epoch, rank, where the lines are) kept in `$SKYFIND_DATA/element-index`,
  checked against each file's size and mtime; `nearest_epoch` picks in
  columns and makes `Element`s only for the sets picked (the same choice as
  the list path, ties to the first listed).
- **Many Earth-fixed conversions at once** (`data.times`): IAU 2000A
  nutation interpolated from 257 samples when the times span < 3 days
  (~1e-6 arcsec; the series' shortest periods are days).
- **Batched, not one at a time** (`batchfit.py`): a track's candidates are
  fitted together by a numpy Levenberg-Marquardt with `solve.fit`'s loss,
  scales, bounds and steps (each candidate its own damping, so batch
  composition never changes a result); `joint` tries a track's objects as
  one batch. `SKYFIND_SCIPY_FITS=1` brings back the per-candidate SciPy fits
  for comparison (`scripts/ab_batchfit.py`, `ab_joint.py`, `ab_identify.py`).
- **GPU** (`gpu.py`, CuPy): the orbit table goes to the card once per solve
  (`solve._gpu_orbits`, passed as `dev`) and serves every stage. GeForce
  FP64 runs at 1/64 of FP32, so sweeps are single precision — where the
  CPU's is, or with margins and a double-precision recheck — and fine work
  is double. Each mirrors a CPU function test for test, in the same order:
  - free-site search (`free_passes_all`): every track at once — a thread
    per (row, object) reads the position once and tests every track (line
    of sight's distance via |p x d|, which doesn't cancel; the sky test,
    dearest, last), then one double-precision thread per survivor for the
    recheck and one for the refinement; passes merged in rounds
    (`_merge`);
  - skyline search (`near_objects` sweep, float32 like the CPU's;
    `check_passes`, one thread per (object, clock, grid site) cell, the
    survivors' per-segment results in a second pass over them alone);
  - fits (`fit_all`: `batchfit.fit` with a fused residual kernel;
    candidates grouped shortest track first, by padded ends);
  - the joint fit's join step (`join_tracks`: every candidate site's
    other tracks' objects at once — end orders with `_PASS`, median end
    errors with `FitBatch`);
  - identification (`identify`: every segment at every solution at once —
    tie-break sites, the chance check's wrong clocks).
  Same results as the CPU up to rare ties between sample times
  (`test_the_gpu_*`, `scripts/ab_gpu.py`, `ab_gpufit.py`). Off with
  `SKYFIND_GPU=0` (which the trails stage passes on to the cleaner: its
  GPU path writes the CPU's file bit for bit), or when the table won't fit
  in GPU memory. In kernels
  prefer `rsqrt` and reciprocals to FP64 division: it's slow on GeForce.
- **Ties are broken by identifying everything** (`pipeline._tiebreak`):
  only a top cluster with `TIE_MARGIN` (1.5x) the next one's independent
  objects goes straight to its joint fit (`clear_winner`: a one-object
  lead can be the wrong end of a Starlink train's line). Otherwise the
  clusters tie: the top ones and the top ones among objects not flown
  in trains (`TRAINS`) are each joint-fitted and every trail identified; a
  site must confirm `TIE_MARGIN` (1.5x) the next one's objects. Every
  site's joint fit runs in three steps, all sites at once in each
  (`joint_start` side by side on the pool; `join_tracks` on the GPU;
  `joint_finish` on the pool), without error bars (`joint_fit`); then all
  sites' identification in one go (`identify_all`); only the winner gets
  its error bars (`joint_errors`, the refits side by side).
- **Clustering in arrays** (`solve.cluster`): neighbour pairs through a
  sorted cell index, best member per track by one lexsort, medians by group
  arithmetic, distinct orbits once per member set; the same clusters, in
  the same order, with members in the same order, as the per-fit loop.
- **Elements over `tles.MAX_AGE_DAYS` (30) from the night are dropped.**
- **Dates must be real and single** (`pipeline._past`, `_unambiguous`,
  `date.DEEP_SKY`): planets and comets are searched 1990 to this year, and
  dates after today are dropped (a planet's next return fits as well as the
  last). Bright Messier/NGC nebulae and clusters are not planets. When a
  date more than 2 days away matches as many objects as the best, the clip
  is undated: one glow lines up with a planet somewhere in the decades by
  chance. The pole of date is not used to pick a year: the pole moves only
  ~0.0056 deg/yr, less than a fit can pin down.
- **No trails, no dating**: the anchor tracks are found right after the
  camera fit, and a clip without any stops there. Dating and elements
  would be wasted on it.
- **Regression**: `scripts/regress.py` re-solves the real clips with
  known answers in scratch work folders against their reference answers;
  the clips, folders, options and answers are in the untracked
  `$SKYFIND_DATA/regress.json` (the clips are on loan).
- **The Android app runs this same code** (`android/`, Chaquopy): the
  packages are copied in at build time, and platform differences sit
  behind small switches, never forks of the pipeline:
  - `par` (SKYFIND_PROCESSES=0): threads for processes; `solve.Threads`
    has `Workers`' interface; the table is built in-process.
  - `sgp4lib`: python-sgp4's C++ where installed, else `libsgp4c.so`
    (android/native: the same Vallado code and the wrapper's arithmetic,
    through ctypes).
  - remove-satellites' `video.capture` (and `read_frame`): Android's
    decoder can't seek, reports no count, rate or true size, and pads HEVC;
    OpenCV 4.5.1 converts its NV12 as NV21 (U and V swapped: reds come out
    blue), undone by one affine map (`video._UV_FIX`).
  - `Workdir.cleaned_frame`: no cleaned copy there (no encoder; trails come
    from the median itself, SKYFIND_EXACT_RESIDUAL, threshold 16).
  - SKYFIND_GRID_DT=120 (memory); the replay's MP4 through the phone's
    encoder (`Mp4Writer.kt`).
  - `native` (SKYFIND_NATIVE): the free-site search's GPU kernels as C++
    threads (android/native/freesearch.cpp), driven by `freesearch.run`
    like the GPU's (`gpu._GpuFree`); Python threads can't share that
    search (GIL: a 1080p clip's took 37 min on the desktop, 1 s native).
  - `horizon.line_segments`: the phone's OpenCV 4.5.1 has no LSD (licence);
    Canny + probabilistic Hough instead.

  - **The app records its own clips** (`CaptureActivity.kt`, Camera2):
    exposure, gain, white balance and lens held; noise reduction, edge
    enhancement and stabilisation off (they eat faint trails); each YUV
    frame encoded as it comes (`Mp4Writer`, up to 4K, played at 30 fps);
    beside the clip a JSON of where and when (GPS fix; frame 0's
    mid-exposure from the sensor's timestamps on the GPS's clock, else the
    phone's; the exposure; every frame's time).
    - Focus is by star size (`skyfind/focus.py`, `tests/test_focus.py`):
      autofocus has nothing to hold on a dark sky and a phone lens's
      infinity is uncalibrated. `Sweep`: a coarse pass over the far end of
      the lens's travel fits the V-curve of the stars' half-flux radius, a
      fine pass takes the peak of their sharpness (the brightest pixel's
      share: a phone's stars are a pixel or two wide and their HFR goes
      flat at focus), every stop approached from the near side (voice
      coils have hysteresis). A star is measured over its own lit pixels
      only (the whole window's noise pulls the HFR out). Kotlin only moves
      the lens and hands over luma; `focus.drifted` warns while recording.
    - `pipeline.run_known` (`skyfind solve --recorded`): up is the site's
      zenith through the fitted camera (`_site_up`), not the skyline's;
      every segment identified at the recorded site and time; the clock
      then set by the median `dt` of `CLOCK_IDS` tight identifications
      (the site turned with it: dlst = T at the sidereal rate), since
      `identify` finds a pass up to 8 s off. No anchors needed.
      `test_a_recorded_site_names_the_trails`: a 1.3 s clock error
      recovered to 0.05 s.
    - **Samsung's exposure range**: Android's own key stops short (the
      test phone: 0.35 s); the sensor's real range is in the vendor key
      `samsung.android.sensor.info.exposureTimeRange` (10 s), and a plain
      SENSOR_EXPOSURE_TIME up to it is honoured in a repeating request
      (frames 4.005 and 8.009 s apart, 4K YUV). The screen offers
      it; `results` logs what the sensor gave and the JSON records that,
      not what was asked. A change waits behind ~5 frames under way (46 s
      going from 8 s to 4 s; `abortCaptures` doesn't hurry it), so Record
      is refused until the sensor has answered.
    - Tested on the emulator's camera (UI, lens stepping through Python,
      recording, the JSON) and on the test phone indoors (exposures,
      recording);
      not yet under a real sky, and the focus run not yet on real stars.

  The phone's stack is Python 3.10, numpy 1.23, scipy 1.8, OpenCV 4.5
  (Chaquopy's newest wheels that fit together): keep the code working on
  them. The `phone` container tests them on the desktop (see
  android/README.md). Never type the test phone's PIN; `SolveService` can
  be started over adb on files pushed to the app's folder.

## Layout

```
skyfind/camera.py      Camera: project / unproject, pole-of-date axis
skyfind/catalog.py     Hipparcos bright stars (auto-download, proper motion)
skyfind/tles.py        TLE parse, indexed columns / nearest epoch / Space-Track
skyfind/platesolve.py  blind quad-hash plate solve of one frame
skyfind/calibrate.py   whole-clip camera fit, usable-frame range
skyfind/horizon.py     skyline -> up vector -> rough latitude
skyfind/date.py        planets (+ Sun, precession) -> nominal time of frame 0
skyfind/trails.py      segments from original - cleaned; linking
skyfind/orbits.py      SGP4 table in GCRS, Hermite interpolation, shared memory
skyfind/site.py        observer's inertial position from camera + up
skyfind/solve.py       per-track fits, cluster, joint fit, identify, null rate
skyfind/batchfit.py    all of a track's candidates fitted at once (numpy LM)
skyfind/gpu.py         pass search, fits, identification on the GPU (CuPy)
skyfind/pipeline.py    per-clip work folder; trails / solve stages; events.jsonl
skyfind/report.py      trail graphic, close-up, JSON, site map
skyfind/replay.py      solve replay on the footage (or --still), object list, world map; GIF/MP4
skyfind/gui.py         live window following events.jsonl (PySide6, `gui` extra)
skyfind/cli.py         skyfind trails | solve | report | replay | gui | run
skyfind/par.py         processes, or threads where there can't be any (Android)
skyfind/sgp4lib.py     SGP4 for the table: python-sgp4's C++, or the app's libsgp4c
skyfind/freesearch.py  free-site search driver for kernel backends (GPU, native)
skyfind/native.py      the free-site kernels in C++ threads (the phone's backend)
skyfind/motion.py      a turning camera: skyline profiles, the interval, sky track, land per frame
skyfind/shadow.py      Earth's shadow: which trails a sunlit satellite could make
skyfind/focus.py       focus by star size: HFR, the V-curve sweep (the phone's capture)
tests/synth.py         the synthetic sky: made-up site/clock/lens/constellation
android/               the phone app (Chaquopy + Kotlin): see android/README.md
```

## Example data

README examples use public-domain or openly licensed footage only,
credited, and none of it in git. The other test clips are on loan: run
them for regressions, never publish them or anything derived, and never
name them, their sites or their answers in anything tracked.

