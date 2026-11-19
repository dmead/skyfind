# skyfind for Android

The phone app: pick a night timelapse, solve it on the phone, and find the
results in Google Photos (Pictures/skyfind and Movies/skyfind: the trails
named on the sky, a close-up, the site map and the solve replay).

It runs skyfind's own pipeline (trails, solve, report, replay), the same
Python the desktop runs, inside the app through
[Chaquopy](https://chaquo.com/chaquopy/). Only a few platform switches
differ (see "Differences" below).

## Using it

1. **Record the sky**, or **choose a clip** (any video the phone can
   play: MP4, MOV, WebM; H.264, HEVC, VP9). To record: phone on a tripod,
   pointed at the sky, away from lights.
   - *Focus on the stars* steps the lens through the far end of its
     travel and keeps the position where the stars are smallest (about a
     minute; the phone's autofocus can't see stars, and its "infinity"
     setting is rarely exact). It says so when it finds no stars.
   - Exposure and ISO are held as set (default 4 s, ISO 3200, or the most
     the camera gives). Longer exposures show more stars; they don't make
     satellite trails brighter.
   - *Record*, leave it for ten minutes or more, *Stop*, then *Solve this
     clip*. The clip and its record of where and when (GPS, frame times)
     stay in the app's folder (`files/captures`). With a GPS fix the solve
     skips the search for the site and time and just names the trails;
     without one the recording time still dates a blind solve.
2. **Orbital elements:** either type your Space-Track login (kept
   encrypted on the phone; used only to fetch the night's elements), or
   add element files (`.3le` / `.tle`) with *Add orbital elements*.
3. Optional: the night (UTC) when neither planets, a comet nor the Moon
   can date the clip, and *Free site* when the skyline misleads.
4. **Solve.** It keeps working with the screen off (a notification shows
   the stage). With the app open, the solve story plays live at the top
   once the trails are found: the tracks being fitted, the candidate sites
   closing in on the map, then every trail named (the desktop's `skyfind
   watch`, redrawn every few seconds). When it's done the pictures and the
   replay are in Google Photos, and the app has buttons to open them.

On a mid-range phone a 1080p clip of 800 frames takes about 17 minutes
to find its trails, 6 minutes to solve (5 of them fitting candidates to
the tracks) and under 2 minutes for the pictures and the replay; 4K takes
about four times as long for the trails. A clip solved again reuses its
trails. The phone's answer for that clip fell within the desktop's error
bars.

## Differences from the desktop

- **No other processes** (Android can't start more Pythons): the solve's
  workers are threads (`skyfind/par.py`, `solve.Threads`); remove-
  satellites runs in the app's process.
- **No GPU** (CuPy is desktop only).
- **No cleaned copy**: with no video encoder for it, the trails come from
  the median itself (`SKYFIND_EXACT_RESIDUAL=1`), thresholded at 16 (what
  a copy's encoding made the desktop's 18 into: `scripts/ab_residual.py`);
  single cleaned frames are made as needed (`Workdir.cleaned_frame`).
- **SGP4** is python-sgp4's own C++ (Vallado), built for the phone
  (`native/`, loaded by `skyfind/sgp4lib.py`); python-sgp4 installs as
  pure Python for skyfield.
- **The free-site search** (no skyline to trust) runs the desktop GPU's
  kernels compiled as C++ threads (`native/freesearch.cpp`,
  `skyfind/native.py`): Python threads can't share that loop.
- **Vertical structures** are found with Canny + Hough: the phone's
  OpenCV has no line segment detector.
- **The orbit table** is on a 120 s grid, not 20 s (a sixth of the
  memory; Hermite interpolation keeps LEO positions to metres).
- **Video** is read through Android's decoder (remove-satellites'
  `video.capture` makes up for what it can't do, including OpenCV 4.5.1
  swapping U and V: reds came out blue) and the replay MP4 is made by its
  encoder (`Mp4Writer.kt`).
- The phone's Python stack is Python 3.10, numpy 1.23, scipy 1.8,
  OpenCV 4.5: the newest Chaquopy has wheels for that fit together.

## Building

Needs the Android SDK (`ANDROID_HOME`), JDK 17 (`JAVA_HOME`), Gradle 8.7
(`gradle` on the PATH, or `GRADLE`), the NDK (`NDK`, else the SDK's
26.3), a Python 3.10 for Chaquopy (`uv python install 3.10`; its path in
`CHAQUOPY_PYTHON`), the remove-satellites checkout beside this repo, and
the reference data in `SKYFIND_DATA` (Hipparcos, DE421, Natural Earth,
MPC comets), which goes into the APK. The build scripts source an
untracked `android/local.sh` first: put your machine's paths there.

```sh
sh android/native/build.sh           # libsgp4c.so, libskyfind_native.so
sh android/build.sh assembleDebug     # the APK
sh android/build.sh installDebug      # onto the connected phone
```

The tests on a desktop with the phone's versions and the app's switches
(to try changes before a build), in a container (see the repo's
`docker-compose.yml` and `.env.example`):

```sh
docker compose --profile phone build phone
docker compose run --rm phone                  # all the tests
docker compose run --rm phone -q tests/test_focus.py
```

## Testing on the phone without touching it

The solve service can be started over adb on files pushed to the app's
own folder (the phone may be locked; the service keeps the CPU awake):

```sh
adb push night.mp4 night.3le /sdcard/Android/data/com.skyfind/files/in/
adb shell am start-foreground-service -n com.skyfind/.SolveService \
    --es clip_path /sdcard/Android/data/com.skyfind/files/in/night.mp4 \
    --esa tle_paths /sdcard/Android/data/com.skyfind/files/in/night.3le
adb logcat -s skyfind
```

The work folder is `/sdcard/Android/data/com.skyfind/files/work/<clip>.skyfind/`
(`app.log` is the whole log); the outputs are also saved to the gallery.
`--es dump_frame <clip>` (with `--ei k N`) writes frame N as the pipeline
reads it to `files/frame.png` instead: a check of the phone's decoder.

The emulator (x86_64) runs it too: `sh android/build.sh assembleDebug
-Pemu` adds that ABI (and `sh android/native/build.sh` builds both).
