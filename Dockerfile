# syntax=docker/dockerfile:1.7
# skyfind's Python in a container: the CLI (trails, solve, report, replay,
# scan) and its tests, with the GPU path when the host passes a GPU through.
# Built by docker-compose.yml. The image engine is remove-satellites at the
# commit uv.lock pins, fetched from GitHub; docker-compose.engine.yml puts
# a local checkout in its place (the `remove_satellites` build context).
#
#   target skyfind: the desktop's stack, the versions in uv.lock
#   target phone:   the phone app's stack (Python 3.10, numpy 1.23, scipy
#                   1.8, OpenCV 4.5) with the app's switches, for its tests

# -------------------------------------------------------------- engine
FROM python:3.12-slim-bookworm AS engine-pinned
COPY uv.lock /tmp/
RUN python - <<'EOF'
import io, re, tarfile, urllib.request
lock = open("/tmp/uv.lock").read()
url, rev = re.search(r'git = "(https://github\.com/[^"?]+/remove-satellites)'
                     r'\?rev=([0-9a-f]{40})', lock).groups()
with urllib.request.urlopen(f"{url}/archive/{rev}.tar.gz") as f:
    tar = tarfile.open(fileobj=io.BytesIO(f.read()))
for m in tar.getmembers():          # the archive's top folder dropped
    m.name = m.name.partition("/")[2]
    if m.name:
        tar.extract(m, "/rs", filter="data")
print("remove-satellites", rev)
EOF

# a checkout's files: pyproject.toml, README.md, remove_satellites/
FROM scratch AS remove_satellites
COPY --from=engine-pinned /rs/ /

# ------------------------------------------------------------- desktop
FROM python:3.12-slim-bookworm AS skyfind
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN uv venv /opt/venv
WORKDIR /src

# the locked third-party versions first (a layer that changes rarely)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-hashes --no-emit-project --group dev \
        --no-emit-package remove-satellites -o /tmp/req.txt \
    && uv pip install -r /tmp/req.txt
# the GPU path: CuPy for CUDA 13, and of the toolkit only what skyfind
# uses (the runtime's headers and NVRTC to compile its kernels, cuBLAS for
# its matrix products); the driver comes from the host. GPU=0 builds
# without it.
ARG GPU=1
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$GPU" = "1" ]; then \
        uv pip install cupy-cuda13x==14.2.0 nvidia-cuda-runtime==13.4.92 \
            nvidia-cuda-nvrtc==13.4.92 nvidia-cublas==13.8.0.4; fi

# the image engine, then skyfind itself
COPY --from=remove_satellites pyproject.toml README.md /rs/
COPY --from=remove_satellites remove_satellites /rs/remove_satellites
RUN uv pip install --no-deps /rs
COPY skyfind ./skyfind
COPY tests ./tests
COPY scripts ./scripts
RUN uv pip install --no-deps .

# the reference data (catalogue, ephemeris, elements, maps): a volume
ENV SKYFIND_DATA=/data
WORKDIR /clips
ENTRYPOINT ["skyfind"]

# --------------------------------------------------------------- phone
FROM python:3.10-slim-bookworm AS phone
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
# The phone's OpenCV is 4.5.1; 4.5.4 is the oldest with wheels for 3.10
# (4.5.5's Linux wheel crashes in connectedComponentsWithStats on a 4K
# frame).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv \
    && uv pip install numpy==1.23.3 scipy==1.8.1 \
        opencv-python-headless==4.5.4.60 skyfield jplephem sgp4 pytest typer
COPY --from=remove_satellites remove_satellites /src/remove_satellites
COPY skyfind /src/skyfind
COPY tests /src/tests
COPY pyproject.toml /src/
WORKDIR /src
# the app's switches (android/app/src/main/python/skyfind_android.py)
ENV PYTHONPATH=/src SKYFIND_DATA=/data SKYFIND_PROCESSES=0 \
    REMOVE_SATELLITES_WORKERS=1 SKYFIND_GPU=0 REMOVE_SATELLITES_GPU=0 \
    SKYFIND_EXACT_RESIDUAL=1 SKYFIND_TRAIL_THRESH=16 SKYFIND_GRID_DT=120
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q", "tests"]
