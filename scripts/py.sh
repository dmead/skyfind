# source or prefix: this machine's uv caches, managed Pythons and reference
# data folder, from the untracked scripts/py.local.sh when there is one
# (e.g. UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR and SKYFIND_DATA off a
# nearly-full system drive)
# usage:  . scripts/py.sh && uv sync
_here="$(dirname "${BASH_SOURCE:-$0}")"
[ -f "$_here/py.local.sh" ] && . "$_here/py.local.sh"
unset _here
