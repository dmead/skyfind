#!/bin/sh
# The app's native libraries: libsgp4c.so, Vallado's SGP4 behind sgp4c.cpp's
# C API, and libskyfind_native.so, the free-site search. No fused
# multiply-adds, like the desktop build's arithmetic. arm64 (the phones)
# and x86_64 (the emulator: `android/build.sh assembleDebug -Pemu`).
#   sh android/native/build.sh
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$here/../local.sh" ] && . "$here/../local.sh"
NDK=${NDK:-$ANDROID_HOME/ndk/26.3.11579264}
BIN="$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin"
here=$(cd "$(dirname "$0")" && pwd)
for abi in arm64-v8a x86_64; do
    case $abi in
        arm64-v8a) CXX="$BIN/aarch64-linux-android26-clang++.cmd" ;;
        x86_64) CXX="$BIN/x86_64-linux-android26-clang++.cmd" ;;
    esac
    out="$here/../app/src/main/jniLibs/$abi"
    mkdir -p "$out"
    "$CXX" -O2 -fPIC -shared -ffp-contract=off -static-libstdc++ \
        -o "$out/libsgp4c.so" "$here/sgp4c.cpp" "$here/vallado/SGP4.cpp" \
        2>/dev/null && echo "built $abi/libsgp4c.so"
    # the free-site search's kernels (gpu.py's, in threads): skyfind/native.py
    "$CXX" -O3 -fPIC -shared -std=c++17 -ffp-contract=off -static-libstdc++ \
        -o "$out/libskyfind_native.so" "$here/freesearch.cpp" \
        && echo "built $abi/libskyfind_native.so"
done
