#!/bin/sh
# Build the app. JDK 17, the Android SDK and Gradle 8.7 come from the
# environment (JAVA_HOME, ANDROID_HOME, gradle on the PATH or GRADLE); an
# untracked android/local.sh next to this script, when there is one, sets
# them for this machine.
#   sh android/build.sh                 # assembleDebug
#   sh android/build.sh installDebug    # and put it on the connected phone
cd "$(dirname "$0")" || exit 1
[ -f local.sh ] && . ./local.sh
: "${JAVA_HOME:?set JAVA_HOME to a JDK 17}"
: "${ANDROID_HOME:?set ANDROID_HOME to the Android SDK}"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$ANDROID_HOME}"
# sgp4 installs as pure Python: the app propagates with libsgp4c.so
export PYTHON_SGP4_COMPILE=never
exec "${GRADLE:-gradle}" --console=plain "${@:-assembleDebug}"
