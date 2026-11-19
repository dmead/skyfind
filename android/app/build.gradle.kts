plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

// skyfind's Python (this repo), the remove-satellites engine beside it and
// the reference data (Hipparcos, DE421, map layers) go into the app at
// build time; none of them is copied into this folder's git
val skyfindRoot: File = rootProject.projectDir.parentFile
val removeSatellites = file(
    (findProperty("removeSatellitesDir") as String?)
        ?: File(skyfindRoot.parentFile, "remove-satellites").path)
val skyfindData = file(
    (findProperty("skyfindData") as String?)
        ?: System.getenv("SKYFIND_DATA")
        ?: error("set SKYFIND_DATA (or -PskyfindData) to the reference data"))

android {
    namespace = "com.skyfind"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.skyfind"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
        // phones today are arm64; Chaquopy's numpy, scipy
        // and OpenCV wheels are per ABI, so one ABI keeps the APK small
        ndk {
            abiFilters += listOf("arm64-v8a")
            // the emulator is x86_64: `build.sh assembleDebug -Pemu`
            if (project.hasProperty("emu")) abiFilters += "x86_64"
        }
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // libsgp4c.so (android/native) is loaded by path through ctypes: it
    // must exist as a file in the app's native library folder
    packaging { jniLibs { useLegacyPackaging = true } }
    // the ephemeris and star catalogue are read whole: no gain compressing
    androidResources { noCompress += listOf("bsp", "npz") }
}

chaquopy {
    defaultConfig {
        // the newest Python every package below has an Android wheel for
        // (scipy 1.8.1 and OpenCV 4.5.1 stop at 3.10; scipy 1.8 wants
        // numpy < 1.25)
        version = "3.10"
        buildPython((findProperty("buildPython") as String?)
            ?: System.getenv("CHAQUOPY_PYTHON") ?: "python3.10")
        pip {
            install("numpy==1.23.3")
            install("scipy==1.8.1")
            install("opencv-python==4.5.1.48")
            // pure Python here (PYTHON_SGP4_COMPILE=never, build.sh): the
            // table's propagation is libsgp4c.so (skyfind/sgp4lib.py)
            install("sgp4==2.27")
            install("jplephem")
            install("skyfield")
            install("certifi")
        }
    }
    sourceSets {
        getByName("main") { srcDir("src/main/python-gen") }
    }
}

val syncPython by tasks.registering(Sync::class) {
    description = "skyfind's and remove-satellites' packages for the app"
    into("src/main/python-gen")
    from(File(skyfindRoot, "skyfind")) {
        into("skyfind")
        exclude("**/__pycache__/**", "gui.py", "workbench.py")
    }
    from(File(removeSatellites, "remove_satellites")) {
        into("remove_satellites")
        exclude("**/__pycache__/**", "gui/**")
    }
}

val syncData by tasks.registering(Sync::class) {
    description = "reference data the app copies out on first start"
    into("src/main/assets/data")
    from(skyfindData) {
        include("de421.bsp", "hipparcos_v10.npz", "ne_*.geojson",
                "CometEls.txt")
    }
}

tasks.named("preBuild") { dependsOn(syncPython, syncData) }
tasks.matching { it.name.endsWith("PythonSources") }.configureEach { dependsOn(syncPython) }
tasks.matching { it.name.startsWith("merge") && it.name.endsWith("Assets") }
    .configureEach { dependsOn(syncData) }

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")
}
