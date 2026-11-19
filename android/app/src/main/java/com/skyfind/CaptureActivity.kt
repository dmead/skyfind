package com.skyfind

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.graphics.Typeface
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.PyObject
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import kotlin.math.abs

/**
 * Record a night clip for the solve, on a tripod: the camera held at one
 * exposure, gain and white balance (no noise reduction or sharpening: they
 * eat faint trails), focused on the stars themselves, every frame encoded
 * as it comes, and where and when it was shot (GPS, the frames' own times)
 * kept beside the clip for the solve.
 *
 * Focus: autofocus has nothing to hold on a dark sky, and a phone lens's
 * "infinity" is a number, not a stop. So the lens is stepped through the
 * far end of its travel and the stars measured at each stop
 * (skyfind/focus.py, the same code the desktop tests): the position where
 * they're smallest is kept, and watched while recording.
 */
class CaptureActivity : AppCompatActivity() {
    private enum class Mode { IDLE, FOCUS, RECORD }

    private lateinit var preview: TextureView
    private lateinit var status: TextView
    private lateinit var gpsLabel: TextView
    private lateinit var expLabel: TextView
    private lateinit var isoLabel: TextView
    private lateinit var focusBtn: Button
    private lateinit var recordBtn: Button
    private lateinit var useBtn: Button
    private lateinit var logView: TextView

    private val camThread = HandlerThread("skyfind-camera").apply { start() }
    private val cam = Handler(camThread.looper)
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var previewSurface: Surface? = null
    private lateinit var chars: CameraCharacteristics
    private var cameraId = ""
    private var manual = false            // exposure, gain and lens are ours
    private var size = Size(1920, 1080)
    private var realtime = false          // frame times on the boot clock

    private var exposures = listOf(1_000_000_000L)
    private var standardMaxNs = 0L        // the longest exposure Android's own range allows
    @Volatile private var askedNs = 0L    // the exposure last sent, until the sensor answers
    @Volatile private var gotNs = 0L      // the exposure the sensor reports for its frames
    private var isos = listOf(800)
    private var exposureNs = 1_000_000_000L
    private var iso = 800
    @Volatile private var lens = 0f       // dioptres; 0: the maker's infinity
    private var focusedHfr: Double? = null

    @Volatile private var mode = Mode.IDLE
    // the focus run's eyes: frames' luma as they come, and when the lens
    // was last reported where it was sent, at rest
    private val lumas = LinkedBlockingQueue<Pair<Long, ByteArray>>(4)
    @Volatile private var wantLuma = false
    @Volatile private var settledAt = 0L
    @Volatile private var lensTarget = -1f

    private var writer: Mp4Writer? = null
    private var clip: File? = null
    private val times = ArrayList<Long>()        // frames' sensor times, ns
    private var clockOffsetNs = 0L                // UTC ns - sensor ns
    private var clockSource = "system"
    private var fix: Location? = null
    private var i420 = ByteArray(0)
    private var checking = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val pad = (12 * resources.displayMetrics.density).toInt()
        val red = Color.rgb(200, 40, 30)          // kind to dark-adapted eyes
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.BLACK)
            setPadding(pad, pad, pad, pad)
        }
        fun label(text: String, size: Float = 14f) = TextView(this).apply {
            this.text = text; textSize = size; setTextColor(red)
        }
        fun button(text: String, on: () -> Unit) = Button(this).apply {
            this.text = text; setTextColor(red)
            setBackgroundColor(Color.rgb(40, 8, 6))
            setOnClickListener { on() }
        }
        fun row(vararg views: View) = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            for (v in views) addView(v, LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, if (v is TextView && v !is Button) 2f else 1f))
        }

        // sized before the camera is known (a view with no size never gets
        // its surface); open() gives it the sensor's shape
        val pw = resources.displayMetrics.widthPixels / 2
        preview = TextureView(this).apply {
            layoutParams = FrameLayout.LayoutParams(pw, pw * 4 / 3, Gravity.CENTER_HORIZONTAL)
        }
        col.addView(FrameLayout(this).apply { addView(preview) })
        status = label("opening the camera...", 16f).also { col.addView(it) }
        gpsLabel = label("GPS: waiting for a fix").also { col.addView(it) }
        expLabel = label("")
        isoLabel = label("")
        col.addView(row(button("-") { stepExposure(-1) }, expLabel, button("+") { stepExposure(1) }))
        col.addView(row(button("-") { stepIso(-1) }, isoLabel, button("+") { stepIso(1) }))
        focusBtn = button("Focus on the stars") { startFocus() }
        recordBtn = button("Record") { if (mode == Mode.RECORD) stopRecording() else startRecording() }
        useBtn = button("Solve this clip") { useClip() }.apply { visibility = View.GONE }
        col.addView(focusBtn); col.addView(recordBtn); col.addView(useBtn)
        logView = TextView(this).apply {
            typeface = Typeface.MONOSPACE; textSize = 11f; setTextColor(red)
        }
        col.addView(logView)
        setContentView(ScrollView(this).apply {
            setBackgroundColor(Color.BLACK); addView(col) })

        preview.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(s: SurfaceTexture, w: Int, h: Int) { open() }
            override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(s: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(s: SurfaceTexture) {}
        }
        // Python started now (numpy and OpenCV take seconds to load), not
        // at the first focus stop
        thread(name = "skyfind-warm") {
            try { py().callAttr("focus_ready") } catch (e: Throwable) { Log.w(TAG, "python", e) }
        }
    }

    private fun py(): PyObject = SolveService.python(this).getModule("skyfind_android")

    private fun say(s: String) {
        Log.i(TAG, s)
        runOnUiThread {
            logView.text = (logView.text.toString() + s + "\n").lines().takeLast(60).joinToString("\n")
        }
    }

    private fun show(s: String) = runOnUiThread { status.text = s }

    // ------------------------------------------------------------ permissions
    private fun granted(p: String) = checkSelfPermission(p) == PackageManager.PERMISSION_GRANTED

    override fun onResume() {
        super.onResume()
        val need = listOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION)
            .filter { !granted(it) }
        if (need.isNotEmpty()) requestPermissions(need.toTypedArray(), 2)
        else { locate(); if (preview.isAvailable) open() }
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (granted(Manifest.permission.CAMERA)) { if (preview.isAvailable) open() }
        else show("The camera permission is needed to record.")
        locate()
    }

    override fun onPause() {
        if (mode == Mode.RECORD) stopRecording()
        mode = Mode.IDLE
        (getSystemService(Context.LOCATION_SERVICE) as LocationManager).removeUpdates(onFix)
        cam.post { close() }
        super.onPause()
    }

    override fun onDestroy() { camThread.quitSafely(); super.onDestroy() }

    // ------------------------------------------------------------------- GPS
    private val onFix = object : LocationListener {
        override fun onLocationChanged(l: Location) {
            fix = l
            gpsLabel.text = String.format(Locale.US, "GPS: %.5f %.5f, %.0f m (+-%.0f m)",
                l.latitude, l.longitude, l.altitude, l.accuracy)
        }
        @Deprecated("") override fun onStatusChanged(p: String?, s: Int, e: Bundle?) {}
        override fun onProviderEnabled(p: String) {}
        override fun onProviderDisabled(p: String) {}
    }

    @SuppressLint("MissingPermission")
    private fun locate() {
        if (!granted(Manifest.permission.ACCESS_FINE_LOCATION)) {
            gpsLabel.text = "GPS: no permission (the clip will be solved blind)"; return
        }
        val lm = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        if (!lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            gpsLabel.text = "GPS: location is switched off (the clip will be solved blind)"; return
        }
        lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, onFix)
    }

    // ---------------------------------------------------------------- camera
    @SuppressLint("MissingPermission")
    private fun open() {
        if (device != null || !granted(Manifest.permission.CAMERA)) return
        val mgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        // the back camera with manual control (the main one comes first)
        val backs = mgr.cameraIdList.filter {
            mgr.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) ==
                CameraMetadata.LENS_FACING_BACK }
        fun caps(id: String) = mgr.getCameraCharacteristics(id)
            .get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
        cameraId = backs.firstOrNull {
            CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR in caps(it) }
            ?: backs.firstOrNull() ?: run { show("No back camera."); return }
        chars = mgr.getCameraCharacteristics(cameraId)
        manual = CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR in caps(cameraId)
        realtime = chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
            CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
        describe()
        size = pickSize()
        // the preview as tall as the sensor's shape makes it (portrait: the
        // sensor's long side runs up the screen)
        val pw = resources.displayMetrics.widthPixels / 2
        preview.layoutParams = FrameLayout.LayoutParams(pw, pw * size.width / size.height,
            Gravity.CENTER_HORIZONTAL)
        mgr.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(d: CameraDevice) { device = d; configure(size) }
            override fun onDisconnected(d: CameraDevice) { d.close(); device = null }
            override fun onError(d: CameraDevice, e: Int) {
                d.close(); device = null; show("The camera failed to open (error $e).")
            }
        }, cam)
    }

    /** What this camera lets the app hold, said once (and logged: the first
     *  thing to read when a phone records badly). */
    private fun describe() {
        var er = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        standardMaxNs = er?.upper ?: 0L
        // Samsung states a short range in Android's own key (0.35 s on one
        // test phone) and the sensor's real one (10 s) in its own: its camera app
        // reads that one. Offered here too; what the sensor then does is
        // read back from every frame (`results`)
        val vendor = try {
            chars.get(CameraCharacteristics.Key(
                "samsung.android.sensor.info.exposureTimeRange", LongArray::class.java))
        } catch (e: IllegalArgumentException) { null }
        if (er != null && vendor != null && vendor.size == 2 && vendor[1] > er.upper) {
            say(String.format(Locale.US, "the maker's own exposure range goes to %.1f s " +
                "(Android's stops at %.2f s): longer ones offered, and checked against " +
                "what the sensor reports", vendor[1] / 1e9, er.upper / 1e9))
            er = android.util.Range(er.lower, vendor[1])
        }
        val sr = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val level = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
        val minFocus = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        val ap = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.firstOrNull()
        val fl = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
        say("camera $cameraId: level $level, manual ${if (manual) "yes" else "NO"}, " +
            "f/$ap ${fl} mm, exposure ${er?.lower?.div(1e9)}..${er?.upper?.div(1e9)} s, " +
            "ISO ${sr?.lower}..${sr?.upper}, lens 0..$minFocus dioptres " +
            "(calibration ${chars.get(CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION)}), " +
            "frame times ${if (realtime) "on the boot clock" else "on their own clock"}")
        if (manual && er != null && sr != null) {
            exposures = (listOf(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0).map { (it * 1e9).toLong() }
                .filter { it >= er.lower && it <= er.upper } + er.upper).distinct().sorted()
            isos = (listOf(400, 800, 1600, 3200, 6400, 12800).filter { it >= sr.lower && it <= sr.upper }
                + sr.upper).distinct().sorted()
            // stars drift a pixel in about 4 s through a phone's main lens
            // (the maker's longer range included: the test phone gives it, and what
            // the sensor did is read back and recorded either way)
            exposureNs = exposures.lastOrNull { it <= 4_000_000_000L } ?: exposures.first()
            iso = isos.lastOrNull { it <= 3200 } ?: isos.first()
        } else {
            say("this camera gives no manual control: exposure and focus stay automatic, " +
                "and faint trails will be lost")
            focusBtn.isEnabled = false
        }
        runOnUiThread { labels() }
    }

    private fun labels() {
        expLabel.text = if (manual) String.format(Locale.US, "  exposure %.2f s", exposureNs / 1e9)
            else "  exposure: automatic"
        isoLabel.text = if (manual) "  ISO $iso" else "  ISO: automatic"
    }

    /** The largest frame the camera gives as YUV that the phone's H.264
     *  encoder takes (up to 4K). */
    private fun pickSize(): Size {
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        val codecs = MediaCodecList(MediaCodecList.REGULAR_CODECS)
        fun encodes(s: Size) = codecs.findEncoderForFormat(MediaFormat.createVideoFormat(
            MediaFormat.MIMETYPE_VIDEO_AVC, s.width / 16 * 16, s.height / 16 * 16)) != null
        return map.getOutputSizes(ImageFormat.YUV_420_888)
            .filter { it.width <= 3840 && it.height <= 2160 && encodes(it) }
            .maxByOrNull { it.width.toLong() * it.height } ?: Size(1920, 1080)
    }

    private fun configure(s: Size) {
        val d = device ?: return
        size = s
        reader?.close()
        reader = ImageReader.newInstance(s.width, s.height, ImageFormat.YUV_420_888, 3).apply {
            setOnImageAvailableListener({ r -> r.acquireNextImage()?.use { frame(it) } }, cam)
        }
        val tex = preview.surfaceTexture ?: return
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        val shape = s.width.toDouble() / s.height
        val ps = map.getOutputSizes(SurfaceTexture::class.java)
            .filter { it.width <= 1280 && abs(it.width.toDouble() / it.height - shape) < 0.02 }
            .maxByOrNull { it.width } ?: Size(640, 480)
        tex.setDefaultBufferSize(ps.width, ps.height)
        val shown = Surface(tex)
        previewSurface = shown
        @Suppress("DEPRECATION")
        d.createCaptureSession(listOf(shown, reader!!.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(c: CameraCaptureSession) {
                    session = c; say("recording size ${s.width}x${s.height}"); apply()
                    show("Point at the sky, then Focus on the stars.")
                }
                override fun onConfigureFailed(c: CameraCaptureSession) {
                    // a camera that can't give this size beside the preview
                    if (s.width > 1920) configure(Size(1920, 1080))
                    else show("The camera wouldn't start at ${s.width}x${s.height}.")
                }
            }, cam)
    }

    /** The camera's settings sent again (exposure, gain, lens). */
    private fun apply() {
        val d = device ?: return
        val c = session ?: return
        try {
            val b = d.createCaptureRequest(if (manual) CameraDevice.TEMPLATE_MANUAL
                                           else CameraDevice.TEMPLATE_RECORD)
            b.addTarget(previewSurface!!); b.addTarget(reader!!.surface)
            if (manual) {
                b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureNs)
                b.set(CaptureRequest.SENSOR_FRAME_DURATION, exposureNs)
                b.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
                b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                b.set(CaptureRequest.LENS_FOCUS_DISTANCE, lens)
                b.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT)
            }
            // what a phone does to make photos pretty wipes out faint trails
            b.set(CaptureRequest.NOISE_REDUCTION_MODE, CameraMetadata.NOISE_REDUCTION_MODE_OFF)
            b.set(CaptureRequest.EDGE_MODE, CameraMetadata.EDGE_MODE_OFF)
            b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)
            b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            askedNs = if (manual) exposureNs else 0L
            c.setRepeatingRequest(b.build(), results, cam)
        } catch (e: Exception) {
            Log.e(TAG, "request", e); show("The camera refused the settings: ${e.message}")
        }
    }

    private val results = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(s: CameraCaptureSession, r: CaptureRequest, t: TotalCaptureResult) {
            // what the sensor did with the exposure asked for, said once per
            // change (a camera may quietly shorten it)
            val asked = r.get(CaptureRequest.SENSOR_EXPOSURE_TIME)
            val got = t.get(CaptureResult.SENSOR_EXPOSURE_TIME)
            if (got != null) gotNs = got
            if (asked != null && got != null && asked == askedNs) {
                askedNs = 0L
                val dur = t.get(CaptureResult.SENSOR_FRAME_DURATION) ?: 0L
                say(String.format(Locale.US, "exposure asked %.2f s: the sensor gave %.2f s " +
                    "(a frame every %.2f s)%s", asked / 1e9, got / 1e9, dur / 1e9,
                    if (abs(got - asked) > asked / 20) " - NOT what was asked" else ""))
            }
            val at = t.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: return
            val still = t.get(CaptureResult.LENS_STATE) != CameraMetadata.LENS_STATE_MOVING
            if (settledAt == 0L && lensTarget >= 0 && still && abs(at - lensTarget) < 0.02f)
                settledAt = t.get(CaptureResult.SENSOR_TIMESTAMP) ?: 0L
        }
    }

    private fun close() {
        try { session?.close() } catch (_: Exception) {}
        session = null
        device?.close(); device = null
        reader?.close(); reader = null
    }

    private fun stepExposure(by: Int) {
        if (mode != Mode.IDLE || !manual) return
        val i = (exposures.indexOf(exposureNs) + by).coerceIn(0, exposures.size - 1)
        exposureNs = exposures[i]; labels()
        cam.post { try { session?.abortCaptures() } catch (_: Exception) {}; apply() }
    }

    private fun stepIso(by: Int) {
        if (mode != Mode.IDLE || !manual) return
        val i = (isos.indexOf(iso) + by).coerceIn(0, isos.size - 1)
        iso = isos[i]; labels(); cam.post { apply() }
    }

    // ---------------------------------------------------------------- frames
    /** Every frame the camera makes (camera thread). */
    private fun frame(img: Image) {
        if (wantLuma) lumas.offer(Pair(img.timestamp, luma(img)))
        if (mode != Mode.RECORD) return
        val w = writer ?: return
        toI420(img)
        w.write(i420, 1)
        times.add(img.timestamp)
        val n = times.size
        if (n % 5 == 0 || n < 5) show(String.format(Locale.US,
            "recording: %d frames, %.0f s", n, (img.timestamp - times[0]) / 1e9))
        // the stars' size now and then: a lens pointed up sags, and cools
        if (focusedHfr != null && n % 40 == 0 && !checking) {
            checking = true
            val y = luma(img)
            thread(name = "skyfind-focus-check") {
                try {
                    val grown = py().callAttr("focus_check", y, size.width, size.height,
                        focusedHfr).toString()
                    if (grown.isNotEmpty()) say(grown)
                } catch (e: Throwable) { Log.w(TAG, "focus check", e) }
                checking = false
            }
        }
    }

    private fun luma(img: Image): ByteArray {
        val p = img.planes[0]
        val out = ByteArray(img.width * img.height)
        val buf = p.buffer
        for (y in 0 until img.height) {
            buf.position(y * p.rowStride)
            buf.get(out, y * img.width, img.width)
        }
        return out
    }

    /** The frame as I420 (whole macroblocks: the right and bottom edges
     *  trimmed to multiples of 16). */
    private fun toI420(img: Image) {
        val w = img.width / 16 * 16
        val h = img.height / 16 * 16
        if (i420.size != w * h * 3 / 2) i420 = ByteArray(w * h * 3 / 2)
        val y = img.planes[0]
        for (r in 0 until h) {
            y.buffer.position(r * y.rowStride)
            y.buffer.get(i420, r * w, w)
        }
        var at = w * h
        val row = ByteArray(img.planes[1].rowStride)
        for (pl in 1..2) {
            val p = img.planes[pl]
            val buf = p.buffer
            for (r in 0 until h / 2) {
                buf.position(r * p.rowStride)
                val n = minOf(row.size, buf.remaining())
                buf.get(row, 0, n)
                var x = 0
                val step = p.pixelStride
                for (c in 0 until w / 2) { i420[at++] = row[x]; x += step }
            }
        }
    }

    // ----------------------------------------------------------------- focus
    private fun startFocus() {
        if (mode != Mode.IDLE || !manual || session == null) return
        val minFocus = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        if (minFocus <= 0f) { show("This lens is fixed-focus: nothing to adjust."); return }
        mode = Mode.FOCUS
        focusBtn.isEnabled = false; recordBtn.isEnabled = false
        thread(name = "skyfind-focus") {
            val keep = exposureNs
            try {
                // short enough to step quickly, long enough for stars
                exposureNs = minOf(exposureNs,
                    exposures.lastOrNull { it <= 2_000_000_000L } ?: exposures.first())
                val hyper = chars.get(CameraCharacteristics.LENS_INFO_HYPERFOCAL_DISTANCE) ?: 0f
                val sweep = py().callAttr("focus_sweep", minFocus, hyper)
                var stop = 0
                while (true) {
                    val x = sweep.callAttr("next") ?: break
                    stop++
                    val frames = at(x.toFloat(), 1)
                    val said = py().callAttr("focus_add", sweep, frames.toTypedArray(),
                        size.width, size.height).toString()
                    say(said); show("focusing: stop $stop  $said")
                }
                val best = sweep.get("best")
                if (best != null) {
                    lens = best.toFloat()
                    focusedHfr = sweep.get("hfr")?.toDouble()
                }
                val sum = sweep.callAttr("summary").toString()
                say(sum); show(sum)
            } catch (e: Throwable) {
                Log.e(TAG, "focus", e); show("Focus failed: ${e.message}")
            } finally {
                wantLuma = false; lensTarget = -1f
                exposureNs = keep
                cam.post { apply() }
                mode = Mode.IDLE
                runOnUiThread { focusBtn.isEnabled = true; recordBtn.isEnabled = true }
            }
        }
    }

    /** The lens sent to `x`, and `n` frames exposed wholly after it came to
     *  rest there. */
    private fun at(x: Float, n: Int): List<ByteArray> {
        val frameMs = exposureNs / 1_000_000 + 400
        wantLuma = false; lumas.clear()
        settledAt = 0L; lensTarget = x; lens = x
        cam.post { apply() }
        val until = SystemClock.elapsedRealtime() + 4 * frameMs + 3000
        while (settledAt == 0L && SystemClock.elapsedRealtime() < until) Thread.sleep(20)
        // a camera that never reports its lens: two frames' grace instead
        val after = if (settledAt != 0L) settledAt else {
            Thread.sleep(2 * frameMs); Long.MAX_VALUE }
        wantLuma = true
        val out = ArrayList<ByteArray>()
        var waited = 0
        while (out.size < n && waited < n + 4) {
            val f = lumas.poll(3 * frameMs, TimeUnit.MILLISECONDS) ?: break
            waited++
            if (after == Long.MAX_VALUE || f.first > after) out.add(f.second)
        }
        wantLuma = false
        return out
    }

    // ---------------------------------------------------------------- record
    private fun startRecording() {
        if (mode != Mode.IDLE || session == null) return
        if (askedNs != 0L) {
            // frames at the old exposure are still coming (the test phone took
            // half a minute to go from 8 s to 4 s): the clip would be
            // recorded at one exposure and labelled with another
            show("Wait: the camera hasn't taken the new exposure yet."); return
        }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC") }.format(Date())
        val dir = getExternalFilesDir("captures")!!.apply { mkdirs() }
        val f = File(dir, "sky_$stamp.mp4")
        // UTC of the frames' clock: the GPS's time where there's a fix (it
        // comes stamped on the boot clock, as the frames are), else the
        // phone's own
        val boot = SystemClock.elapsedRealtimeNanos()
        val now = if (realtime) boot else System.nanoTime()
        val l = fix
        val utc = if (l != null) {
            clockSource = "gps"; l.time * 1_000_000L + (boot - l.elapsedRealtimeNanos)
        } else {
            clockSource = "system"; System.currentTimeMillis() * 1_000_000L
        }
        clockOffsetNs = utc - now
        mode = Mode.RECORD
        cam.post {
            times.clear()
            clip = f
            try {
                writer = Mp4Writer(f.path, size.width / 16 * 16, size.height / 16 * 16, 30)
            } catch (e: Exception) {
                Log.e(TAG, "encoder", e)
                show("The phone's encoder wouldn't start: ${e.message}")
                runOnUiThread { stopRecording() }
            }
        }
        recordBtn.text = "Stop"
        focusBtn.isEnabled = false; useBtn.visibility = View.GONE
        window.attributes = window.attributes.apply { screenBrightness = 0.02f }
        say("recording ${f.name}" + if (focusedHfr == null && manual)
            " (not focused on the stars: the lens is at its own infinity)" else "")
    }

    private fun stopRecording() {
        if (mode != Mode.RECORD) return
        mode = Mode.IDLE
        recordBtn.text = "Record"; focusBtn.isEnabled = manual
        window.attributes = window.attributes.apply {
            screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE }
        cam.post {
            val w = writer ?: return@post
            writer = null
            try { w.close() } catch (e: Exception) { Log.e(TAG, "closing the clip", e) }
            val f = clip ?: return@post
            if (times.size < 30) {
                say("only ${times.size} frames: too few to solve (the trails are found " +
                    "against the frames around them)")
            }
            File(f.path.removeSuffix(".mp4") + ".json").writeText(sidecar().toString(1))
            say("saved ${f.name}: ${times.size} frames, ${f.length() / 1_000_000} MB")
            show("Recorded ${times.size} frames.")
            runOnUiThread { useBtn.visibility = if (times.size >= 30) View.VISIBLE else View.GONE }
        }
    }

    /** Where and when the clip was shot, for the solve (run_known). */
    private fun sidecar(): JSONObject {
        val o = JSONObject()
        val e = (if (gotNs > 0) gotNs else exposureNs) / 1e9     // as the sensor reported it
        if (times.isNotEmpty()) {
            o.put("frame0_unix", (times[0] + clockOffsetNs) / 1e9 + (if (manual) e / 2 else 0.0))
            o.put("frame_times_s", JSONArray(times.map { (it - times[0]) / 1e9 }))
        }
        o.put("clock", clockSource)
        if (manual) { o.put("exposure_s", e); o.put("iso", iso); o.put("lens_dioptres", lens.toDouble()) }
        focusedHfr?.let { o.put("focus_hfr_px", it) }
        fix?.let {
            o.put("lat", it.latitude); o.put("lon", it.longitude)
            if (it.hasAltitude()) o.put("height_m", it.altitude)
            o.put("accuracy_m", it.accuracy.toDouble())
        }
        o.put("camera", cameraId); o.put("width", size.width / 16 * 16)
        o.put("height", size.height / 16 * 16)
        return o
    }

    private fun useClip() {
        val f = clip ?: return
        setResult(Activity.RESULT_OK, Intent()
            .putExtra("clip_path", f.path)
            .putExtra("capture_path", f.path.removeSuffix(".mp4") + ".json"))
        finish()
    }

    companion object { const val TAG = "skyfind" }
}
