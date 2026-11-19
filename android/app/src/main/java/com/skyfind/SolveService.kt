package com.skyfind

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.IBinder
import android.os.PowerManager
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import kotlin.concurrent.thread

/**
 * Runs a solve in the background (a foreground service, the CPU kept awake):
 * the clip and any element files copied in, skyfind's pipeline in Python
 * (skyfind_android.solve), the pictures and replay saved where Google Photos
 * finds them.
 *
 * From the app: extras "clip_uri" (Uri) and "tle_uris" (Uri list). For tests
 * over adb, file paths the app can read (its own external files folder):
 *   adb shell am start-foreground-service -n com.skyfind/.SolveService \
 *     --es clip_path /sdcard/Android/data/com.skyfind/files/in/night.mp4 \
 *     --esa tle_paths /sdcard/Android/data/com.skyfind/files/in/night.3le
 * plus "around" ("YYYY-MM-DD HH:MM"), "free_site" (boolean) and
 * "capture_path" (a clip recorded by the app: CaptureActivity's JSON of
 * where and when).
 */
class SolveService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        channel(this)
        val n = notification("starting")
        if (Build.VERSION.SDK_INT >= 29)
            startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        else startForeground(ID, n)
        if (SolveLog.state == SolveLog.State.RUNNING) return START_NOT_STICKY
        val args = intent ?: return START_NOT_STICKY
        thread(name = "skyfind-solve") { run(args); stopSelf() }
        return START_NOT_STICKY
    }

    private fun run(intent: Intent) {
        // a check over adb: one decoded frame to a PNG, no solve
        intent.getStringExtra("dump_frame")?.let { clip ->
            val out = File(getExternalFilesDir(null), "frame.png").path
            val r = try {
                python(this).getModule("skyfind_android").callAttr("dump_frame",
                    clip, out, intent.getIntExtra("k", 0)).toString()
            } catch (e: Throwable) { "FAILED: $e" }
            Log.i(TAG, "dump_frame: $r")
            stopForeground(STOP_FOREGROUND_REMOVE)
            return
        }
        val wake = (getSystemService(POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "skyfind:solve")
        wake.acquire(6 * 3600 * 1000L)
        try {
            val clip = intent.getStringExtra("clip_path")?.let { File(it) }
                ?: copyIn(intent.parcelable("clip_uri")!!, "clips")
            val tles = (intent.getStringArrayExtra("tle_paths")?.map { File(it) }
                ?: intent.parcelables("tle_uris").map { copyIn(it, "elements") })
            val work = File(getExternalFilesDir("work"), clip.nameWithoutExtension + ".skyfind")
            SolveLog.start(clip.name, File(work, "app.log"), clip, work)
            SolveLog.line("clip ${clip.name} (${clip.length() / 1_000_000} MB)" +
                (if (tles.isNotEmpty()) "; elements: " + tles.joinToString { it.name } else ""))
            val py = python(this)
            val prefs = Settings(this)
            val sink = Sink(this)
            val made = py.getModule("skyfind_android").callAttr(
                "solve", clip.path, work.path, tles.map { it.path }.toTypedArray(),
                intent.getStringExtra("around")?.ifBlank { null },
                intent.getBooleanExtra("free_site", false),
                prefs.user, prefs.password, sink,
                intent.getStringExtra("capture_path"))
            val outs = made.asList().map { File(it.toString()) }.filter { it.exists() }
                .mapNotNull { save(it, clip.nameWithoutExtension) }
            SolveLog.finish(outs)
            update("done: ${outs.size} files in Photos")
        } catch (e: Throwable) {
            // skyfind's own failures say what's wrong in their message; the
            // app's (skyfind_android.AppError, LoginNeeded) are for the phone
            Log.e(TAG, "solve failed", e)
            val msg = e.message ?: e.toString()
            SolveLog.fail(msg.removePrefix("LoginNeeded: ").removePrefix("AppError: "),
                login = msg.startsWith("LoginNeeded: "))
            update(if (SolveLog.needsLogin) "failed: check the Space-Track login" else "failed")
        } finally {
            if (wake.isHeld) wake.release()
            stopForeground(STOP_FOREGROUND_DETACH)
        }
    }

    /** Called from Python (skyfind_android.solve's `sink`). */
    class Sink(private val svc: SolveService) {
        fun line(s: String) { Log.i(TAG, s); SolveLog.line(s) }
        fun stage(s: String) { SolveLog.stage(s); svc.update(s) }
    }

    private fun copyIn(uri: Uri, folder: String): File {
        var name = "input"
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { if (it.moveToFirst()) name = it.getString(0) ?: name }
        val dir = File(filesDir, folder).apply { mkdirs() }
        val out = File(dir, name.replace('/', '_'))
        contentResolver.openInputStream(uri)!!.use { i -> out.outputStream().use { i.copyTo(it, 1 shl 20) } }
        return out
    }

    /** Into Pictures/skyfind or Movies/skyfind, where Google Photos shows it. */
    private fun save(f: File, clip: String): SolveLog.Output? {
        val mime = when (f.extension.lowercase()) {
            "jpg", "jpeg" -> "image/jpeg"; "png" -> "image/png"; "mp4" -> "video/mp4"
            else -> return null
        }
        val video = mime.startsWith("video")
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, f.name)
            put(MediaStore.MediaColumns.MIME_TYPE, mime)
            put(MediaStore.MediaColumns.RELATIVE_PATH,
                (if (video) Environment.DIRECTORY_MOVIES else Environment.DIRECTORY_PICTURES) + "/skyfind")
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val coll = if (video) MediaStore.Video.Media.EXTERNAL_CONTENT_URI
                   else MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        // a clip solved again replaces its pictures (ours to replace),
        // rather than piling up "name (1).jpg" copies in Photos
        contentResolver.delete(coll,
            "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND ${MediaStore.MediaColumns.RELATIVE_PATH}=?",
            arrayOf(f.name, values.getAsString(MediaStore.MediaColumns.RELATIVE_PATH) + "/"))
        val uri = contentResolver.insert(coll, values) ?: return null
        contentResolver.openOutputStream(uri)!!.use { o -> f.inputStream().use { it.copyTo(o) } }
        contentResolver.update(uri, ContentValues().apply {
            put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
        SolveLog.line("saved ${f.name} for $clip")
        return SolveLog.Output(f.name, uri, mime)
    }

    private fun notification(text: String): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentTitle("skyfind")
            .setContentText(text)
            .setContentIntent(open)
            .setOngoing(SolveLog.state == SolveLog.State.RUNNING)
            .build()
    }

    fun update(text: String) {
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
            .notify(ID, notification(text))
    }

    companion object {
        const val TAG = "skyfind"
        const val CHANNEL = "solve"
        const val ID = 1

        fun channel(ctx: Context) {
            (ctx.getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
                .createNotificationChannel(NotificationChannel(CHANNEL, "Solving",
                    NotificationManager.IMPORTANCE_LOW))
        }

        /** Python started once, with the phone's settings and the reference
         *  data copied out of the app on first use. */
        @Synchronized fun python(ctx: Context): Python {
            if (!Python.isStarted()) Python.start(AndroidPlatform(ctx))
            val data = File(ctx.filesDir, "skyfind-data").apply { mkdirs() }
            val assets = ctx.assets
            for (name in assets.list("data") ?: emptyArray()) {
                val out = File(data, name)
                if (out.exists() && out.length() > 0) continue
                assets.open("data/$name").use { i -> out.outputStream().use { i.copyTo(it) } }
            }
            val py = Python.getInstance()
            py.getModule("skyfind_android").callAttr("configure", data.path,
                ctx.applicationInfo.nativeLibraryDir)
            return py
        }
    }
}

@Suppress("DEPRECATION")
private inline fun <reified T : android.os.Parcelable> Intent.parcelable(key: String): T? =
    if (Build.VERSION.SDK_INT >= 33) getParcelableExtra(key, T::class.java) else getParcelableExtra(key)

@Suppress("DEPRECATION")
private fun Intent.parcelables(key: String): List<Uri> =
    (if (Build.VERSION.SDK_INT >= 33) getParcelableArrayListExtra(key, Uri::class.java)
     else getParcelableArrayListExtra(key)) ?: emptyList()
