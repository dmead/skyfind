package com.skyfind

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.text.InputType
import android.text.util.Linkify
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import kotlin.concurrent.thread

/**
 * Pick a timelapse, solve it (SolveService), see the log as it goes; the
 * pictures and the replay land in Google Photos (Pictures/skyfind,
 * Movies/skyfind).
 */
class MainActivity : AppCompatActivity() {
    private var clip: Uri? = null
    private var recorded: String? = null  // a clip recorded here, and
    private var capture: String? = null   // where and when it was (its JSON)
    private val tles = mutableListOf<Uri>()
    private lateinit var clipLabel: TextView
    private lateinit var tleLabel: TextView
    private lateinit var loginLabel: TextView
    private lateinit var user: EditText
    private lateinit var password: EditText
    private lateinit var around: EditText
    private lateinit var freeSite: CheckBox
    private lateinit var solve: Button
    private lateinit var status: TextView
    private lateinit var logView: TextView
    private lateinit var scroll: ScrollView
    private lateinit var results: LinearLayout
    private lateinit var live: ImageView
    @Volatile private var watching = false
    private var liveThread: Thread? = null
    private var pointedAt = 0L            // the failed run already pointed at the login
    private val refresh = { runOnUiThread { show() } }

    private val pickClip = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) { clip = uri; recorded = null; capture = null; clipLabel.text = name(uri) }
    }
    private val record = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { r ->
        val path = r.data?.getStringExtra("clip_path")
        if (r.resultCode == RESULT_OK && path != null) {
            recorded = path; capture = r.data?.getStringExtra("capture_path"); clip = null
            clipLabel.text = File(path).name + " (recorded here: its site and time are known)"
        }
    }
    private val pickTles = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        tles.addAll(uris); tleLabel.text = tleText()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        fun add(v: View) = col.addView(v)
        fun label(text: String, size: Float = 14f) = TextView(this).apply {
            this.text = text; textSize = size
        }

        add(label("skyfind", 26f).apply { setTypeface(typeface, Typeface.BOLD) })
        add(label("Where and when a night timelapse was shot, and which satellites crossed it."))

        // the solve under way (or the last one) first: its stage, the story
        // as it happens (the desktop's live window), the results
        status = label("").also { add(it) }
        live = ImageView(this).apply {
            adjustViewBounds = true
            visibility = View.GONE
        }
        add(live)
        results = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        add(results)

        add(Button(this).apply { text = "Record the sky"; setOnClickListener {
            record.launch(Intent(this@MainActivity, CaptureActivity::class.java)) } })
        add(Button(this).apply { text = "Choose a clip"; setOnClickListener {
            pickClip.launch(arrayOf("video/*")) } })
        clipLabel = label("no clip chosen").also { add(it) }

        // the satellites are named from the night's orbital elements: the
        // login fetches them (or files the owner saved from the website)
        add(label("\nSpace-Track login", 18f).apply { setTypeface(typeface, Typeface.BOLD) })
        add(label("The satellites are named from Space-Track's orbital elements for the " +
            "night. Your login is kept encrypted on this phone and sent only to Space-Track. " +
            "No account? A free one: https://www.space-track.org/auth/createAccount").apply {
            autoLinkMask = Linkify.WEB_URLS; linksClickable = true })
        val settings = Settings(this)
        user = EditText(this).apply { hint = "Space-Track user name"; setText(settings.user ?: "")
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS }
        password = EditText(this).apply { hint = "Space-Track password"; setText(settings.password ?: "")
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            typeface = Typeface.DEFAULT }            // the password type sets monospace
        add(user); add(password)
        add(Button(this).apply { text = "Check login"; setOnClickListener { checkLogin() } })
        loginLabel = label("").also { add(it) }

        add(label("\nOr element files (.3le / .tle) saved from Space-Track's website"))
        add(Button(this).apply { text = "Add orbital elements"; setOnClickListener {
            pickTles.launch(arrayOf("*/*")) } })
        tleLabel = label(tleText()).also { add(it) }
        add(Button(this).apply { text = "Clear elements"; setOnClickListener {
            tles.clear(); tleLabel.text = tleText() } })

        add(label("\nOptional"))
        around = EditText(this).apply { hint = "night, if the sky can't date it: YYYY-MM-DD HH:MM (UTC)" }
        add(around)
        freeSite = CheckBox(this).apply { text = "Free site (don't trust the skyline's latitude)" }
        add(freeSite)

        solve = Button(this).apply { text = "Solve"; setOnClickListener { start() } }
        add(solve)
        logView = TextView(this).apply {
            typeface = Typeface.MONOSPACE; textSize = 11f; setTextIsSelectable(true)
        }
        add(logView)

        scroll = ScrollView(this).apply { addView(col) }
        setContentView(scroll)

        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
    }

    override fun onResume() {
        super.onResume(); SolveLog.listen(refresh); show()
        watching = true; watchStory()
    }

    override fun onPause() {
        watching = false
        SolveLog.unlisten(refresh); super.onPause()
    }

    /** While the screen is showing: every few seconds, the story so far
     *  drawn in Python (skyfind_android.live_frame) when there's more of
     *  it; nothing to draw until the solve begins (after the trails). */
    private fun watchStory() {
        if (liveThread?.isAlive == true) return
        val out = File(cacheDir, "live.png")
        liveThread = thread(name = "skyfind-live") {
            var run = -1L
            while (watching) {
                // a new solve (or the screen's first look) starts blank: the
                // work folder's story may be an earlier solve's until this
                // one's begins (a run that fails before it never replaces it)
                val started = SolveLog.started
                if (started != run) {
                    run = started
                    out.delete()
                    runOnUiThread { live.visibility = View.GONE }
                }
                val clip = SolveLog.clipPath
                val work = SolveLog.workPath
                if (clip != null && work != null && SolveLog.state != SolveLog.State.IDLE) {
                    val changed = try {
                        SolveService.python(this).getModule("skyfind_android")
                            .callAttr("live_frame", clip, work, out.path, started).toBoolean()
                    } catch (e: Throwable) {
                        Log.w(SolveService.TAG, "live view", e); false
                    }
                    if (changed && out.exists()) {
                        val bmp = BitmapFactory.decodeFile(out.path)
                        if (bmp != null)
                            runOnUiThread { live.setImageBitmap(bmp); live.visibility = View.VISIBLE }
                    }
                }
                Thread.sleep(3000)
            }
        }
    }

    private fun saveLogin() {
        val s = Settings(this)
        s.user = user.text.toString().trim()
        s.password = password.text.toString()
    }

    /** Scrolls to the login and marks it (focus brings it into view). */
    private fun pointAtLogin(why: String) {
        val field = if (user.text.isBlank()) user else password
        field.requestFocus()
        field.error = why
    }

    /** Signs in to Space-Track (nothing downloaded) to say now whether the
     *  login works, rather than after a quarter of an hour of trails. */
    private fun checkLogin() {
        val u = user.text.toString().trim()
        val p = password.text.toString()
        if (u.isEmpty() || p.isEmpty()) {
            loginLabel.text = "Type the user name and the password first."
            pointAtLogin("needed"); return
        }
        saveLogin()
        loginLabel.text = "signing in..."
        thread(name = "skyfind-login") {
            val why = try {
                SolveService.python(this).getModule("skyfind_android")
                    .callAttr("check_login", u, p).toString()
            } catch (e: Throwable) { e.message ?: e.toString() }
            runOnUiThread {
                if (why.isEmpty()) {
                    loginLabel.text = "Signed in: the login works."
                    user.error = null; password.error = null
                } else {
                    loginLabel.text = why
                    pointAtLogin("check this")
                }
            }
        }
    }

    private fun start() {
        val c = clip
        if (c == null && recorded == null) { status.text = "Record or choose a clip first."; return }
        if (tles.isEmpty() && (user.text.isBlank() || password.text.isEmpty())) {
            status.text = "Type your Space-Track login first (or add element files): " +
                "the satellites are named from the night's orbital elements."
            pointAtLogin("needed"); return
        }
        saveLogin()
        user.error = null; password.error = null
        val i = Intent(this, SolveService::class.java)
            .apply { if (c != null) putExtra("clip_uri", c) else {
                putExtra("clip_path", recorded); putExtra("capture_path", capture) } }
            .putParcelableArrayListExtra("tle_uris", ArrayList(tles))
            .putExtra("around", around.text.toString().trim())
            .putExtra("free_site", freeSite.isChecked)
        startForegroundService(i)
        status.text = "starting..."
    }

    private fun show() {
        val running = SolveLog.state == SolveLog.State.RUNNING
        solve.isEnabled = !running
        val mins = if (SolveLog.started > 0)
            (System.currentTimeMillis() - SolveLog.started) / 60000 else 0
        status.text = when (SolveLog.state) {
            SolveLog.State.IDLE -> ""
            SolveLog.State.RUNNING -> "${SolveLog.clip}: ${SolveLog.stage} ($mins min so far)"
            SolveLog.State.DONE -> "${SolveLog.clip}: done - in Google Photos (Pictures/skyfind, Movies/skyfind)"
            SolveLog.State.FAILED -> "${SolveLog.clip}: failed. ${SolveLog.error}"
        }
        if (SolveLog.state == SolveLog.State.FAILED && SolveLog.needsLogin &&
                pointedAt != SolveLog.started) {
            pointedAt = SolveLog.started
            pointAtLogin("check this")
        }
        results.removeAllViews()
        for (o in SolveLog.outputs) {
            results.addView(Button(this).apply {
                text = "Open ${o.name}"
                setOnClickListener {
                    startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(o.uri, o.mime)
                        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
                }
            })
        }
        logView.text = SolveLog.text().lines().takeLast(200).joinToString("\n")
    }

    private fun tleText() = if (tles.isEmpty()) "none: Space-Track is asked, with the login above"
        else tles.joinToString { name(it) }

    private fun name(uri: Uri): String {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { if (it.moveToFirst()) return it.getString(0) ?: "" }
        return uri.lastPathSegment ?: uri.toString()
    }
}
