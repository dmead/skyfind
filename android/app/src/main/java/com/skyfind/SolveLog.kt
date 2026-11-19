package com.skyfind

import android.net.Uri
import java.io.File

/** What the running (or last) solve is doing, for the screen to show. */
object SolveLog {
    enum class State { IDLE, RUNNING, DONE, FAILED }

    data class Output(val name: String, val uri: Uri, val mime: String)

    private val lines = ArrayDeque<String>()
    var state = State.IDLE
        private set
    var stage = ""
        private set
    var clip = ""
        private set
    var started = 0L
        private set
    /** The running (or last) solve's clip and work folder: the live view
     *  draws the story from its event log. */
    var clipPath: String? = null
        private set
    var workPath: String? = null
        private set
    val outputs = mutableListOf<Output>()
    /** The last failure, in the app's words, and whether the Space-Track
     *  login is what's missing or wrong (the screen points at it). */
    var error = ""
        private set
    var needsLogin = false
        private set
    private var file: File? = null
    private val listeners = mutableListOf<() -> Unit>()

    @Synchronized fun start(name: String, logFile: File, clipFile: File, work: File) {
        lines.clear(); outputs.clear()
        clip = name; stage = "starting"; state = State.RUNNING
        clipPath = clipFile.path; workPath = work.path
        error = ""; needsLogin = false
        started = System.currentTimeMillis()
        file = logFile.also { it.parentFile?.mkdirs(); it.writeText("") }
        notifyAll_()
    }

    @Synchronized fun line(s: String) {
        for (ln in s.split("\n")) {
            lines.addLast(ln)
            while (lines.size > 600) lines.removeFirst()
        }
        file?.appendText(s + "\n")
        notifyAll_()
    }

    @Synchronized fun stage(s: String) { stage = s; line("== $s"); }

    @Synchronized fun finish(outs: List<Output>) {
        outputs.clear(); outputs.addAll(outs)
        state = State.DONE; stage = "done"
        line("== done: ${outs.size} files saved to the phone's Pictures/Movies (Google Photos)")
    }

    @Synchronized fun fail(text: String, login: Boolean = false) {
        state = State.FAILED; stage = "failed"
        error = text; needsLogin = login
        line("FAILED: $text")
    }

    @Synchronized fun text(): String = lines.joinToString("\n")

    @Synchronized fun listen(fn: () -> Unit) { listeners.add(fn) }
    @Synchronized fun unlisten(fn: () -> Unit) { listeners.remove(fn) }
    private fun notifyAll_() { for (l in listeners.toList()) l() }
}
