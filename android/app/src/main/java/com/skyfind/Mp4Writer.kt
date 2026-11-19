package com.skyfind

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer

/**
 * H.264 MP4 from I420 frames with the phone's encoder: what the desktop
 * does through ffmpeg for the replay (skyfind/replay.py `VideoOut` calls
 * this on Android). Width and height are whole macroblocks (multiples of 16).
 */
class Mp4Writer(path: String, private val width: Int, private val height: Int,
                private val fps: Int) {
    private val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
    private val muxer = MediaMuxer(path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    private val info = MediaCodec.BufferInfo()
    private var track = -1
    private var frames = 0L

    init {
        val fmt = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
        fmt.setInteger(MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        fmt.setInteger(MediaFormat.KEY_BIT_RATE, width * height * 8)
        fmt.setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        fmt.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
        codec.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
    }

    /** One frame (I420: all of Y, then U, then V), shown `repeat` times. */
    fun write(i420: ByteArray, repeat: Int) {
        for (r in 0 until repeat) feed(i420)
    }

    private fun feed(i420: ByteArray) {
        var idx: Int
        while (true) {
            idx = codec.dequeueInputBuffer(10_000)
            if (idx >= 0) break
            drain(false)
        }
        val image = codec.getInputImage(idx)!!
        val cw = width / 2
        val ch = height / 2
        // Y, then U and V (planar or interleaved, whatever the encoder has)
        for ((plane, spec) in image.planes.zip(listOf(
                Triple(0, width, height), Triple(width * height, cw, ch),
                Triple(width * height + cw * ch, cw, ch)))) {
            val (start, w, h) = spec
            val buf = plane.buffer
            val rs = plane.rowStride
            val ps = plane.pixelStride
            for (y in 0 until h) {
                val src = start + y * w
                if (ps == 1) {
                    buf.position(y * rs)
                    buf.put(i420, src, w)
                } else {
                    val base = y * rs
                    for (x in 0 until w) buf.put(base + x * ps, i420[src + x])
                }
            }
        }
        codec.queueInputBuffer(idx, 0, width * height * 3 / 2,
            frames * 1_000_000L / fps, 0)
        frames++
        drain(false)
    }

    private fun drain(end: Boolean) {
        while (true) {
            val out = codec.dequeueOutputBuffer(info, if (end) 10_000 else 0)
            when {
                out == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!end) return
                out == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    track = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                }
                out >= 0 -> {
                    val buf = codec.getOutputBuffer(out)!!
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                    if (info.size > 0 && track >= 0) {
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        muxer.writeSampleData(track, buf, info)
                    }
                    codec.releaseOutputBuffer(out, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }

    fun close() {
        var idx: Int
        while (true) {
            idx = codec.dequeueInputBuffer(10_000)
            if (idx >= 0) break
            drain(false)
        }
        codec.queueInputBuffer(idx, 0, 0, frames * 1_000_000L / fps,
            MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        drain(true)
        codec.stop()
        codec.release()
        if (track >= 0) muxer.stop()
        muxer.release()
    }
}
