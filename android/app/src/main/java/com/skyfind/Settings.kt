package com.skyfind

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/** The Space-Track login (typed by the phone's owner), kept encrypted. */
class Settings(ctx: Context) {
    private val prefs = EncryptedSharedPreferences.create(
        ctx, "skyfind",
        MasterKey.Builder(ctx).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM)

    var user: String?
        get() = prefs.getString("spacetrack_user", null)?.ifBlank { null }
        set(v) = prefs.edit().putString("spacetrack_user", v).apply()

    var password: String?
        get() = prefs.getString("spacetrack_password", null)?.ifBlank { null }
        set(v) = prefs.edit().putString("spacetrack_password", v).apply()
}
