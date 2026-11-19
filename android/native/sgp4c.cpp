// SGP4 for skyfind on Android, called through ctypes (skyfind/sgp4lib.py):
// Vallado's code (vallado/, as python-sgp4 2.27 ships it) behind a small C
// API that does exactly what python-sgp4's extension does for
// Satrec.twoline2rv and SatrecArray.sgp4 (extension/wrapper.cpp), so the
// phone's orbit table matches the desktop's.
#include <cmath>
#include <cstdint>
#include <cstring>

#include "vallado/SGP4.h"

extern "C" {

elsetrec *sgp4c_alloc(int n) { return new elsetrec[n]; }

void sgp4c_free(elsetrec *p) { delete[] p; }

int sgp4c_size() { return (int) sizeof(elsetrec); }

// Satrec.twoline2rv(line1, line2) into p[i]; returns its error code.
int sgp4c_twoline2rv(elsetrec *p, int i, const char *s1, const char *s2) {
    char line1[130], line2[130];
    double dummy;
    strncpy(line1, s1, 130);
    strncpy(line2, s2, 130);
    line1[129] = line2[129] = '\0';
    // before any checksum: scanf would read it into the value before
    line1[68] = '\0';
    line2[68] = '\0';
    // leading spaces in the catalogue number: zeros, as the wrapper does
    for (int k = 2; k < 7; k++) {
        if (line1[k] == ' ') line1[k] = '0';
        if (line2[k] == ' ') line2[k] = '0';
    }
    elsetrec &s = p[i];
    SGP4Funcs::twoline2rv(line1, line2, ' ', ' ', 'i', wgs72,
                          dummy, dummy, dummy, s);
    s.jdsatepochF = round(s.jdsatepochF * 1e8) / 1e8;
    return s.error;
}

// The epoch (whole and fractional Julian day) of p[i].
void sgp4c_epoch(const elsetrec *p, int i, double *jd, double *fr) {
    *jd = p[i].jdsatepoch;
    *fr = p[i].jdsatepochF;
}

// SatrecArray.sgp4(jd, fr) for satellites p[lo..hi): e (n, nt) uint8,
// r and v (n, nt, 3) double, rows lo.. of them; NaN where SGP4 fails.
void sgp4c_array(elsetrec *p, int lo, int hi, const double *jd,
                 const double *fr, int nt, uint8_t *e, double *r, double *v) {
    for (int i = lo; i < hi; i++) {
        elsetrec &satrec = p[i];
        for (int j = 0; j < nt; j++) {
            double t = (jd[j] - satrec.jdsatepoch) * 1440.0
                     + (fr[j] - satrec.jdsatepochF) * 1440.0;
            long k1 = (long) i * nt + j;
            long k3 = 3 * k1;
            SGP4Funcs::sgp4(satrec, t, r + k3, v + k3);
            e[k1] = (uint8_t) satrec.error;
            if (satrec.error && satrec.error < 6) {
                r[k3] = r[k3 + 1] = r[k3 + 2] = NAN;
                v[k3] = v[k3 + 1] = v[k3 + 2] = NAN;
            }
        }
    }
}

}  // extern "C"
