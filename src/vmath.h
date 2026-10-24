// AVX2 vector math used on hot paths (softmax, SwiGLU-OAI). exp256_ps is the Cephes polynomial
// (relative error ~2e-7 on [-88, 88]); inputs below -88 return ~0, as softmax wants.
#ifndef INZ_VMATH_H
#define INZ_VMATH_H

#include <immintrin.h>
#include <math.h>

static inline __m256 exp256_ps(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_add_ps(_mm256_fmadd_ps(y, z, x), _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

static inline float hmax256_ps(__m256 v) {
    __m128 lo = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// In-place softmax of s[0..n). Returns nothing; s sums to 1.
static inline void softmax_inplace(float *s, int n) {
    __m256 vm = _mm256_set1_ps(-INFINITY);
    int i = 0;
    for (; i + 8 <= n; i += 8) vm = _mm256_max_ps(vm, _mm256_loadu_ps(s + i));
    float mx = hmax256_ps(vm);
    for (; i < n; i++) mx = s[i] > mx ? s[i] : mx;
    const __m256 vmx = _mm256_set1_ps(mx);
    __m256 vs = _mm256_setzero_ps();
    for (i = 0; i + 8 <= n; i += 8) {
        const __m256 v = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(s + i), vmx));
        _mm256_storeu_ps(s + i, v);
        vs = _mm256_add_ps(vs, v);
    }
    float sum = hsum256_ps(vs);
    for (; i < n; i++) { s[i] = expf(s[i] - mx); sum += s[i]; }
    const __m256 inv = _mm256_set1_ps(1.0f / sum);
    for (i = 0; i + 8 <= n; i += 8) _mm256_storeu_ps(s + i, _mm256_mul_ps(_mm256_loadu_ps(s + i), inv));
    for (; i < n; i++) s[i] *= 1.0f / sum;
}

// SwiGLU-OAI over a row: g[r] = min(g,L) * sigmoid(alpha*min(g,L)) * (clamp(u,-L,L) + 1)
static inline void swigluoai_row(float *g, const float *u, int n, float alpha, float limit) {
    const __m256 L = _mm256_set1_ps(limit), nL = _mm256_set1_ps(-limit), na = _mm256_set1_ps(-alpha);
    const __m256 one = _mm256_set1_ps(1.0f);
    int r = 0;
    for (; r + 8 <= n; r += 8) {
        const __m256 gv = _mm256_min_ps(_mm256_loadu_ps(g + r), L);
        const __m256 uv = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(u + r), nL), L);
        const __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_mul_ps(na, gv))));
        _mm256_storeu_ps(g + r, _mm256_mul_ps(_mm256_mul_ps(gv, sig), _mm256_add_ps(uv, one)));
    }
    for (; r < n; r++) {
        float gg = g[r] < limit ? g[r] : limit;
        float uu = u[r] < -limit ? -limit : u[r] > limit ? limit : u[r];
        g[r] = gg / (1.0f + expf(-alpha * gg)) * (uu + 1.0f);
    }
}

#endif
