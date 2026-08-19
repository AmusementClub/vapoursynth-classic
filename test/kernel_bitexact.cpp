/* Standalone bit-exact checks for the AVX2/C++ ports.
 * Build (x64, AVX2):
 *   cl /nologo /O2 /EHsc /arch:AVX2 /I include test\kernel_bitexact.cpp
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <vector>

static int g_fails = 0;

static void fail(const char *msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++g_fails;
}

/* ---- FastDivU32 (copy of boxblurfilter.cpp) ---- */
struct FastDivU32 {
    uint32_t magic;
    uint32_t shift;
    uint32_t add;
};

static FastDivU32 makeFastDivU32(uint32_t d) {
    FastDivU32 fd;
    uint32_t l = 0;
    while ((d >> l) > 1)
        ++l;
    if ((d & (d - 1)) == 0) {
        fd.magic = 0;
        fd.shift = l;
        fd.add = 0;
        return fd;
    }
    uint64_t two_l = static_cast<uint64_t>(1) << (32 + l);
    uint32_t m = static_cast<uint32_t>(two_l / d);
    uint32_t rem = static_cast<uint32_t>(two_l - static_cast<uint64_t>(m) * d);
    if (d - rem < (1u << l)) {
        fd.magic = m + 1;
        fd.shift = l;
        fd.add = 0;
    } else {
        uint32_t twice = rem + rem;
        m += m;
        if (twice >= d || twice < rem)
            m += 1;
        fd.magic = m + 1;
        fd.shift = l;
        fd.add = 1;
    }
    return fd;
}

static inline uint32_t fastDivU32(uint32_t n, FastDivU32 fd) {
    if (fd.magic == 0)
        return n >> fd.shift;
    uint32_t q = static_cast<uint32_t>((static_cast<uint64_t>(fd.magic) * n) >> 32);
    if (fd.add) {
        uint32_t t = ((n - q) >> 1) + q;
        return t >> fd.shift;
    }
    return q >> fd.shift;
}

static void test_fastdiv() {
    const uint32_t ds[] = { 3, 5, 7, 9, 15, 17, 63, 65, 255, 511, 1023, 2047, 4095, 60001 };
    for (uint32_t d : ds) {
        FastDivU32 fd = makeFastDivU32(d);
        for (uint32_t n = 0; n < 200000; n += 1) {
            if (fastDivU32(n, fd) != n / d) {
                std::fprintf(stderr, "fastDiv mismatch d=%u n=%u got=%u want=%u\n", d, n, fastDivU32(n, fd), n / d);
                fail("fastDivU32");
                return;
            }
        }
        /* 16-bit boxblur acc range: max 65535*(2*r+1)+round */
        uint32_t accmax = 65535u * d + (d - 1);
        for (uint32_t n = accmax > 10000 ? accmax - 10000 : 0; n <= accmax; n += 17) {
            if (fastDivU32(n, fd) != n / d) {
                std::fprintf(stderr, "fastDiv acc-range mismatch d=%u n=%u\n", d, n);
                fail("fastDivU32 acc");
                return;
            }
        }
    }
}

/* ---- BoxBlur ring vs ping-pong ---- */
template<typename T>
static void blurH_div(const T *src, T *dst, int width, int radius, unsigned div, unsigned round) {
    unsigned acc = radius * src[0];
    for (int x = 0; x < radius; x++)
        acc += src[std::min(x, width - 1)];
    for (int x = 0; x < std::min(radius, width); x++) {
        acc += src[std::min(x + radius, width - 1)];
        dst[x] = static_cast<T>((acc + round) / div);
        acc -= src[std::max(x - radius, 0)];
    }
    if (width > radius) {
        for (int x = radius; x < width - radius; x++) {
            acc += src[x + radius];
            dst[x] = static_cast<T>((acc + round) / div);
            acc -= src[x - radius];
        }
        for (int x = std::max(width - radius, radius); x < width; x++) {
            acc += src[std::min(x + radius, width - 1)];
            dst[x] = static_cast<T>((acc + round) / div);
            acc -= src[std::max(x - radius, 0)];
        }
    }
}

template<typename T>
static void blurH_magic(const T *src, T *dst, int width, int radius, FastDivU32 fd, unsigned round) {
    unsigned acc = radius * src[0];
    for (int x = 0; x < radius; x++)
        acc += src[std::min(x, width - 1)];
    for (int x = 0; x < std::min(radius, width); x++) {
        acc += src[std::min(x + radius, width - 1)];
        dst[x] = static_cast<T>(fastDivU32(acc + round, fd));
        acc -= src[std::max(x - radius, 0)];
    }
    if (width > radius) {
        for (int x = radius; x < width - radius; x++) {
            acc += src[x + radius];
            dst[x] = static_cast<T>(fastDivU32(acc + round, fd));
            acc -= src[x - radius];
        }
        for (int x = std::max(width - radius, radius); x < width; x++) {
            acc += src[std::min(x + radius, width - 1)];
            dst[x] = static_cast<T>(fastDivU32(acc + round, fd));
            acc -= src[std::max(x - radius, 0)];
        }
    }
}

template<typename T>
static void blurH_inplace(const T *src, T *dst, int width, int radius, FastDivU32 fd, unsigned round, T *ring) {
    const int R = std::min(radius + 1, width);
    const unsigned first = src[0];
    int wr = 0;
    unsigned acc = radius * src[0];
    for (int x = 0; x < radius; x++)
        acc += src[std::min(x, width - 1)];
    for (int x = 0; x < std::min(radius, width); x++) {
        ring[wr] = src[x];
        if (++wr == R) wr = 0;
        acc += src[std::min(x + radius, width - 1)];
        dst[x] = static_cast<T>(fastDivU32(acc + round, fd));
        acc -= first;
    }
    if (width > radius) {
        for (int x = radius; x < width - radius; x++) {
            ring[wr] = src[x];
            if (++wr == R) wr = 0;
            acc += src[x + radius];
            dst[x] = static_cast<T>(fastDivU32(acc + round, fd));
            acc -= ring[wr];
        }
        for (int x = std::max(width - radius, radius); x < width; x++) {
            ring[wr] = src[x];
            if (++wr == R) wr = 0;
            acc += src[std::min(x + radius, width - 1)];
            dst[x] = static_cast<T>(fastDivU32(acc + round, fd));
            acc -= ring[wr];
        }
    }
}

static void test_boxblur() {
    struct Case { int w, r, passes; };
    Case cases[] = {
        {5, 8, 1}, {5, 8, 2}, {5, 8, 3},
        {8, 8, 2}, {16, 8, 2}, {17, 8, 2}, {9, 8, 2},
        {32, 7, 1}, {32, 7, 2}, {32, 7, 3},
        {1, 2, 2}, {2, 3, 3}, {15, 2, 2},
        {64, 31, 2},
    };
    for (Case c : cases) {
        unsigned div = c.r * 2 + 1;
        unsigned round0 = div - 1;
        FastDivU32 fd = makeFastDivU32(div);
        std::vector<uint8_t> src(c.w), ping(c.w), pong(c.w), dst(c.w), ring(c.r + 1);
        for (int i = 0; i < c.w; i++)
            src[i] = static_cast<uint8_t>((i * 37 + 11) & 255);
        /* ping-pong reference with hardware div */
        blurH_div(src.data(), ping.data(), c.w, c.r, div, round0);
        uint8_t *a = ping.data(), *b = pong.data();
        for (int p = 1; p < c.passes; p++) {
            blurH_div(a, b, c.w, c.r, div, (p & 1) ? 0 : round0);
            std::swap(a, b);
        }
        /* magic + ring */
        blurH_magic(src.data(), dst.data(), c.w, c.r, fd, round0);
        for (int p = 1; p < c.passes; p++)
            blurH_inplace(dst.data(), dst.data(), c.w, c.r, fd, (p & 1) ? 0 : round0, ring.data());
        if (std::memcmp(a, dst.data(), c.w) != 0) {
            std::fprintf(stderr, "boxblur mismatch w=%d r=%d passes=%d\n", c.w, c.r, c.passes);
            fail("boxblur ring/magic");
            return;
        }
        /* 16-bit too */
        std::vector<uint16_t> s16(c.w), p16(c.w), q16(c.w), d16(c.w), r16(c.r + 1);
        for (int i = 0; i < c.w; i++)
            s16[i] = static_cast<uint16_t>((i * 1301 + 77) & 65535);
        blurH_div(s16.data(), p16.data(), c.w, c.r, div, round0);
        uint16_t *a16 = p16.data(), *b16 = q16.data();
        for (int p = 1; p < c.passes; p++) {
            blurH_div(a16, b16, c.w, c.r, div, (p & 1) ? 0 : round0);
            std::swap(a16, b16);
        }
        blurH_magic(s16.data(), d16.data(), c.w, c.r, fd, round0);
        for (int p = 1; p < c.passes; p++)
            blurH_inplace(d16.data(), d16.data(), c.w, c.r, fd, (p & 1) ? 0 : round0, r16.data());
        if (std::memcmp(a16, d16.data(), c.w * 2) != 0) {
            std::fprintf(stderr, "boxblur16 mismatch w=%d r=%d passes=%d\n", c.w, c.r, c.passes);
            fail("boxblur16");
            return;
        }
    }
}

/* ---- Merge 8-bit: old 16-px vs pmulhrsw 32-px ---- */
#define MERGESHIFT 15
#define MROUND (1U << (MERGESHIFT - 1))

static void merge_byte_old(const uint8_t *srcp1, const uint8_t *srcp2, uint8_t *dstp, uint16_t w, unsigned n) {
    __m256i ww = _mm256_set1_epi16(w);
    for (unsigned i = 0; i < n; i += 16) {
        __m256i v1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(srcp1 + i)));
        __m256i v2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(srcp2 + i)));
        __m256i tmp1 = _mm256_slli_epi16(_mm256_sub_epi16(v2, v1), 1);
        __m256i tmp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_mulhi_epi16(tmp1, ww), _mm256_srli_epi16(_mm256_mullo_epi16(tmp1, ww), 15)), v1);
        __m256i result = _mm256_packus_epi16(tmp2, tmp2);
        result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
        _mm_storeu_si128((__m128i *)(dstp + i), _mm256_castsi256_si128(result));
    }
}

static void merge_byte_new(const uint8_t *srcp1, const uint8_t *srcp2, uint8_t *dstp, uint16_t w, unsigned n) {
    __m256i ww = _mm256_set1_epi16(w);
    for (unsigned i = 0; i < n; i += 32) {
        __m256i v1a = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(srcp1 + i)));
        __m256i v1b = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(srcp1 + i + 16)));
        __m256i v2a = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(srcp2 + i)));
        __m256i v2b = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(srcp2 + i + 16)));
        __m256i tmpa = _mm256_add_epi16(_mm256_mulhrs_epi16(_mm256_sub_epi16(v2a, v1a), ww), v1a);
        __m256i tmpb = _mm256_add_epi16(_mm256_mulhrs_epi16(_mm256_sub_epi16(v2b, v1b), ww), v1b);
        __m256i result = _mm256_packus_epi16(tmpa, tmpb);
        result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
        _mm256_storeu_si256((__m256i *)(dstp + i), result);
    }
}

static uint8_t merge_px(uint8_t v1, uint8_t v2, unsigned w) {
    int inc = ((int)v2 - (int)v1) * (int)w + (int)MROUND;
    inc >>= MERGESHIFT;
    int r = (int)v1 + inc;
    if (r < 0) r = 0;
    if (r > 255) r = 255;
    return (uint8_t)r;
}

static void test_merge_byte() {
    const unsigned ws[] = { 0, 1, 2, 16384, 32767, 100, 20000, 32766 };
    alignas(32) uint8_t a[64], b[64], oldo[64], newo[64];
    for (unsigned w : ws) {
        for (int v1 = 0; v1 < 256; v1++) {
            for (int v2 = 0; v2 < 256; v2++) {
                std::memset(a, (uint8_t)v1, 64);
                std::memset(b, (uint8_t)v2, 64);
                merge_byte_old(a, b, oldo, (uint16_t)w, 32);
                merge_byte_new(a, b, newo, (uint16_t)w, 32);
                uint8_t want = merge_px((uint8_t)v1, (uint8_t)v2, w);
                if (oldo[0] != newo[0] || newo[0] != want) {
                    std::fprintf(stderr, "merge v1=%d v2=%d w=%u old=%u new=%u want=%u\n",
                                 v1, v2, w, oldo[0], newo[0], want);
                    fail("merge_byte pmulhrsw");
                    return;
                }
            }
        }
        /* mixed row, widths 16 and 32 */
        for (int i = 0; i < 64; i++) {
            a[i] = (uint8_t)(i * 3);
            b[i] = (uint8_t)(255 - i * 5);
        }
        merge_byte_old(a, b, oldo, (uint16_t)w, 32);
        merge_byte_new(a, b, newo, (uint16_t)w, 32);
        if (std::memcmp(oldo, newo, 32) != 0) {
            fail("merge mixed row");
            return;
        }
    }
}

static __m256i div255_epu16(__m256i x) {
    x = _mm256_mulhi_epu16(x, _mm256_set1_epi16(0x8081));
    x = _mm256_srli_epi16(x, 7);
    return x;
}

static void mask_merge_old(const uint8_t *s1, const uint8_t *s2, const uint8_t *m, uint8_t *d, unsigned n) {
    for (unsigned i = 0; i < n; i += 16) {
        __m256i v1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(s1 + i)));
        __m256i v2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(s2 + i)));
        __m256i w2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(m + i)));
        __m256i w1 = _mm256_sub_epi16(_mm256_set1_epi16(UINT8_MAX), w2);
        __m256i tmp = _mm256_add_epi16(_mm256_add_epi16(_mm256_mullo_epi16(v1, w1), _mm256_mullo_epi16(v2, w2)), _mm256_set1_epi16(UINT8_MAX / 2));
        tmp = div255_epu16(tmp);
        tmp = _mm256_packus_epi16(tmp, tmp);
        tmp = _mm256_permute4x64_epi64(tmp, _MM_SHUFFLE(3, 1, 2, 0));
        _mm_storeu_si128((__m128i *)(d + i), _mm256_castsi256_si128(tmp));
    }
}

static void test_mask_merge() {
    alignas(32) uint8_t a[64], b[64], m[64], oldo[64], newo[64];
    for (int i = 0; i < 64; i++) {
        a[i] = (uint8_t)i;
        b[i] = (uint8_t)(255 - i);
        m[i] = (uint8_t)(i * 7);
    }
    mask_merge_old(a, b, m, oldo, 32);
    /* fix store in new: rewrite cleanly */
    for (unsigned i = 0; i < 32; i += 32) {
        __m256i v1a = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(a + i)));
        __m256i v1b = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(a + i + 16)));
        __m256i v2a = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(b + i)));
        __m256i v2b = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(b + i + 16)));
        __m256i w2a = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(m + i)));
        __m256i w2b = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(m + i + 16)));
        __m256i w1a = _mm256_sub_epi16(_mm256_set1_epi16(UINT8_MAX), w2a);
        __m256i w1b = _mm256_sub_epi16(_mm256_set1_epi16(UINT8_MAX), w2b);
        __m256i tmpa = _mm256_add_epi16(_mm256_add_epi16(_mm256_mullo_epi16(v1a, w1a), _mm256_mullo_epi16(v2a, w2a)), _mm256_set1_epi16(UINT8_MAX / 2));
        __m256i tmpb = _mm256_add_epi16(_mm256_add_epi16(_mm256_mullo_epi16(v1b, w1b), _mm256_mullo_epi16(v2b, w2b)), _mm256_set1_epi16(UINT8_MAX / 2));
        tmpa = div255_epu16(tmpa);
        tmpb = div255_epu16(tmpb);
        __m256i result = _mm256_packus_epi16(tmpa, tmpb);
        result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
        _mm256_storeu_si256((__m256i *)newo, result);
    }
    if (std::memcmp(oldo, newo, 32) != 0)
        fail("mask_merge 32 vs 16");
}

/* ---- PlaneStats 16-bit old psadbw vs pmaddwd ---- */
static unsigned hmin_epu16(__m256i x) {
    __m128i t = _mm_min_epu16(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    t = _mm_min_epu16(t, _mm_srli_si128(t, 8));
    t = _mm_min_epu16(t, _mm_srli_si128(t, 4));
    t = _mm_min_epu16(t, _mm_srli_si128(t, 2));
    return (unsigned)_mm_extract_epi16(t, 0);
}
static unsigned hmax_epu16(__m256i x) {
    __m128i t = _mm_max_epu16(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    t = _mm_max_epu16(t, _mm_srli_si128(t, 8));
    t = _mm_max_epu16(t, _mm_srli_si128(t, 4));
    t = _mm_max_epu16(t, _mm_srli_si128(t, 2));
    return (unsigned)_mm_extract_epi16(t, 0);
}

static const uint16_t ascend16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

static void stats_old(const uint16_t *src, ptrdiff_t stride, unsigned width, unsigned height,
                      unsigned *mn, unsigned *mx, uint64_t *acc) {
    const uint8_t *srcp = (const uint8_t *)src;
    unsigned tail = width & ~15u;
    __m256i mmin = _mm256_set1_epi16(-1);
    __m256i mmax = _mm256_setzero_si256();
    __m256i macc_lo = _mm256_setzero_si256();
    __m256i macc_hi = _mm256_setzero_si256();
    __m256i mask = _mm256_cmpgt_epi16(_mm256_set1_epi16(width % 16), _mm256_loadu_si256((const __m256i *)ascend16));
    __m256i onesmask = _mm256_andnot_si256(mask, _mm256_set1_epi16(-1));
    __m256i low8mask = _mm256_set1_epi16(0xFF);
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < tail; x += 16) {
            __m256i v = _mm256_loadu_si256((const __m256i *)((const uint16_t *)srcp + x));
            mmin = _mm256_min_epu16(mmin, v);
            mmax = _mm256_max_epu16(mmax, v);
            macc_lo = _mm256_add_epi64(macc_lo, _mm256_sad_epu8(_mm256_and_si256(low8mask, v), _mm256_setzero_si256()));
            macc_hi = _mm256_add_epi64(macc_hi, _mm256_sad_epu8(_mm256_andnot_si256(low8mask, v), _mm256_setzero_si256()));
        }
        if (width != tail) {
            __m256i v = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)((const uint16_t *)srcp + tail)), mask);
            mmin = _mm256_min_epu16(mmin, _mm256_or_si256(v, onesmask));
            mmax = _mm256_max_epu16(mmax, v);
            macc_lo = _mm256_add_epi64(macc_lo, _mm256_sad_epu8(_mm256_and_si256(low8mask, v), _mm256_setzero_si256()));
            macc_hi = _mm256_add_epi64(macc_hi, _mm256_sad_epu8(_mm256_andnot_si256(low8mask, v), _mm256_setzero_si256()));
        }
        srcp += stride;
    }
    *mn = hmin_epu16(mmin);
    *mx = hmax_epu16(mmax);
    __m256i tmp = _mm256_add_epi64(_mm256_unpacklo_epi64(macc_lo, macc_hi), _mm256_unpackhi_epi64(macc_lo, macc_hi));
    tmp = _mm256_add_epi64(tmp, _mm256_slli_epi64(_mm256_unpackhi_epi64(tmp, tmp), 8));
    _mm_storel_epi64((__m128i *)acc, _mm_add_epi64(_mm256_castsi256_si128(tmp), _mm256_extracti128_si256(tmp, 1)));
}

static void stats_new(const uint16_t *src, ptrdiff_t stride, unsigned width, unsigned height,
                     unsigned *mn, unsigned *mx, uint64_t *acc) {
    const uint8_t *srcp = (const uint8_t *)src;
    unsigned tail = width & ~15u;
    __m256i mmin = _mm256_set1_epi16(-1);
    __m256i mmax = _mm256_setzero_si256();
    __m256i macc = _mm256_setzero_si256();
    __m256i mbias = _mm256_set1_epi16((short)0x8000);
    __m256i ones = _mm256_set1_epi16(1);
    __m256i mask = _mm256_cmpgt_epi16(_mm256_set1_epi16(width % 16), _mm256_loadu_si256((const __m256i *)ascend16));
    __m256i onesmask = _mm256_andnot_si256(mask, _mm256_set1_epi16(-1));
    for (unsigned y = 0; y < height; y++) {
        __m256i racc = _mm256_setzero_si256();
        unsigned pending = 0;
        for (unsigned x = 0; x < tail; x += 16) {
            __m256i v = _mm256_loadu_si256((const __m256i *)((const uint16_t *)srcp + x));
            mmin = _mm256_min_epu16(mmin, v);
            mmax = _mm256_max_epu16(mmax, v);
            racc = _mm256_add_epi32(racc, _mm256_madd_epi16(_mm256_add_epi16(v, mbias), ones));
            if (++pending == 32768) {
                macc = _mm256_add_epi64(macc, _mm256_add_epi64(
                    _mm256_cvtepi32_epi64(_mm256_castsi256_si128(racc)),
                    _mm256_cvtepi32_epi64(_mm256_extracti128_si256(racc, 1))));
                racc = _mm256_setzero_si256();
                pending = 0;
            }
        }
        if (width != tail) {
            __m256i v = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)((const uint16_t *)srcp + tail)), mask);
            mmin = _mm256_min_epu16(mmin, _mm256_or_si256(v, onesmask));
            mmax = _mm256_max_epu16(mmax, v);
            racc = _mm256_add_epi32(racc, _mm256_madd_epi16(_mm256_and_si256(_mm256_add_epi16(v, mbias), mask), ones));
        }
        macc = _mm256_add_epi64(macc, _mm256_add_epi64(
            _mm256_cvtepi32_epi64(_mm256_castsi256_si128(racc)),
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(racc, 1))));
        srcp += stride;
    }
    *mn = hmin_epu16(mmin);
    *mx = hmax_epu16(mmax);
    __m128i t = _mm_add_epi64(_mm256_castsi256_si128(macc), _mm256_extracti128_si256(macc, 1));
    int64_t sum_biased;
    _mm_storel_epi64((__m128i *)&sum_biased, _mm_add_epi64(t, _mm_srli_si128(t, 8)));
    *acc = (uint64_t)(sum_biased + (int64_t)(((uint64_t)width * height) << 15));
}

static void test_planestats() {
    const unsigned widths[] = { 1, 8, 15, 16, 17, 31, 32, 33, 64 };
    for (unsigned w : widths) {
        unsigned stride_px = (w + 15) & ~15u;
        if (stride_px < 16) stride_px = 16;
        std::vector<uint16_t> buf(stride_px * 3, 0);
        for (unsigned y = 0; y < 3; y++)
            for (unsigned x = 0; x < w; x++)
                buf[y * stride_px + x] = (uint16_t)((x * 111 + y * 777) & 65535);
        /* unique min/max on last pixel */
        buf[w - 1] = 0;
        buf[stride_px + w - 1] = 65535;
        unsigned mn0, mx0, mn1, mx1;
        uint64_t a0, a1;
        stats_old(buf.data(), stride_px * 2, w, 3, &mn0, &mx0, &a0);
        stats_new(buf.data(), stride_px * 2, w, 3, &mn1, &mx1, &a1);
        uint64_t ref = 0;
        unsigned rmin = 65535, rmax = 0;
        for (unsigned y = 0; y < 3; y++)
            for (unsigned x = 0; x < w; x++) {
                uint16_t v = buf[y * stride_px + x];
                ref += v;
                if (v < rmin) rmin = v;
                if (v > rmax) rmax = v;
            }
        if (mn0 != mn1 || mx0 != mx1 || a0 != a1 || a1 != ref || mn1 != rmin || mx1 != rmax) {
            std::fprintf(stderr, "planestats w=%u old(%u,%u,%llu) new(%u,%u,%llu) ref(%u,%u,%llu)\n",
                         w, mn0, mx0, (unsigned long long)a0, mn1, mx1, (unsigned long long)a1,
                         rmin, rmax, (unsigned long long)ref);
            fail("planestats word");
            return;
        }
    }
}

int main() {
    test_fastdiv();
    test_boxblur();
    test_merge_byte();
    test_mask_merge();
    test_planestats();
    if (g_fails) {
        std::fprintf(stderr, "%d check(s) failed\n", g_fails);
        return 1;
    }
    std::puts("kernel_bitexact: all checks passed");
    return 0;
}
