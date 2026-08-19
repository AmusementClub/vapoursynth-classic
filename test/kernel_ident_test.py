import random
import struct
import unittest

import vapoursynth as vs


MERGESHIFT = 15
ROUND = 1 << (MERGESHIFT - 1)


def plane_rows(frame, plane=0):
    arr = frame[plane]
    h, w = arr.shape
    return [[int(arr[y, x]) for x in range(w)] for y in range(h)]


def plane_rows_f(frame, plane=0):
    arr = frame[plane]
    h, w = arr.shape
    return [[float(arr[y, x]) for x in range(w)] for y in range(h)]


def f32(x):
    return struct.unpack('f', struct.pack('f', float(x)))[0]


def merge_weight_u(weight):
    return min(int(weight * (1 << MERGESHIFT) + 0.5), (1 << MERGESHIFT) - 1)


def merge_byte_px(v1, v2, w):
    inc = ((int(v2) - int(v1)) * w + ROUND) >> MERGESHIFT
    return max(0, min(255, int(v1) + inc))


def assert_frames_equal(test, a, b, msg=''):
    fa, fb = a.get_frame(0), b.get_frame(0)
    test.assertEqual(fa.format.id, fb.format.id, msg)
    test.assertEqual(fa.width, fb.width, msg)
    test.assertEqual(fa.height, fb.height, msg)
    for p in range(fa.format.num_planes):
        ra, rb = plane_rows(fa, p), plane_rows(fb, p)
        if ra != rb:
            for y, (rowa, rowb) in enumerate(zip(ra, rb)):
                if rowa != rowb:
                    for x, (pa, pb) in enumerate(zip(rowa, rowb)):
                        if pa != pb:
                            test.fail('%s plane %d mismatch at (%d,%d): %s vs %s' % (msg, p, x, y, pa, pb))
            test.fail('%s plane %d differs' % (msg, p))


def assert_frames_equal_f32(test, a, b, msg=''):
    fa, fb = a.get_frame(0), b.get_frame(0)
    for p in range(fa.format.num_planes):
        ra, rb = plane_rows_f(fa, p), plane_rows_f(fb, p)
        for y, (rowa, rowb) in enumerate(zip(ra, rb)):
            for x, (pa, pb) in enumerate(zip(rowa, rowb)):
                if struct.pack('f', pa) != struct.pack('f', pb):
                    test.fail('%s plane %d mismatch at (%d,%d): %r vs %r' % (msg, p, x, y, pa, pb))


def with_cpu(level):
    vs.core.std.SetMaxCPU(level)


def fill_gray(w, h, fmt, fn, color=0):
    """Classic Expr has no X/Y coordinates; paint pixels with ModifyFrame."""
    clip = vs.core.std.BlankClip(width=w, height=h, format=fmt, length=1, color=color)

    def sel(n, f):
        fout = f.copy()
        arr = fout[0]
        hh, ww = arr.shape
        for y in range(hh):
            for x in range(ww):
                arr[y, x] = fn(x, y)
        return fout

    return clip.std.ModifyFrame(clip, sel)


def gray8(w, h, fn, color=0):
    if fn is None:
        return vs.core.std.BlankClip(width=w, height=h, format=vs.GRAY8, length=1, color=color)
    return fill_gray(w, h, vs.GRAY8, fn, color)


def yuv420(w, h, yfn, ufn=None, vfn=None):
    y = gray8(w, h, yfn)
    u = gray8(w // 2, h // 2, ufn, color=128)
    v = gray8(w // 2, h // 2, vfn, color=128)
    return vs.core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV)


def const8(v):
    return lambda x, y, _v=v: _v


def ramp_x(scale=1, offset=0, peak=255):
    return lambda x, y, _s=scale, _o=offset, _p=peak: max(0, min(_p, x * _s + _o))


def inv_ramp_x(peak=255):
    return lambda x, y, _p=peak: max(0, min(_p, _p - x))


def checker(a=0, b=255):
    return lambda x, y, _a=a, _b=b: _a if (x % 2) == 0 else _b


def blur_h_int(src, radius, round_val, div):
    w = len(src)
    dst = [0] * w
    acc = radius * src[0]
    for x in range(radius):
        acc += src[min(x, w - 1)]
    for x in range(min(radius, w)):
        acc += src[min(x + radius, w - 1)]
        dst[x] = (acc + round_val) // div
        acc -= src[max(x - radius, 0)]
    if w > radius:
        for x in range(radius, w - radius):
            acc += src[x + radius]
            dst[x] = (acc + round_val) // div
            acc -= src[x - radius]
        for x in range(max(w - radius, radius), w):
            acc += src[min(x + radius, w - 1)]
            dst[x] = (acc + round_val) // div
            acc -= src[max(x - radius, 0)]
    return dst


def blur_h_f32(src, radius):
    w = len(src)
    div = f32(1.0 / (radius * 2 + 1))
    dst = [0.0] * w
    acc = f32(radius * src[0])
    for x in range(radius):
        acc = f32(acc + src[min(x, w - 1)])
    for x in range(min(radius, w)):
        acc = f32(acc + src[min(x + radius, w - 1)])
        dst[x] = f32(acc * div)
        acc = f32(acc - src[max(x - radius, 0)])
    if w > radius:
        for x in range(radius, w - radius):
            acc = f32(acc + src[x + radius])
            dst[x] = f32(acc * div)
            acc = f32(acc - src[x - radius])
        for x in range(max(w - radius, radius), w):
            acc = f32(acc + src[min(x + radius, w - 1)])
            dst[x] = f32(acc * div)
            acc = f32(acc - src[max(x - radius, 0)])
    return dst


def boxblur_plane(rows, radius, passes):
    div = radius * 2 + 1
    round0 = div - 1
    out = [list(r) for r in rows]
    for p in range(passes):
        rnd = round0 if (p % 2 == 0) else 0
        out = [blur_h_int(r, radius, rnd, div) for r in out]
    return out


def transpose(rows):
    if not rows:
        return []
    return [[rows[y][x] for y in range(len(rows))] for x in range(len(rows[0]))]


def boxblur_ref(rows, hradius, hpasses, vradius, vpasses):
    out = [list(r) for r in rows]
    if hradius > 0 and hpasses > 0:
        out = boxblur_plane(out, hradius, hpasses)
    if vradius > 0 and vpasses > 0:
        out = transpose(boxblur_plane(transpose(out), vradius, vpasses))
    return out


class KernelIdentTests(unittest.TestCase):
    def setUp(self):
        self.core = vs.core
        self.core.num_threads = 1

    def tearDown(self):
        with_cpu('avx2')

    def _pair(self, build):
        with_cpu('none')
        c = build()
        with_cpu('avx2')
        a = build()
        return c, a

    # --- Merge ---

    def test_merge8_avx2_matches_signed_lerp(self):
        widths = [1, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 1920]
        weights = [0.0, 1.0, 0.5, 1.0 / 32768, 32767.0 / 32768] + [random.random() for _ in range(4)]
        pairs = [
            (const8(0), const8(255), '0/255'),
            (const8(255), const8(0), '255/0'),
            (ramp_x(), inv_ramp_x(), 'ramp'),
            (checker(), const8(128), 'checker'),
            (const8(0), const8(0), '0/0'),
            (const8(255), const8(255), '255/255'),
        ]
        with_cpu('avx2')
        for w in widths:
            for we in weights:
                wu = merge_weight_u(we)
                for fa, fb, name in pairs:
                    a = gray8(w, 3, fa)
                    b = gray8(w, 3, fb)
                    out = self.core.std.Merge(a, b, we)
                    got = plane_rows(out.get_frame(0), 0)
                    srca = plane_rows(a.get_frame(0), 0)
                    srcb = plane_rows(b.get_frame(0), 0)
                    exp = [[merge_byte_px(srca[y][x], srcb[y][x], wu) for x in range(w)] for y in range(3)]
                    self.assertEqual(got, exp, 'w=%d weight=%s %s' % (w, we, name))

    def test_merge8_yuv420_chroma_widths(self):
        with_cpu('avx2')
        clipa = yuv420(17, 8, ramp_x(), ramp_x(), inv_ramp_x())
        clipb = yuv420(17, 8, inv_ramp_x(), const8(255), const8(0))
        out = self.core.std.Merge(clipa, clipb, 0.5)
        wu = merge_weight_u(0.5)
        fa, fb, fo = clipa.get_frame(0), clipb.get_frame(0), out.get_frame(0)
        for p in range(3):
            ra, rb, ro = plane_rows(fa, p), plane_rows(fb, p), plane_rows(fo, p)
            exp = [[merge_byte_px(ra[y][x], rb[y][x], wu) for x in range(len(ra[0]))] for y in range(len(ra))]
            self.assertEqual(ro, exp, 'plane %d' % p)

    def test_merge16_none_vs_avx2(self):
        def build():
            a = fill_gray(33, 4, vs.GRAY16, lambda x, y: min(65535, x * 200))
            b = fill_gray(33, 4, vs.GRAY16, lambda x, y: max(0, 65535 - x * 200))
            return self.core.std.Merge(a, b, 0.37)
        c, a = self._pair(build)
        assert_frames_equal(self, c, a, 'merge16')

    def test_merge_float_none_vs_avx2(self):
        def build():
            a = self.core.std.BlankClip(width=32, height=2, format=vs.GRAYS, length=1, color=0.25)
            b = self.core.std.BlankClip(width=32, height=2, format=vs.GRAYS, length=1, color=0.75)
            return self.core.std.Merge(a, b, 0.3)
        c, a = self._pair(build)
        assert_frames_equal_f32(self, c, a, 'merge_float')

    # --- MaskedMerge / premul ---

    def test_maskedmerge8_none_vs_avx2(self):
        widths = [1, 8, 15, 16, 17, 31, 32, 33, 64]
        masks = [
            (const8(0), '0'),
            (const8(255), '255'),
            (const8(128), '128'),
            (ramp_x(), 'ramp'),
            (checker(), 'checker'),
        ]
        for w in widths:
            for mfn, name in masks:
                def build(ww=w, fn=mfn):
                    a = gray8(ww, 4, const8(0))
                    b = gray8(ww, 4, const8(255))
                    m = gray8(ww, 4, fn)
                    return self.core.std.MaskedMerge(a, b, m)
                c, a = self._pair(build)
                assert_frames_equal(self, c, a, 'mm w=%d mask=%s' % (w, name))

    def test_maskedmerge8_first_plane_420(self):
        def build():
            a = yuv420(16, 8, const8(0), const8(16), const8(16))
            b = yuv420(16, 8, const8(255), const8(240), const8(240))
            m = yuv420(16, 8, lambda x, y: min(255, x * 16), const8(0), const8(0))
            return self.core.std.MaskedMerge(a, b, m, first_plane=True)
        c, a = self._pair(build)
        assert_frames_equal(self, c, a, 'mm first_plane 420')

    def test_maskedmerge8_premul_colorrange(self):
        for limited in (0, 1):
            def build(lim=limited):
                a = gray8(33, 3, ramp_x())
                b = gray8(33, 3, inv_ramp_x())
                m = gray8(33, 3, const8(128))
                a = a.std.SetFrameProps(_ColorRange=lim)
                b = b.std.SetFrameProps(_ColorRange=lim)
                return self.core.std.MaskedMerge(a, b, m, premultiplied=True)
            c, a = self._pair(build)
            assert_frames_equal(self, c, a, 'premul limited=%d' % limited)

    # --- BoxBlur ---

    def test_fastdiv_matches_hw_div(self):
        # Exercise the same magic used in C via the filter: integer BoxBlur
        # of a constant plane is (value) after any number of passes.
        with_cpu('avx2')
        for bits, fmt, val in ((8, vs.GRAY8, 40), (16, vs.GRAY16, 4000)):
            clip = self.core.std.BlankClip(width=8, height=2, format=fmt, length=1, color=val)
            out = self.core.std.BoxBlur(clip, hradius=3, hpasses=2, vradius=0, vpasses=0)
            got = plane_rows(out.get_frame(0), 0)
            self.assertTrue(all(p == val for row in got for p in row), 'constant %d-bit' % bits)

    def test_boxblur_int_edges(self):
        cases = [
            (5, 4, 8, 1, 0, 0),   # width < radius
            (8, 3, 8, 1, 0, 0),   # width == radius
            (16, 3, 8, 2, 0, 0),  # width == 2*radius
            (17, 3, 8, 2, 0, 0),  # width == 2*radius+1
            (9, 3, 8, 2, 0, 0),   # width == radius+1
            (32, 8, 7, 1, 0, 0),
            (32, 8, 7, 2, 0, 0),
            (32, 8, 7, 3, 0, 0),
            (15, 1, 1, 2, 0, 0),  # radius==1 fast path
            (16, 2, 2, 1, 2, 1),
            (8, 1, 3, 0, 3, 1),   # height==1 vertical via transpose
            (8, 2, 0, 0, 3, 2),
        ]
        patterns = [
            (ramp_x(), 'ramp'),
            (inv_ramp_x(), 'inv'),
            (checker(), 'checker'),
        ]
        with_cpu('avx2')
        for w, h, hr, hp, vr, vp in cases:
            for fn, name in patterns:
                src = gray8(w, h, fn)
                out = self.core.std.BoxBlur(src, hradius=hr, hpasses=hp, vradius=vr, vpasses=vp)
                got = plane_rows(out.get_frame(0), 0)
                exp = boxblur_ref(plane_rows(src.get_frame(0), 0), hr, hp, vr, vp)
                self.assertEqual(got, exp, 'boxblur %s %s' % ((w, h, hr, hp, vr, vp), name))

    def test_boxblur_16bit_and_10bit(self):
        with_cpu('avx2')
        for fmt, peak in ((vs.GRAY16, 65535), (vs.GRAY10, 1023)):
            src = fill_gray(17, 5, fmt, lambda x, y, p=peak: min(p, (x + y) * 3))
            out = self.core.std.BoxBlur(src, hradius=3, hpasses=2, vradius=2, vpasses=1)
            got = plane_rows(out.get_frame(0), 0)
            exp = boxblur_ref(plane_rows(src.get_frame(0), 0), 3, 2, 2, 1)
            self.assertEqual(got, exp, str(fmt))

    def test_boxblur_yuv420_chroma_and_luma_only(self):
        with_cpu('avx2')
        src = yuv420(16, 8, lambda x, y: min(255, x + y), ramp_x(), lambda x, y: y)
        out = self.core.std.BoxBlur(src, hradius=2, hpasses=2, vradius=2, vpasses=1)
        fsrc, fout = src.get_frame(0), out.get_frame(0)
        for p in range(3):
            exp = boxblur_ref(plane_rows(fsrc, p), 2, 2, 2, 1)
            self.assertEqual(plane_rows(fout, p), exp, '420 plane %d' % p)
        luma = self.core.std.BoxBlur(src, planes=[0], hradius=2, hpasses=1, vradius=0, vpasses=0)
        fl = luma.get_frame(0)
        self.assertEqual(plane_rows(fl, 0), boxblur_ref(plane_rows(fsrc, 0), 2, 1, 0, 0))
        self.assertEqual(plane_rows(fl, 1), plane_rows(fsrc, 1))
        self.assertEqual(plane_rows(fl, 2), plane_rows(fsrc, 2))

    def test_boxblur_float_ring(self):
        with_cpu('avx2')
        src = fill_gray(17, 3, vs.GRAYS, lambda x, y: x / 16.0 + y / 2.0)
        out = self.core.std.BoxBlur(src, hradius=3, hpasses=2, vradius=0, vpasses=0)
        got = plane_rows_f(out.get_frame(0), 0)
        rows = plane_rows_f(src.get_frame(0), 0)
        exp = [r for r in rows]
        for p in range(2):
            exp = [blur_h_f32(r, 3) for r in exp]
        for y in range(len(got)):
            for x in range(len(got[0])):
                self.assertEqual(struct.pack('f', got[y][x]), struct.pack('f', exp[y][x]),
                                 'float (%d,%d)' % (x, y))

    def test_boxblur_2px_tall_420_vertical(self):
        with_cpu('avx2')
        src = yuv420(8, 2, lambda x, y: min(255, y * 200), const8(128), const8(64))
        out = self.core.std.BoxBlur(src, hradius=0, hpasses=0, vradius=1, vpasses=2)
        fsrc, fout = src.get_frame(0), out.get_frame(0)
        for p in range(3):
            exp = boxblur_ref(plane_rows(fsrc, p), 0, 0, 1, 2)
            self.assertEqual(plane_rows(fout, p), exp, 'vblur 420 plane %d' % p)

    # --- PlaneStats ---

    def _stats_ident(self, fmt, w, h, fn_a, fn_b=None, label=''):
        def build():
            a = fill_gray(w, h, fmt, fn_a)
            if fn_b is None:
                return self.core.std.PlaneStats(a)
            b = fill_gray(w, h, fmt, fn_b)
            return self.core.std.PlaneStats(a, b)
        c, a = self._pair(build)
        fc, fa = c.get_frame(0), a.get_frame(0)
        for key in ('PlaneStatsMin', 'PlaneStatsMax'):
            self.assertEqual(fc.props[key], fa.props[key], '%s w=%d h=%d %s' % (key, w, h, label))
        self.assertAlmostEqual(float(fc.props['PlaneStatsAverage']), float(fa.props['PlaneStatsAverage']),
                               places=12, msg='Average w=%d %s' % (w, label))
        if fn_b is not None:
            self.assertAlmostEqual(float(fc.props['PlaneStatsDiff']), float(fa.props['PlaneStatsDiff']),
                                   places=12, msg='Diff w=%d' % w)

    def test_planestats16_none_vs_avx2(self):
        widths = [1, 8, 15, 16, 17, 31, 32]
        patterns = [
            (lambda x, y: 0, '0'),
            (lambda x, y: 65535, 'max'),
            (lambda x, y: 32768, 'mid'),
            (ramp_x(1, 0, 65535), 'ramp'),
            (lambda x, y: 0 if (x % 16) == 0 else 65535, 'mod16'),
        ]
        for w in widths:
            for fn, name in patterns:
                self._stats_ident(vs.GRAY16, w, 3, fn, label=name)
                self._stats_ident(vs.GRAY16, w, 1, fn, label=name)

    def test_planestats16_tail_unique_minmax(self):
        # leftover width%16==1; last pixel is unique min or max
        self._stats_ident(vs.GRAY16, 17, 2, lambda x, y: 0 if x == 16 else 40000, label='tail-min')
        self._stats_ident(vs.GRAY16, 17, 2, lambda x, y: 65535 if x == 16 else 1000, label='tail-max')

    def test_planestats16_two_clip(self):
        self._stats_ident(vs.GRAY16, 17, 3, lambda x, y: min(65535, x * 100),
                          lambda x, y: min(65535, x * 100), label='same')
        self._stats_ident(vs.GRAY16, 17, 3, lambda x, y: min(65535, x * 100),
                          lambda x, y: max(0, 65535 - x * 100), label='inv')
        self._stats_ident(vs.GRAY16, 17, 2, lambda x, y: 1000,
                          lambda x, y: 0 if x == 16 else 1000, label='last0')

    def test_planestats_sub16_formats(self):
        self._stats_ident(vs.GRAY10, 17, 2, lambda x, y: min(1023, x * 4), label='p10')
        self._stats_ident(vs.GRAY12, 17, 2, lambda x, y: min(4095, x * 8), label='p12')

    def test_planestats_yuv420_chroma(self):
        def build():
            y = self.core.std.BlankClip(width=16, height=8, format=vs.GRAY16, length=1, color=0)
            u = fill_gray(8, 4, vs.GRAY16, lambda x, y: min(65535, x * 2000))
            v = self.core.std.BlankClip(width=8, height=4, format=vs.GRAY16, length=1, color=32768)
            clip = self.core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV)
            return self.core.std.PlaneStats(clip, plane=1)
        c, a = self._pair(build)
        fc, fa = c.get_frame(0), a.get_frame(0)
        self.assertEqual(fc.props['PlaneStatsMin'], fa.props['PlaneStatsMin'])
        self.assertEqual(fc.props['PlaneStatsMax'], fa.props['PlaneStatsMax'])
        self.assertAlmostEqual(float(fc.props['PlaneStatsAverage']), float(fa.props['PlaneStatsAverage']), places=12)


if __name__ == '__main__':
    unittest.main()
