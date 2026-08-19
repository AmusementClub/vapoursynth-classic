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


def gray8(w, h, expr, color=0):
    blank = vs.core.std.BlankClip(width=w, height=h, format=vs.GRAY8, length=1, color=color)
    return vs.core.std.Expr(blank, expr) if expr else blank


def yuv420(w, h, y_expr, u_expr='128', v_expr='128'):
    y = gray8(w, h, y_expr)
    u = gray8(w // 2, h // 2, u_expr, color=128)
    v = gray8(w // 2, h // 2, v_expr, color=128)
    return vs.core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV)


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
        exprs = [
            ('0', '255'),
            ('255', '0'),
            ('X', '255 X -'),
            ('X 2 % 0 = 0 255 ?', '128'),
            ('0', '0'),
            ('255', '255'),
        ]
        with_cpu('avx2')
        for w in widths:
            for we in weights:
                wu = merge_weight_u(we)
                for ea, eb in exprs:
                    a = gray8(w, 3, ea)
                    b = gray8(w, 3, eb)
                    out = self.core.std.Merge(a, b, we)
                    got = plane_rows(out.get_frame(0), 0)
                    srca = plane_rows(a.get_frame(0), 0)
                    srcb = plane_rows(b.get_frame(0), 0)
                    exp = [[merge_byte_px(srca[y][x], srcb[y][x], wu) for x in range(w)] for y in range(3)]
                    self.assertEqual(got, exp, 'w=%d weight=%s expr=%s' % (w, we, (ea, eb)))

    def test_merge8_yuv420_chroma_widths(self):
        with_cpu('avx2')
        clipa = yuv420(17, 8, 'X', 'X', '255 X -')
        clipb = yuv420(17, 8, '255 X -', '255', '0')
        out = self.core.std.Merge(clipa, clipb, 0.5)
        wu = merge_weight_u(0.5)
        fa, fb, fo = clipa.get_frame(0), clipb.get_frame(0), out.get_frame(0)
        for p in range(3):
            ra, rb, ro = plane_rows(fa, p), plane_rows(fb, p), plane_rows(fo, p)
            exp = [[merge_byte_px(ra[y][x], rb[y][x], wu) for x in range(len(ra[0]))] for y in range(len(ra))]
            self.assertEqual(ro, exp, 'plane %d' % p)

    def test_merge16_none_vs_avx2(self):
        def build():
            a = self.core.std.BlankClip(width=33, height=4, format=vs.GRAY16, length=1, color=1000)
            b = self.core.std.BlankClip(width=33, height=4, format=vs.GRAY16, length=1, color=50000)
            a = self.core.std.Expr(a, 'X 200 *')
            b = self.core.std.Expr(b, '65535 X 200 * -')
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
        masks = ['0', '255', '128', 'X', 'X 2 % 0 = 0 255 ?']
        for w in widths:
            for me in masks:
                def build(ww=w, mexpr=me):
                    a = gray8(ww, 4, '0')
                    b = gray8(ww, 4, '255')
                    m = gray8(ww, 4, mexpr)
                    return self.core.std.MaskedMerge(a, b, m)
                c, a = self._pair(build)
                assert_frames_equal(self, c, a, 'mm w=%d mask=%s' % (w, me))

    def test_maskedmerge8_first_plane_420(self):
        def build():
            a = yuv420(16, 8, '0', '16', '16')
            b = yuv420(16, 8, '255', '240', '240')
            m = yuv420(16, 8, 'X 16 *', '0', '0')
            return self.core.std.MaskedMerge(a, b, m, first_plane=True)
        c, a = self._pair(build)
        assert_frames_equal(self, c, a, 'mm first_plane 420')

    def test_maskedmerge8_premul_colorrange(self):
        for limited in (0, 1):
            def build(lim=limited):
                a = gray8(33, 3, 'X')
                b = gray8(33, 3, '255 X -')
                m = gray8(33, 3, '128')
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
        exprs = ['X', '255 X -', 'X 2 % 0 = 0 255 ?']
        with_cpu('avx2')
        for w, h, hr, hp, vr, vp in cases:
            for expr in exprs:
                src = gray8(w, h, expr)
                out = self.core.std.BoxBlur(src, hradius=hr, hpasses=hp, vradius=vr, vpasses=vp)
                got = plane_rows(out.get_frame(0), 0)
                exp = boxblur_ref(plane_rows(src.get_frame(0), 0), hr, hp, vr, vp)
                self.assertEqual(got, exp, 'boxblur %s expr=%s' % ((w, h, hr, hp, vr, vp), expr))

    def test_boxblur_16bit_and_10bit(self):
        with_cpu('avx2')
        for fmt, peak in ((vs.GRAY16, 65535), (vs.GRAY10, 1023)):
            src = self.core.std.BlankClip(width=17, height=5, format=fmt, length=1, color=0)
            src = self.core.std.Expr(src, 'X Y + 3 *')
            out = self.core.std.BoxBlur(src, hradius=3, hpasses=2, vradius=2, vpasses=1)
            got = plane_rows(out.get_frame(0), 0)
            exp = boxblur_ref(plane_rows(src.get_frame(0), 0), 3, 2, 2, 1)
            self.assertEqual(got, exp, str(fmt))

    def test_boxblur_yuv420_chroma_and_luma_only(self):
        with_cpu('avx2')
        src = yuv420(16, 8, 'X Y +', 'X', 'Y')
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
        src = self.core.std.BlankClip(width=17, height=3, format=vs.GRAYS, length=1, color=0)
        src = self.core.std.Expr(src, 'X 16 / Y 2 / +')
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
        src = yuv420(8, 2, 'Y 200 *', '128', '64')
        out = self.core.std.BoxBlur(src, hradius=0, hpasses=0, vradius=1, vpasses=2)
        fsrc, fout = src.get_frame(0), out.get_frame(0)
        for p in range(3):
            exp = boxblur_ref(plane_rows(fsrc, p), 0, 0, 1, 2)
            self.assertEqual(plane_rows(fout, p), exp, 'vblur 420 plane %d' % p)

    # --- PlaneStats ---

    def _stats_ident(self, fmt, w, h, expr_a, expr_b=None, bits=None):
        def build():
            a = self.core.std.BlankClip(width=w, height=h, format=fmt, length=1, color=0)
            a = self.core.std.Expr(a, expr_a)
            if expr_b is None:
                return self.core.std.PlaneStats(a)
            b = self.core.std.BlankClip(width=w, height=h, format=fmt, length=1, color=0)
            b = self.core.std.Expr(b, expr_b)
            return self.core.std.PlaneStats(a, b)
        c, a = self._pair(build)
        fc, fa = c.get_frame(0), a.get_frame(0)
        for key in ('PlaneStatsMin', 'PlaneStatsMax'):
            self.assertEqual(fc.props[key], fa.props[key], '%s w=%d h=%d %s' % (key, w, h, expr_a))
        self.assertAlmostEqual(float(fc.props['PlaneStatsAverage']), float(fa.props['PlaneStatsAverage']),
                               places=12, msg='Average w=%d %s' % (w, expr_a))
        if expr_b is not None:
            self.assertAlmostEqual(float(fc.props['PlaneStatsDiff']), float(fa.props['PlaneStatsDiff']),
                                   places=12, msg='Diff w=%d' % w)

    def test_planestats16_none_vs_avx2(self):
        widths = [1, 8, 15, 16, 17, 31, 32]
        exprs = ['0', '65535', '32768', 'X', 'X 16 % 0 = 0 65535 ?']
        for w in widths:
            for e in exprs:
                self._stats_ident(vs.GRAY16, w, 3, e)
                self._stats_ident(vs.GRAY16, w, 1, e)

    def test_planestats16_tail_unique_minmax(self):
        # leftover width%16==1; last pixel is unique min or max
        self._stats_ident(vs.GRAY16, 17, 2, 'X 16 = 0 40000 ?')
        self._stats_ident(vs.GRAY16, 17, 2, 'X 16 = 65535 1000 ?')

    def test_planestats16_two_clip(self):
        self._stats_ident(vs.GRAY16, 17, 3, 'X 100 *', 'X 100 *')
        self._stats_ident(vs.GRAY16, 17, 3, 'X 100 *', '65535 X 100 * -')
        self._stats_ident(vs.GRAY16, 17, 2, '1000', 'X 16 = 0 1000 ?')

    def test_planestats_sub16_formats(self):
        for fmt, expr in ((vs.GRAY10, 'X 4 *'), (vs.GRAY12, 'X 8 *')):
            self._stats_ident(fmt, 17, 2, expr)

    def test_planestats_yuv420_chroma(self):
        def build():
            y = self.core.std.BlankClip(width=16, height=8, format=vs.GRAY16, length=1, color=0)
            u = self.core.std.Expr(self.core.std.BlankClip(width=8, height=4, format=vs.GRAY16, length=1), 'X 2000 *')
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
