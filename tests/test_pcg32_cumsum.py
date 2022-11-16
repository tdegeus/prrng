import time
import unittest

import numpy as np
import prrng

seed = int(time.time())


class Test_pcg32_cumum(unittest.TestCase):
    def test_delta(self):
        """
        Chunked storage: delta
        """

        scale = 5
        offset = 0.1
        ref = prrng.pcg32(seed)
        state = ref.state()
        x0 = ref.random([])
        ref = prrng.pcg32(seed)
        self.assertEqual(state, ref.state())
        xref = np.cumsum(offset + ref.delta([10000], scale)) + x0
        self.assertEqual(state, ref.state())

        n = 100
        chunk = prrng.pcg32_cumsum([n], seed, distribution=prrng.delta, parameters=[scale, offset])
        chunk += x0
        self.assertEqual(chunk.start, 0)
        self.assertTrue(np.allclose(chunk.data, xref[0:n]))
        self.assertEqual(state, chunk.state(chunk.start))

        for i in range(1, 10):
            chunk.next()
            self.assertEqual(chunk.start, i * n)
            self.assertTrue(np.allclose(chunk.data, xref[i * n : (i + 1) * n]))
            self.assertEqual(state, chunk.state(chunk.start))

        index = chunk.start
        state = chunk.state(index)
        value = chunk.data[0]
        ref = np.copy(chunk.data)

        for i in range(i - 1, 5, -1):
            chunk.prev()
            self.assertEqual(chunk.start, i * n)
            self.assertTrue(np.allclose(chunk.data, xref[i * n : (i + 1) * n]))
            self.assertEqual(state, chunk.state(chunk.start))

        i = n * 15 + 2
        target = 0.5 * (xref[i] + xref[i + 1])
        chunk.align(target)
        self.assertEqual(chunk.start + chunk.index, i)
        self.assertAlmostEqual(chunk.data[0], xref[i])
        self.assertAlmostEqual(chunk.data[1], xref[i + 1])

        chunk.restore(state, value, index)
        self.assertTrue(np.allclose(ref, chunk.data))
        self.assertEqual(state, chunk.state(chunk.start))

    def test_random(self):
        """
        Chunked storage: random
        """

        scale = 5
        offset = 0.1
        ref = prrng.pcg32(seed)
        xref = np.cumsum(offset + scale * ref.random([10000]))

        n = 100
        chunk = prrng.pcg32_cumsum([n], seed, distribution=prrng.random, parameters=[scale, offset])
        self.assertEqual(chunk.start, 0)
        self.assertTrue(np.allclose(chunk.data, xref[0:n]))

        for i in range(1, 10):
            chunk.next()
            self.assertEqual(chunk.start, i * n)
            self.assertTrue(np.allclose(chunk.data, xref[i * n : (i + 1) * n]))

        index = chunk.start
        state = chunk.state(index)
        value = chunk.data[0]
        ref = np.copy(chunk.data)

        for i in range(i - 1, 5, -1):
            chunk.prev()
            self.assertEqual(chunk.start, i * n)
            self.assertTrue(np.allclose(chunk.data, xref[i * n : (i + 1) * n]))

        i = n * 15 + 2
        target = 0.5 * (xref[i] + xref[i + 1])
        chunk.align(target)
        self.assertEqual(chunk.start + chunk.index, i)
        self.assertAlmostEqual(chunk.data[0], xref[i])
        self.assertAlmostEqual(chunk.data[1], xref[i + 1])

        chunk.restore(state, value, index)
        self.assertTrue(np.allclose(ref, chunk.data))

    def test_weibull(self):
        """
        Chunked storage: Weibull
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32(seed)
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        n = 100
        chunk = prrng.pcg32_cumsum(
            [n], seed, distribution=prrng.weibull, parameters=[k, scale, offset]
        )
        self.assertEqual(chunk.start, 0)
        self.assertTrue(np.allclose(chunk.data, xref[0:n]))

        for i in range(1, 10):
            chunk.next()
            self.assertEqual(chunk.start, i * n)
            self.assertTrue(np.allclose(chunk.data, xref[i * n : (i + 1) * n]))

        index = chunk.start
        state = chunk.state(index)
        value = chunk.data[0]
        ref = np.copy(chunk.data)

        for i in range(i - 1, 5, -1):
            chunk.prev()
            self.assertEqual(chunk.start, i * n)
            self.assertTrue(np.allclose(chunk.data, xref[i * n : (i + 1) * n]))

        i = n * 15 + 2
        target = 0.5 * (xref[i] + xref[i + 1])
        chunk.align(target)
        self.assertEqual(chunk.start + chunk.index, i)
        self.assertAlmostEqual(chunk.data[0], xref[i])
        self.assertAlmostEqual(chunk.data[1], xref[i + 1])

        chunk.restore(state, value, index)
        self.assertTrue(np.allclose(ref, chunk.data))

    def test_external_weibull(self):
        """
        Chunked storage: Weibull
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32(seed)
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        n = 100
        chunk = prrng.pcg32_cumsum([n], seed)

        def mydraw(n):
            return chunk.generator.weibull([n], k, scale) + offset

        def mycumsum(n):
            return chunk.generator.cumsum_weibull(n, k, scale) + n * offset

        chunk.set_functions(mydraw, mycumsum)
        self.assertEqual(chunk.start, 0)
        self.assertTrue(np.allclose(chunk.data, xref[0:n]))

        for i in range(1, 10):
            chunk.next()
            self.assertEqual(chunk.start, i * n)
            self.assertTrue(np.allclose(chunk.data, xref[i * n : (i + 1) * n]))

        index = chunk.start
        state = chunk.state(index)
        value = chunk.data[0]
        ref = np.copy(chunk.data)

        for i in range(i - 1, 5, -1):
            chunk.prev()
            self.assertEqual(chunk.start, i * n)
            self.assertTrue(np.allclose(chunk.data, xref[i * n : (i + 1) * n]))

        chunk.restore(state, value, index)
        self.assertTrue(np.allclose(ref, chunk.data))

    def test_array_delta(self):
        """
        Array: delta.
        """

        scale = 5
        offset = 0.1
        N = 6
        initstate = np.arange(N, dtype=np.uint64)
        seq = np.zeros_like(initstate)
        ref = prrng.pcg32_array(initstate, seq)
        state = ref.state()
        x0 = ref.random([]).reshape(-1, 1)
        ref = prrng.pcg32_array(initstate, seq)
        self.assertTrue(np.all(np.equal(state, ref.state())))
        xref = np.cumsum(offset + scale * ref.delta([10000]), axis=-1) + x0
        self.assertTrue(np.all(np.equal(state, ref.state())))

        n = 100
        margin = 10
        align = prrng.alignment(margin=margin, strict=True)
        chunk = prrng.pcg32_array_cumsum([n], initstate, seq, prrng.delta, [scale, offset], align)
        chunk += x0
        self.assertTrue(np.allclose(xref[..., :n], chunk.data))
        self.assertTrue(np.allclose(xref[np.arange(N), chunk.start], chunk.data[..., 0]))

        for i in [500, 2012, 101]:
            target = 0.5 * (xref[..., i] + xref[..., i + 1])
            chunk.align(target)
            self.assertTrue(np.all(chunk.start + chunk.index == i))
            self.assertTrue(np.allclose(xref[np.arange(N), chunk.start], chunk.data[..., 0]))
            self.assertTrue(np.all(chunk.data[..., margin] <= target))
            self.assertTrue(np.all(chunk.data[..., margin + 1] > target))

        index = chunk.start
        value = np.copy(chunk.data[..., 0])
        state = chunk.state(index)
        cp = np.copy(chunk.data)

        i = 3000
        target = 0.5 * (xref[..., i] + xref[..., i + 1])
        chunk.align(target)
        self.assertTrue(np.all(chunk.start + chunk.index == i))

        chunk.restore(state, value, index)
        self.assertTrue(np.allclose(cp, chunk.data))

    def test_array_random(self):
        """
        Array: random.
        """

        scale = 5
        offset = 0.1
        N = 6
        initstate = np.arange(N, dtype=np.uint64)
        seq = np.zeros_like(initstate)
        ref = prrng.pcg32_array(initstate, seq)
        x0 = ref.random([]).reshape(-1, 1)
        ref = prrng.pcg32_array(initstate, seq)
        xref = np.cumsum(offset + scale * ref.random([10000]), axis=-1) + x0

        n = 100
        margin = 10
        align = prrng.alignment(margin=margin, strict=True)
        chunk = prrng.pcg32_array_cumsum([n], initstate, seq, prrng.random, [scale, offset], align)
        chunk += x0
        self.assertTrue(np.allclose(xref[..., :n], chunk.data))
        self.assertTrue(np.allclose(xref[np.arange(N), chunk.start], chunk.data[..., 0]))

        for i in [500, 2012, 101]:
            target = 0.5 * (xref[..., i] + xref[..., i + 1])
            chunk.align(target)
            self.assertTrue(np.all(chunk.start + chunk.index == i))
            self.assertTrue(np.allclose(xref[np.arange(N), chunk.start], chunk.data[..., 0]))
            self.assertTrue(np.all(chunk.data[..., margin] <= target))
            self.assertTrue(np.all(chunk.data[..., margin + 1] > target))

        index = chunk.start
        value = np.copy(chunk.data[..., 0])
        state = chunk.state(index)
        cp = np.copy(chunk.data)

        i = 3000
        target = 0.5 * (xref[..., i] + xref[..., i + 1])
        chunk.align(target)
        self.assertTrue(np.all(chunk.start + chunk.index == i))

        chunk.restore(state, value, index)
        self.assertTrue(np.allclose(cp, chunk.data))

    def test_array_weibull(self):
        """
        Array: Weibull.
        """

        k = 2
        scale = 5
        offset = 0.1
        N = 6
        state = np.arange(N, dtype=np.uint64)
        seq = np.zeros_like(state)
        ref = prrng.pcg32_array(state, seq)
        x0 = ref.random([]).reshape(-1, 1)
        ref = prrng.pcg32_array(state, seq)
        xref = np.cumsum(offset + ref.weibull([10000], k, scale), axis=-1) + x0

        n = 100
        margin = 10
        align = prrng.alignment(margin=margin, strict=True)
        chunk = prrng.pcg32_array_cumsum([n], state, seq, prrng.weibull, [k, scale, offset], align)
        chunk += x0
        self.assertTrue(np.allclose(xref[..., :n], chunk.data))
        self.assertTrue(np.allclose(xref[np.arange(N), chunk.start], chunk.data[..., 0]))

        for i in [500, 2012, 101]:
            target = 0.5 * (xref[..., i] + xref[..., i + 1])
            chunk.align(target)
            self.assertTrue(np.all(chunk.start + chunk.index == i))
            self.assertTrue(np.allclose(xref[np.arange(N), chunk.start], chunk.data[..., 0]))
            self.assertTrue(np.all(chunk.data[..., margin] <= target))
            self.assertTrue(np.all(chunk.data[..., margin + 1] > target))

        index = chunk.start
        value = np.copy(chunk.data[..., 0])
        state = chunk.state(index)
        cp = np.copy(chunk.data)

        i = 3000
        target = 0.5 * (xref[..., i] + xref[..., i + 1])
        chunk.align(target)
        self.assertTrue(np.all(chunk.start + chunk.index == i))

        chunk.restore(state, value, index)
        self.assertTrue(np.allclose(cp, chunk.data))

    def test_loose_margins(self):
        """
        Chunked storage: random, different margins.
        """

        scale = 5
        offset = 0.1
        ref = prrng.pcg32(seed)
        xref = np.cumsum(offset + scale * ref.random([10000]))

        n = 100
        margin = 10
        min_margin = 2
        align = prrng.alignment(buffer=margin, margin=margin, min_margin=min_margin, strict=False)
        chunk = prrng.pcg32_cumsum(
            [n], seed, distribution=prrng.random, parameters=[scale, offset], align=align
        )
        self.assertEqual(chunk.start, 0)
        self.assertTrue(np.allclose(chunk.data, xref[0:n]))

        for i in range(1, 10):
            chunk.next(margin)
            self.assertEqual(chunk.start, i * n - i * margin)
            self.assertTrue(np.allclose(chunk.data, xref[chunk.start : chunk.start + n]))

        i0 = i * n - i * margin

        for i in range(1, 7):
            chunk.prev(margin)
            self.assertEqual(chunk.start, i0 - i * n + i * margin)
            self.assertTrue(np.allclose(chunk.data, xref[chunk.start : chunk.start + n]))

        index = chunk.start
        value = np.copy(chunk.data[..., 0])
        state = chunk.state(index)

        i = index + n // 2
        chunk.restore(state, value, index)
        chunk.align(xref[i])
        self.assertEqual(chunk.start, index)

        for i in range(index + n - 9, index + n + 20):
            chunk.restore(state, value, index)
            target = 0.5 * (xref[i - 1] + xref[i])
            chunk.align(target)
            self.assertGreater(chunk.start, index)
            self.assertGreaterEqual(np.argmin(target > chunk.data), min_margin + 1)

    def test_strict_margins(self):
        """
        Chunked storage: random, different margins.
        """

        scale = 5
        offset = 0.1
        ref = prrng.pcg32(seed)
        xref = np.cumsum(offset + scale * ref.random([10000]))

        n = 100
        margin = 10
        align = prrng.alignment(buffer=margin, margin=margin, strict=True)
        chunk = prrng.pcg32_cumsum(
            [n], seed, distribution=prrng.random, parameters=[scale, offset], align=align
        )
        self.assertEqual(chunk.start, 0)
        self.assertTrue(np.allclose(chunk.data, xref[0:n]))

        for i in range(1, 10):
            chunk.next(margin)
            self.assertEqual(chunk.start, i * n - i * margin)
            self.assertTrue(np.allclose(chunk.data, xref[chunk.start : chunk.start + n]))

        i0 = i * n - i * margin

        for i in range(1, 7):
            chunk.prev(margin)
            self.assertEqual(chunk.start, i0 - i * n + i * margin)
            self.assertTrue(np.allclose(chunk.data, xref[chunk.start : chunk.start + n]))

        index = chunk.start
        value = np.copy(chunk.data[..., 0])
        state = chunk.state(index)

        i = index + n // 2
        chunk.restore(state, value, index)
        chunk.align(xref[i])
        self.assertEqual(chunk.start, index)

        for i in range(index + n - 9, index + n + 20):
            chunk.restore(state, value, index)
            target = 0.5 * (xref[i - 1] + xref[i])
            chunk.align(target)
            self.assertGreater(chunk.start, index)
            self.assertEqual(np.argmin(target > chunk.data), margin + 1)


if __name__ == "__main__":

    unittest.main()
