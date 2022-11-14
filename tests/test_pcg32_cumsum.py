import unittest

import numpy as np
import prrng


class Test_pcg32_cumum(unittest.TestCase):
    def test_draw_chunk(self):
        """
        Draw initial chunk, with default and custom draw function.
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        gen = prrng.pcg32_cumsum([100])
        gen.draw_chunk_weibull(k, scale, offset)
        lwr = gen.start
        upr = gen.start + gen.size
        self.assertEqual(gen.start, 0)
        self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))

        gen = prrng.pcg32_cumsum([100])

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        gen.draw_chunk(mydraw)
        lwr = gen.start
        upr = gen.start + gen.size
        self.assertEqual(gen.start, 0)
        self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))

    def test_prev_chunk_next_chunk(self):
        """
        Shift chunks right and left (use custom draw function).
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        gen = prrng.pcg32_cumsum([100])

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        gen.draw_chunk(mydraw)
        lwr = gen.start
        upr = gen.start + gen.size
        self.assertEqual(gen.start, 0)
        self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))

        for _ in range(5):
            gen.next_chunk(mydraw)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))

        for i in range(5):
            gen.prev_chunk(mydraw)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))

    def test_prev_chunk_next_chunk_margin(self):
        """
        Shift chunks right and left leaving some overlap (use custom draw function).
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        gen = prrng.pcg32_cumsum([100])
        margin = 10

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        gen.draw_chunk(mydraw)
        self.assertTrue(np.allclose(gen.chunk, xref[: gen.size]))

        for _ in range(5):
            back = gen.chunk[-1]
            gen.next_chunk(mydraw, margin=margin)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertEqual(np.argmin(back >= gen.chunk), margin)
            self.assertEqual(np.sum(back >= gen.chunk), margin)

        for i in range(5):
            front = gen.chunk[0]
            gen.prev_chunk(mydraw, margin=margin)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertEqual(np.argmax(front <= gen.chunk), gen.size - margin)
            self.assertEqual(np.sum(front <= gen.chunk), margin)

    def test_align_chunk(self):
        """
        Align chunk with target (use custom draw function).
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        n = 100
        margin = 10
        gen = prrng.pcg32_cumsum([n], align=prrng.alignment(margin=margin, strict=True))

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        def mycumsum(n):
            return gen.generator.cumsum_weibull(n, k, scale) + n * offset

        gen.draw_chunk(mydraw)

        for i in [n + 10, 10 * n + 10, 40, n + 20]:
            target = 0.5 * (xref[i] + xref[i + 1])
            gen.align_chunk(mydraw, mycumsum, target)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin + 1)
            self.assertEqual(np.argmin(target > gen.chunk), margin + 1)

            target = 0.5 * (xref[i - 1] + xref[i])
            gen.align_chunk(mydraw, mycumsum, target)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin + 1)
            self.assertEqual(np.argmin(target > gen.chunk), margin + 1)

    def test_align_chunk_internal(self):
        """
        Align chunk with target (use default draw function).
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        n = 100
        margin = 10
        gen = prrng.pcg32_cumsum([n], align=prrng.alignment(margin=margin, strict=True))
        gen.draw_chunk_weibull(k, scale, offset)

        for i in [n + 10, 10 * n + 10, 40, n + 20]:
            target = 0.5 * (xref[i] + xref[i + 1])
            gen.align_chunk_weibull(target, k, scale, offset)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin + 1)
            self.assertEqual(np.argmin(target > gen.chunk), margin + 1)

            target = 0.5 * (xref[i - 1] + xref[i])
            gen.align_chunk_weibull(target, k, scale, offset)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin + 1)
            self.assertEqual(np.argmin(target > gen.chunk), margin + 1)

    def test_align_chunk_first(self):
        """
        Align chunk with target (use default draw function).
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        n = 5
        margin = 0
        gen = prrng.pcg32_cumsum([n], align=prrng.alignment(margin=margin, strict=True))
        gen.draw_chunk_weibull(k, scale, offset)

        for i in [n + 10, 10 * n + 10, 40, n + 20]:
            target = 0.5 * (xref[i] + xref[i + 1])
            gen.align_chunk_weibull(target, k, scale, offset)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin + 1)
            self.assertEqual(np.argmin(target > gen.chunk), margin + 1)

            target = 0.5 * (xref[i - 1] + xref[i])
            gen.align_chunk_weibull(target, k, scale, offset)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin + 1)
            self.assertEqual(np.argmin(target > gen.chunk), margin + 1)

    def test_align_chunk_minmargin(self):
        """
        Align chunk with target (use default draw function).
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        n = 5
        margin = 2
        min_margin = 1
        align = prrng.alignment(margin=margin, min_margin=min_margin, strict=False)
        gen = prrng.pcg32_cumsum([n], align=align)
        gen.draw_chunk_weibull(k, scale, offset)

        for i in [n + 10, 10 * n + 10, 40, n + 20]:
            target = 0.5 * (xref[i] + xref[i + 1])
            gen.align_chunk_weibull(target, k, scale, offset)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertGreater(np.sum(target > gen.chunk), min_margin)
            self.assertGreater(np.argmin(target > gen.chunk), min_margin)

            target = 0.5 * (xref[i - 1] + xref[i])
            gen.align_chunk_weibull(target, k, scale, offset)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertGreater(np.sum(target > gen.chunk), min_margin)
            self.assertGreater(np.argmin(target > gen.chunk), min_margin)

    def test_restore(self):
        """
        Restore state (use custom draw function).
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k, scale))

        gen = prrng.pcg32_cumsum([100])

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        # some manipulations

        gen.draw_chunk(mydraw)

        for _ in range(3):
            gen.next_chunk(mydraw, margin=10)
            value = gen.chunk[0]
            index = gen.start
            state = gen.state(index)

        for _ in range(3):
            gen.next_chunk(mydraw, margin=10)

        # restore

        gen.restore(state, value, index)
        gen.draw_chunk(mydraw)
        lwr = gen.start
        upr = gen.start + gen.size
        self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))

    def test_array(self):
        """
        Array: set a chunk.
        """

        k = 2
        scale = 5
        offset = 0.1
        state = np.arange(6, dtype=np.uint64)
        seq = np.zeros_like(state)
        ref = prrng.pcg32_array(state, seq)
        xref = np.cumsum(offset + ref.weibull([10000], k, scale), axis=-1)

        n = 100
        margin = 10
        align = align = prrng.alignment(margin=margin, strict=True)
        gen = prrng.pcg32_array_cumsum([n], state, seq, prrng.weibull, [k, scale, offset], align)
        self.assertTrue(np.allclose(xref[..., :n], gen.chunk))
        self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))

        for i in [500, 2012, 101]:
            target = 0.5 * (xref[..., i] + xref[..., i + 1])
            gen.align(target)
            self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))
            self.assertTrue(np.all(gen.chunk[..., margin] <= target))
            self.assertTrue(np.all(gen.chunk[..., margin + 1] > target))

    def test_array_weibull(self):
        """
        Array (weibull)
        """

        k = 2
        scale = 5
        offset = 0.1
        state = np.arange(6, dtype=np.uint64)
        seq = np.zeros_like(state)
        ref = prrng.pcg32_array(state, seq)
        x0 = ref.random([]).reshape(-1, 1)
        ref = prrng.pcg32_array(state, seq)
        xref = np.cumsum(offset + ref.weibull([10000], k, scale), axis=-1) + x0

        n = 100
        margin = 10
        align = prrng.alignment(margin=margin, strict=True)
        gen = prrng.pcg32_array_cumsum([n], state, seq, prrng.weibull, [k, scale, offset], align)
        gen.chunk += x0
        self.assertTrue(np.allclose(xref[..., :n], gen.chunk))
        self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))

        for i in [500, 2012, 101]:
            target = 0.5 * (xref[..., i] + xref[..., i + 1])
            gen.align(target)
            self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))
            self.assertTrue(np.all(gen.chunk[..., margin] <= target))
            self.assertTrue(np.all(gen.chunk[..., margin + 1] > target))

        index = gen.start
        value = np.copy(gen.chunk[..., 0])
        state = gen.state(index)
        chunk = np.copy(gen.chunk)

        i = 3000
        target = 0.5 * (xref[..., i] + xref[..., i + 1])
        gen.align(target)

        gen.restore(state, value, index)
        self.assertTrue(np.allclose(chunk, gen.chunk))

    def test_array_delta(self):
        """
        Array (delta)
        """

        scale = 5
        offset = 0.1
        state = np.arange(6, dtype=np.uint64)
        seq = np.zeros_like(state)
        ref = prrng.pcg32_array(state, seq)
        x0 = ref.random([]).reshape(-1, 1)
        ref = prrng.pcg32_array(state, seq)
        xref = np.cumsum(offset + ref.delta([10000], scale), axis=-1) + x0

        n = 100
        margin = 10
        align = prrng.alignment(margin=margin, strict=True)
        gen = prrng.pcg32_array_cumsum([n], state, seq, prrng.delta, [scale, offset], align)
        gen.chunk += x0
        self.assertTrue(np.allclose(xref[..., :n], gen.chunk))
        self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))

        for i in [500, 2012, 101]:
            target = 0.5 * (xref[..., i] + xref[..., i + 1])
            gen.align(target)
            self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))
            self.assertTrue(np.all(gen.chunk[..., margin] <= target))
            self.assertTrue(np.all(gen.chunk[..., margin + 1] > target))

        index = gen.start
        value = np.copy(gen.chunk[..., 0])
        state = gen.state(index)
        chunk = np.copy(gen.chunk)

        i = 3000
        target = 0.5 * (xref[..., i] + xref[..., i + 1])
        gen.align(target)

        gen.restore(state, value, index)
        self.assertTrue(np.allclose(chunk, gen.chunk))

    def test_delta(self):
        """
        Shift chunks right and left for a delta distribution
        """

        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        state = ref.state()
        xref = np.cumsum(offset + ref.delta([10000], scale))
        self.assertEqual(state, ref.state())

        gen = prrng.pcg32_cumsum([100])
        gen.draw_chunk_delta(scale, offset)
        lwr = gen.start
        upr = gen.start + gen.size
        self.assertEqual(gen.start, 0)
        self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
        self.assertEqual(state, gen.generator.state())
        self.assertEqual(gen.generator_index, 0)

        for _ in range(5):
            gen.next_chunk_delta(scale, offset)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertEqual(state, gen.generator.state())
            self.assertEqual(gen.generator_index, 0)

        for i in range(5):
            gen.prev_chunk_delta(scale, offset)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertEqual(state, gen.generator.state())
            self.assertEqual(gen.generator_index, 0)


if __name__ == "__main__":

    unittest.main()
