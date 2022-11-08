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
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

        gen = prrng.pcg32_cumsum([100])
        gen.draw_chunk_weibull(k=k, scale=scale, offset=offset)
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
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

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
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

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
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

        n = 100
        gen = prrng.pcg32_cumsum([n])

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        def mycumsum(n):
            return gen.generator.cumsum_weibull(n, k, scale) + n * offset

        margin = 10
        gen.draw_chunk(mydraw)

        for i in [n + 10, 10 * n + 10, 40, n + 20]:
            target = 0.5 * (xref[i] + xref[i + 1])
            gen.align_chunk(mydraw, mycumsum, target, margin=margin, strict=True)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin)
            self.assertEqual(np.argmin(target > gen.chunk), margin)

            target = 0.5 * (xref[i - 1] + xref[i])
            gen.align_chunk(mydraw, mycumsum, target, margin=margin, strict=True)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin)
            self.assertEqual(np.argmin(target > gen.chunk), margin)

    def test_align_chunk_internal(self):
        """
        Align chunk with target (use default draw function).
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

        gen = prrng.pcg32_cumsum([100])

        n = 100
        margin = 10
        gen.draw_chunk_weibull(k=k, scale=scale, offset=offset)

        for i in [n + 10, 10 * n + 10, 40, n + 20]:
            target = 0.5 * (xref[i] + xref[i + 1])
            gen.align_chunk_weibull(target, k, scale, offset, margin=margin, strict=True)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin)
            self.assertEqual(np.argmin(target > gen.chunk), margin)

            target = 0.5 * (xref[i - 1] + xref[i])
            gen.align_chunk_weibull(target, k, scale, offset, margin=margin, strict=True)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], target)
            self.assertGreater(gen.chunk[-1], target)
            self.assertEqual(np.sum(target > gen.chunk), margin)
            self.assertEqual(np.argmin(target > gen.chunk), margin)

    def test_restore(self):
        """
        Restore state (use custom draw function).
        """

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

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
        gen = prrng.pcg32_array_cumsum([n], state, seq)
        gen.draw_chunk_weibull(k, scale, offset)
        self.assertTrue(np.allclose(xref[..., :n], gen.chunk))
        self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))

        for i in [500, 2012, 101]:

            margin = 10
            target = 0.5 * (xref[..., i] + xref[..., i + 1])
            gen.align_chunk_weibull(target, k, scale, offset, margin=margin, strict=True)
            self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))
            self.assertTrue(np.all(gen.chunk[..., margin - 1] <= target))
            self.assertTrue(np.all(gen.chunk[..., margin] > target))

    def test_array_init(self):
        """
        Array: apply some custom initialisation
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
        gen = prrng.pcg32_array_cumsum([n], state, seq)
        gen.draw_chunk_weibull(k, scale, offset)
        gen.chunk += x0
        self.assertTrue(np.allclose(xref[..., :n], gen.chunk))
        self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))

        for i in [500, 2012, 101]:

            margin = 10
            target = 0.5 * (xref[..., i] + xref[..., i + 1])
            gen.align_chunk_weibull(target, k, scale, offset, margin=margin, strict=True)
            self.assertTrue(np.allclose(xref[np.arange(state.size), gen.start], gen.chunk[..., 0]))
            self.assertTrue(np.all(gen.chunk[..., margin - 1] <= target))
            self.assertTrue(np.all(gen.chunk[..., margin] > target))

        index = gen.start
        value = np.copy(gen.chunk[..., 0])
        state = gen.state(index)
        chunk = np.copy(gen.chunk)

        i = 3000
        target = 0.5 * (xref[..., i] + xref[..., i + 1])
        gen.align_chunk_weibull(target, k, scale, offset)

        gen.restore(state, value, index)
        gen.draw_chunk_weibull(k, scale, offset)
        self.assertTrue(np.allclose(chunk, gen.chunk))


if __name__ == "__main__":

    unittest.main()
