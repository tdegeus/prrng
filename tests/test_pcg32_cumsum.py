import unittest

import numpy as np
import prrng


class Test_pcg32_cumum(unittest.TestCase):
    def test_draw_chunk(self):

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

        gen = prrng.pcg32_cumsum()
        gen.draw_chunk_weibull(100, k=k, scale=scale, offset=offset)
        lwr = gen.start
        upr = gen.start + gen.size
        self.assertTrue(np.allclose(gen.chunk, xref[: gen.size]))
        self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))

        gen = prrng.pcg32_cumsum()

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        gen.draw_chunk(100, mydraw)
        self.assertTrue(np.allclose(gen.chunk, xref[: gen.size]))

    def test_prev_chunk_next_chunk(self):

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

        gen = prrng.pcg32_cumsum()

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        gen.draw_chunk(100, mydraw)
        self.assertTrue(np.allclose(gen.chunk, xref[: gen.size]))

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

    def test_prev_chunk_next_chunk_mergin(self):

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

        gen = prrng.pcg32_cumsum()

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        gen.draw_chunk(100, mydraw)
        self.assertTrue(np.allclose(gen.chunk, xref[: gen.size]))

        for _ in range(5):
            gen.next_chunk(mydraw, margin=10)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))

        for i in range(5):
            gen.prev_chunk(mydraw, margin=10)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))

    def test_align_chunk(self):

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

        gen = prrng.pcg32_cumsum()

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        def mycumsum(n):
            return gen.generator.cumsum_weibull(n, k, scale) + n * offset

        n = 100
        gen.draw_chunk(n, mydraw)

        for i in [n + 10, 10 * n + 10, 40, n + 20]:
            x = 0.5 * (xref[i] + xref[i + 1])
            margin = 10
            gen.align_chunk(mydraw, mycumsum, target=x, margin=margin, strict=True)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], x)
            self.assertGreater(gen.chunk[-1], x)
            self.assertEqual(np.argmax(x <= gen.chunk) - 1, margin)

    def test_align_chunk_internal(self):

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

        gen = prrng.pcg32_cumsum()

        n = 100
        gen.draw_chunk_weibull(n, k=k, scale=scale, offset=offset)

        for i in [n + 10, 10 * n + 10, 40, n + 20]:
            x = 0.5 * (xref[i] + xref[i + 1])
            margin = 10
            gen.align_chunk_weibull(k=k, scale=scale, offset=offset, target=x, margin=margin, strict=True)
            lwr = gen.start
            upr = gen.start + gen.size
            self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))
            self.assertLessEqual(gen.chunk[0], x)
            self.assertGreater(gen.chunk[-1], x)
            self.assertEqual(np.argmax(x <= gen.chunk) - 1, margin)

    def test_restore(self):

        k = 2
        scale = 5
        offset = 0.1
        ref = prrng.pcg32()
        xref = np.cumsum(offset + ref.weibull([10000], k=k, scale=scale))

        gen = prrng.pcg32_cumsum()

        def mydraw(n):
            return gen.generator.weibull([n], k, scale) + offset

        n = 100
        gen.draw_chunk(n, mydraw)

        for _ in range(3):
            gen.next_chunk(mydraw, margin=10)
            value = gen.chunk[0]
            index = gen.start
            state = gen.state(index)

        for _ in range(3):
            gen.next_chunk(mydraw, margin=10)

        gen.restore(state, value, index)
        gen.draw_chunk(n, mydraw)
        lwr = gen.start
        upr = gen.start + gen.size
        self.assertTrue(np.allclose(gen.chunk, xref[lwr:upr]))


if __name__ == "__main__":

    unittest.main()
