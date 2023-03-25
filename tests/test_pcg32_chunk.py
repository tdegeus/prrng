import time
import unittest

import numpy as np
import prrng

seed = int(time.time())
np.random.seed(seed)


class Test_pcg32_chunk(unittest.TestCase):
    def test_array_random(self):
        scale = 5
        offset = 0.1
        N = 6
        initstate = seed + np.arange(N, dtype=np.uint64)
        seq = np.zeros_like(initstate)
        ref = prrng.pcg32_array(initstate, seq)
        xref = offset + scale * ref.random([10000])

        n = 100
        margin = 15
        align = prrng.alignment(margin=margin, buffer=1)
        chunk = prrng.pcg32_array_chunk([n], initstate, seq, prrng.random, [scale, offset], align)
        self.assertTrue(np.allclose(xref[..., :n], chunk.data))
        self.assertTrue(np.allclose(xref[np.arange(N), chunk.start], chunk.data[..., 0]))
        check = np.empty([N, n], dtype=np.float64)

        h = 4 * n
        indices = np.vstack(
            (
                np.sort(np.random.randint(margin, high=h, size=[10, N], dtype=int), axis=0),
                np.sort(np.random.randint(margin, high=h, size=[10, N], dtype=int), axis=0)[::-1],
                np.random.randint(margin, high=xref.shape[1] - 1 - n, size=[10, N], dtype=int),
            )
        )

        for index in indices:
            xl = xref[np.arange(N), index]
            xr = xref[np.arange(N), index + 1]
            chunk.align_at(index)
            for i in range(N):
                check[i, :] = xref[i, chunk.start[i] : chunk.start[i] + n]

            self.assertTrue(np.all(index == chunk.index_at_align))
            self.assertTrue(np.allclose(xl, chunk.left_of_align))
            self.assertTrue(np.allclose(xr, chunk.right_of_align))
            self.assertTrue(np.allclose(chunk.data, check))

            chunk.align_at(index)
            self.assertTrue(np.all(index == chunk.index_at_align))
            self.assertTrue(np.allclose(xl, chunk.left_of_align))
            self.assertTrue(np.allclose(xr, chunk.right_of_align))
            self.assertTrue(np.allclose(chunk.data, check))

    def test_tensor_random(self):
        scale = 5
        offset = 0.1
        N = 6
        initstate = seed + np.arange(N, dtype=np.uint64)
        seq = np.zeros_like(initstate)
        ref = prrng.pcg32_array(initstate, seq)
        xref = offset + scale * ref.random([10000])

        n = 100
        margin = 15
        align = prrng.alignment(margin=margin, buffer=1)
        chunk = prrng.pcg32_tensor_chunk_1_1(
            [n], initstate, seq, prrng.random, [scale, offset], align
        )
        self.assertTrue(np.allclose(xref[..., :n], chunk.data))
        self.assertTrue(np.allclose(xref[np.arange(N), chunk.start], chunk.data[..., 0]))
        check = np.empty([N, n], dtype=np.float64)

        h = 4 * n
        indices = np.vstack(
            (
                np.sort(np.random.randint(margin, high=h, size=[10, N], dtype=int), axis=0),
                np.sort(np.random.randint(margin, high=h, size=[10, N], dtype=int), axis=0)[::-1],
                np.random.randint(margin, high=xref.shape[1] - 1 - n, size=[10, N], dtype=int),
            )
        )

        for index in indices:
            xl = xref[np.arange(N), index]
            xr = xref[np.arange(N), index + 1]
            chunk.align_at(index)
            for i in range(N):
                check[i, :] = xref[i, chunk.start[i] : chunk.start[i] + n]

            self.assertTrue(np.all(index == chunk.index_at_align))
            self.assertTrue(np.allclose(xl, chunk.left_of_align))
            self.assertTrue(np.allclose(xr, chunk.right_of_align))
            self.assertTrue(np.allclose(chunk.data, check))

            chunk.align_at(index)
            self.assertTrue(np.all(index == chunk.index_at_align))
            self.assertTrue(np.allclose(xl, chunk.left_of_align))
            self.assertTrue(np.allclose(xr, chunk.right_of_align))
            self.assertTrue(np.allclose(chunk.data, check))

    def test_array_normal(self):
        mean = 5
        std = 0.1
        N = 6
        initstate = seed + np.arange(N, dtype=np.uint64)
        seq = np.zeros_like(initstate)
        ref = prrng.pcg32_array(initstate, seq)
        xref = ref.normal([10000], mean, std)

        n = 100
        margin = 15
        align = prrng.alignment(margin=margin, buffer=1)
        chunk = prrng.pcg32_array_chunk([n], initstate, seq, prrng.normal, [mean, std], align)
        self.assertTrue(np.allclose(xref[..., :n], chunk.data))
        self.assertTrue(np.allclose(xref[np.arange(N), chunk.start], chunk.data[..., 0]))
        check = np.empty([N, n], dtype=np.float64)

        h = 4 * n
        indices = np.vstack(
            (
                np.sort(np.random.randint(margin, high=h, size=[10, N], dtype=int), axis=0),
                np.sort(np.random.randint(margin, high=h, size=[10, N], dtype=int), axis=0)[::-1],
                np.random.randint(margin, high=xref.shape[1] - 1 - n, size=[10, N], dtype=int),
            )
        )

        for index in indices:
            xl = xref[np.arange(N), index]
            xr = xref[np.arange(N), index + 1]
            chunk.align_at(index)
            for i in range(N):
                check[i, :] = xref[i, chunk.start[i] : chunk.start[i] + n]

            self.assertTrue(np.all(index == chunk.index_at_align))
            self.assertTrue(np.allclose(xl, chunk.left_of_align))
            self.assertTrue(np.allclose(xr, chunk.right_of_align))
            self.assertTrue(np.allclose(chunk.data, check))

            chunk.align_at(index)
            self.assertTrue(np.all(index == chunk.index_at_align))
            self.assertTrue(np.allclose(xl, chunk.left_of_align))
            self.assertTrue(np.allclose(xr, chunk.right_of_align))
            self.assertTrue(np.allclose(chunk.data, check))


if __name__ == "__main__":
    unittest.main()
