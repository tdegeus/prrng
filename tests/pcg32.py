import unittest
import prrng
import time
import numpy as np

class Test_pcg32_basic(unittest.TestCase):

    def test_seed(self):

        seed = int(time.time())

        gen_a = prrng.pcg32(seed)
        a = gen_a.random([100])

        gen_b = prrng.pcg32(seed)
        b = gen_b.random([100])

        self.assertTrue(np.allclose(a, b))


    def test_restore(self):

        seed = int(time.time())

        gen = prrng.pcg32(seed)
        gen.random([123])

        state = gen.state()
        a = gen.random([100])

        gen.restore(state)
        b = gen.random([100])

        self.assertTrue(np.allclose(a, b))


class Test_pcg32_random(unittest.TestCase):

    def test_historic(self):

        gen = prrng.pcg32()

        a = gen.random([100])

        b = np.array([
            0.108379,  0.90696 ,  0.406692,  0.875239,  0.694849,  0.7435  ,
            0.167443,  0.621512,  0.221678,  0.895998,  0.401078,  0.396606,
            0.346894,  0.653979,  0.790445,  0.884927,  0.616019,  0.012579,
            0.377307,  0.0608  ,  0.23995 ,  0.1879  ,  0.328058,  0.278146,
            0.879473,  0.365613,  0.616987,  0.199623,  0.837729,  0.413446,
            0.807033,  0.891212,  0.906384,  0.284194,  0.473226,  0.238198,
            0.333253,  0.360564,  0.501208,  0.389194,  0.502242,  0.736847,
            0.713405,  0.915778,  0.857983,  0.056973,  0.246306,  0.911259,
            0.940772,  0.687423,  0.408766,  0.074081,  0.032931,  0.064742,
            0.001447,  0.95745 ,  0.501345,  0.813252,  0.343431,  0.664789,
            0.829031,  0.22576 ,  0.837668,  0.307977,  0.183911,  0.959587,
            0.170796,  0.424781,  0.924418,  0.933636,  0.614157,  0.007682,
            0.703196,  0.234229,  0.728257,  0.975139,  0.933431,  0.341162,
            0.756521,  0.874001,  0.154687,  0.351131,  0.790386,  0.014452,
            0.213094,  0.378399,  0.62506 ,  0.680397,  0.998596,  0.331519,
            0.03142 ,  0.765982,  0.734759,  0.719876,  0.889892,  0.263362,
            0.989077,  0.308017,  0.273916,  0.766872])

        self.assertTrue(np.allclose(a, b, 1e-3, 1e-4))


class Test_pcg32_array(unittest.TestCase):

    def test_list(self):

        seed = np.arange(5)
        gen = prrng.pcg32_array(seed)
        state = gen.state()
        a = gen.random([4, 5])
        b = gen.random([4, 5])
        self.assertTrue(not np.allclose(a, b));

        # test "restore"

        gen.restore(state)
        self.assertTrue(np.allclose(a, gen.random([4, 5])))
        self.assertTrue(not np.allclose(a, gen.random([4, 5])))

        # test "__getitem__"

        for i in range(gen.size()):
            gen[i].restore(state[i])

        self.assertTrue(np.allclose(a, gen.random([4, 5])))
        self.assertTrue(not np.allclose(a, gen.random([4, 5])))

        # test "initstate" and "initseq"

        initstate = gen.initstate()
        initseq = gen.initseq()

        for i in range(gen.size()):
            self.assertTrue(gen[i].initstate() == initstate[i])
            self.assertTrue(gen[i].initseq() == initseq[i])

        for i in range(gen.size()):
            self.assertTrue(gen[i].initstate() == initstate[i])
            self.assertTrue(gen[i].initseq() == initseq[i])

    def test_array(self):

        seed = np.arange(10).reshape([2, -1])
        gen = prrng.pcg32_array(seed)
        state = gen.state()
        a = gen.random([4, 5])
        b = gen.random([4, 5])
        self.assertTrue(not np.allclose(a, b));

        # test "restore"

        gen.restore(state)
        self.assertTrue(np.allclose(a, gen.random([4, 5])))
        self.assertTrue(not np.allclose(a, gen.random([4, 5])))

        # test "__getitem__"

        for i in range(gen.size()):
            gen[i].restore(state.ravel()[i])

        self.assertTrue(np.allclose(a, gen.random([4, 5])))
        self.assertTrue(not np.allclose(a, gen.random([4, 5])))

        # test "__getitem__"

        for i in range(gen.shape(0)):
            for j in range(gen.shape(1)):
                gen[i, j].restore(state[i, j])

        self.assertTrue(np.allclose(a, gen.random([4, 5])))
        self.assertTrue(not np.allclose(a, gen.random([4, 5])))

        # test "initstate" and "initseq"

        initstate = gen.initstate()
        initseq = gen.initseq()

        for i in range(gen.shape(0)):
            for j in range(gen.shape(1)):
                self.assertTrue(gen[i, j].initstate() == initstate[i, j])
                self.assertTrue(gen[i, j].initseq() == initseq[i, j])

        for i in range(gen.shape(0)):
            for j in range(gen.shape(1)):
                self.assertTrue(gen[i, j].initstate() == initstate[i, j])
                self.assertTrue(gen[i, j].initseq() == initseq[i, j])


if __name__ == '__main__':

    unittest.main()
