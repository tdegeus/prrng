import time
import unittest

import numpy as np
import prrng


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

    def test_rowmajor(self):

        seed = int(time.time())
        gen = prrng.pcg32(seed)
        gen.random([123])
        state = gen.state()

        a = gen.random([100, 5, 11])

        gen.restore(state)
        gen.advance(99 * 5 * 11)
        b = gen.random([5, 11])

        self.assertTrue(np.allclose(a[-1, ...], b))

    def test_pcg32_decide(self):

        seed = int(time.time())
        gen = prrng.pcg32(seed)
        p = gen.random([5, 10])

        gen = prrng.pcg32(seed + 1)
        state = gen.state()
        decision = np.empty(p.shape, dtype=bool)
        decision2 = np.empty(p.shape, dtype=bool)
        gen.decide(p, decision)
        gen.decide(p, decision2)

        gen.restore(state)
        self.assertTrue(np.all(np.equal(decision, gen.random(p.shape) <= p)))
        self.assertTrue(np.all(np.equal(decision2, gen.random(p.shape) <= p)))

        p = np.ones_like(p)
        gen.decide(p, decision)
        self.assertTrue(np.all(decision))

        p = np.zeros_like(p)
        gen.decide(p, decision)
        self.assertTrue(not np.any(decision))

    def test_pcg32_cumsum_random(self):

        seed = int(time.time())
        gen = prrng.pcg32(seed)
        state = gen.state()
        n = 10000
        offset = 0.1
        mean = 2.3

        gen.restore(state)
        a = np.cumsum(gen.random([n]))

        gen.restore(state)
        aprime = np.cumsum(offset + mean * gen.random([n]))

        gen.restore(state)
        b = gen.cumsum_random(n)

        gen.restore(state)
        bprime = offset * n + mean * gen.cumsum_random(n)

        self.assertAlmostEqual(a[-1], b)
        self.assertAlmostEqual(aprime[-1], bprime)

    def test_pcg32_cumsum_normal(self):

        seed = int(time.time())
        gen = prrng.pcg32(seed)
        state = gen.state()
        n = 10000
        offset = 0.1
        mean = 2.3
        mu = 1.2
        sigma = 0.3

        gen.restore(state)
        a = np.cumsum(gen.normal([n]))

        gen.restore(state)
        aprime = np.cumsum(offset + mean * gen.normal([n]))

        gen.restore(state)
        acustom = np.cumsum(offset + mean * gen.normal([n], mu, sigma))

        gen.restore(state)
        b = gen.cumsum_normal(n)

        gen.restore(state)
        bprime = offset * n + mean * gen.cumsum_normal(n)

        gen.restore(state)
        bcustom = offset * n + mean * gen.cumsum_normal(n, mu, sigma)

        self.assertAlmostEqual(a[-1], b)
        self.assertAlmostEqual(aprime[-1], bprime)
        self.assertAlmostEqual(acustom[-1], bcustom)

    def test_pcg32_cumsum_exponential(self):

        seed = int(time.time())
        gen = prrng.pcg32(seed)
        state = gen.state()
        n = 10000
        offset = 0.1
        mean = 2.3
        rate = 1.2

        gen.restore(state)
        a = np.cumsum(gen.exponential([n]))

        gen.restore(state)
        aprime = np.cumsum(offset + mean * gen.exponential([n]))

        gen.restore(state)
        acustom = np.cumsum(offset + mean * gen.exponential([n], rate))

        gen.restore(state)
        b = gen.cumsum_exponential(n)

        gen.restore(state)
        bprime = offset * n + mean * gen.cumsum_exponential(n)

        gen.restore(state)
        bcustom = offset * n + mean * gen.cumsum_exponential(n, rate)

        self.assertAlmostEqual(a[-1], b)
        self.assertAlmostEqual(aprime[-1], bprime)
        self.assertAlmostEqual(acustom[-1], bcustom)

    def test_pcg32_cumsum_weibull(self):

        seed = int(time.time())
        gen = prrng.pcg32(seed)
        state = gen.state()
        n = 10000
        offset = 0.1
        mean = 2.3
        lam = 1.2
        k = 0.3

        gen.restore(state)
        a = np.cumsum(gen.weibull([n]))

        gen.restore(state)
        aprime = np.cumsum(offset + mean * gen.weibull([n]))

        gen.restore(state)
        acustom = np.cumsum(offset + mean * gen.weibull([n], lam, k))

        gen.restore(state)
        b = gen.cumsum_weibull(n)

        gen.restore(state)
        bprime = offset * n + mean * gen.cumsum_weibull(n)

        gen.restore(state)
        bcustom = offset * n + mean * gen.cumsum_weibull(n, lam, k)

        self.assertAlmostEqual(a[-1], b)
        self.assertAlmostEqual(aprime[-1], bprime)
        self.assertAlmostEqual(acustom[-1], bcustom)

    def test_pcg32_cumsum_gamma(self):

        seed = int(time.time())
        gen = prrng.pcg32(seed)
        state = gen.state()
        n = 10000
        offset = 0.1
        mean = 2.3
        k = 1.2
        theta = 0.3

        gen.restore(state)
        a = np.cumsum(gen.gamma([n]))

        gen.restore(state)
        aprime = np.cumsum(offset + mean * gen.gamma([n]))

        gen.restore(state)
        acustom = np.cumsum(offset + mean * gen.gamma([n], k, theta))

        gen.restore(state)
        b = gen.cumsum_gamma(n)

        gen.restore(state)
        bprime = offset * n + mean * gen.cumsum_gamma(n)

        gen.restore(state)
        bcustom = offset * n + mean * gen.cumsum_gamma(n, k, theta)

        self.assertAlmostEqual(a[-1], b)
        self.assertAlmostEqual(aprime[-1], bprime)
        self.assertAlmostEqual(acustom[-1], bcustom)


class Test_pcg32_random(unittest.TestCase):
    """
    Random distribution.
    """

    def test_historic(self):

        gen = prrng.pcg32()

        a = gen.random([100])

        ha = np.array(
            [
                0.108379,
                0.90696,
                0.406692,
                0.875239,
                0.694849,
                0.7435,
                0.167443,
                0.621512,
                0.221678,
                0.895998,
                0.401078,
                0.396606,
                0.346894,
                0.653979,
                0.790445,
                0.884927,
                0.616019,
                0.012579,
                0.377307,
                0.0608,
                0.23995,
                0.1879,
                0.328058,
                0.278146,
                0.879473,
                0.365613,
                0.616987,
                0.199623,
                0.837729,
                0.413446,
                0.807033,
                0.891212,
                0.906384,
                0.284194,
                0.473226,
                0.238198,
                0.333253,
                0.360564,
                0.501208,
                0.389194,
                0.502242,
                0.736847,
                0.713405,
                0.915778,
                0.857983,
                0.056973,
                0.246306,
                0.911259,
                0.940772,
                0.687423,
                0.408766,
                0.074081,
                0.032931,
                0.064742,
                0.001447,
                0.95745,
                0.501345,
                0.813252,
                0.343431,
                0.664789,
                0.829031,
                0.22576,
                0.837668,
                0.307977,
                0.183911,
                0.959587,
                0.170796,
                0.424781,
                0.924418,
                0.933636,
                0.614157,
                0.007682,
                0.703196,
                0.234229,
                0.728257,
                0.975139,
                0.933431,
                0.341162,
                0.756521,
                0.874001,
                0.154687,
                0.351131,
                0.790386,
                0.014452,
                0.213094,
                0.378399,
                0.62506,
                0.680397,
                0.998596,
                0.331519,
                0.03142,
                0.765982,
                0.734759,
                0.719876,
                0.889892,
                0.263362,
                0.989077,
                0.308017,
                0.273916,
                0.766872,
            ]
        )

        self.assertTrue(np.allclose(a, ha, 1e-3, 1e-4))


class Test_pcg32_normal(unittest.TestCase):
    """
    Normal distribution.
    """

    def test_historic(self):

        a = prrng.pcg32().normal([102])
        b = prrng.pcg32().normal([102], 2.0)
        c = prrng.pcg32().normal([102], 2.0, 2.0)

        ha = np.array(
            [
                [-1.23519642, 1.32226552, -0.23606174, 1.1515092, 0.50964317, 0.654174],
                [-0.96431875, 0.30945346, -0.7665385, 1.25907233, -0.25055791, -0.26214164],
                [-0.39371966, 0.39608421, 0.80796671, 1.1999846, 0.29504106, -2.23897327],
                [-0.31256146, -1.54809412, -0.70646239, -0.88566324, -0.44528089, -0.5883569],
                [1.1723582, -0.34349613, 0.29757597, -0.84296958, 0.985169, -0.21868835],
                [0.86701283, 1.23299688, 1.31881226, -0.57042665, -0.06716238, -0.71211011],
                [-0.43094957, -0.35695237, 0.00302745, -0.2814196, 0.00562099, 0.63365361],
                [0.56335888, 1.37721989, 1.07130241, -1.58070225, -0.68615923, 1.34854886],
                [1.56129108, 0.48855752, -0.23071934, -1.44605251, -1.83935692, -1.51613637],
                [-2.97871394, 1.72183269, 0.00337231, 0.88994379, -0.40311648, 0.42556999],
                [0.9503449, -0.75288369, 0.98491785, -0.50159245, -0.90056034, 1.74591675],
                [-0.95102437, -0.18967778, 1.43542874, 1.50343263, 0.29017034, -2.42370467],
                [0.53361573, -0.72499015, 0.60755019, 1.96233964, 1.50184391, -0.40929437],
                [0.69515581, 1.14551022, -1.01653841, -0.38226839, 0.80776125, -2.18479046],
                [-0.79573114, -0.3096885, 0.3187984, 0.46880912, 2.98799707, -0.43572254],
                [-1.86032782, 0.72567679, 0.62727029, 0.58247416, 1.22595408, -0.63301433],
                [2.29305048, -0.50148, -0.60101079, 0.7285828, 1.18369435, 0.36896617],
            ]
        ).ravel()

        hb = np.array(
            [
                [0.76480358, 3.32226552, 1.76393826, 3.1515092, 2.50964317, 2.654174],
                [1.03568125, 2.30945346, 1.2334615, 3.25907233, 1.74944209, 1.73785836],
                [1.60628034, 2.39608421, 2.80796671, 3.1999846, 2.29504106, -0.23897327],
                [1.68743854, 0.45190588, 1.29353761, 1.11433676, 1.55471911, 1.4116431],
                [3.1723582, 1.65650387, 2.29757597, 1.15703042, 2.985169, 1.78131165],
                [2.86701283, 3.23299688, 3.31881226, 1.42957335, 1.93283762, 1.28788989],
                [1.56905043, 1.64304763, 2.00302745, 1.7185804, 2.00562099, 2.63365361],
                [2.56335888, 3.37721989, 3.07130241, 0.41929775, 1.31384077, 3.34854886],
                [3.56129108, 2.48855752, 1.76928066, 0.55394749, 0.16064308, 0.48386363],
                [-0.97871394, 3.72183269, 2.00337231, 2.88994379, 1.59688352, 2.42556999],
                [2.9503449, 1.24711631, 2.98491785, 1.49840755, 1.09943966, 3.74591675],
                [1.04897563, 1.81032222, 3.43542874, 3.50343263, 2.29017034, -0.42370467],
                [2.53361573, 1.27500985, 2.60755019, 3.96233964, 3.50184391, 1.59070563],
                [2.69515581, 3.14551022, 0.98346159, 1.61773161, 2.80776125, -0.18479046],
                [1.20426886, 1.6903115, 2.3187984, 2.46880912, 4.98799707, 1.56427746],
                [0.13967218, 2.72567679, 2.62727029, 2.58247416, 3.22595408, 1.36698567],
                [4.29305048, 1.49852, 1.39898921, 2.7285828, 3.18369435, 2.36896617],
            ]
        ).ravel()

        hc = np.array(
            [
                [-0.47039285, 4.64453103, 1.52787652, 4.30301839, 3.01928635, 3.30834801],
                [0.0713625, 2.61890692, 0.466923, 4.51814466, 1.49888419, 1.47571672],
                [1.21256067, 2.79216842, 3.61593341, 4.39996919, 2.59008212, -2.47794653],
                [1.37487709, -1.09618825, 0.58707522, 0.22867353, 1.10943822, 0.8232862],
                [4.3447164, 1.31300774, 2.59515194, 0.31406084, 3.97033801, 1.5626233],
                [3.73402566, 4.46599376, 4.63762453, 0.8591467, 1.86567523, 0.57577977],
                [1.13810085, 1.28609526, 2.00605489, 1.43716079, 2.01124198, 3.26730721],
                [3.12671776, 4.75443978, 4.14260483, -1.1614045, 0.62768154, 4.69709773],
                [5.12258217, 2.97711503, 1.53856132, -0.89210502, -1.67871384, -1.03227275],
                [-3.95742788, 5.44366538, 2.00674462, 3.77988757, 1.19376704, 2.85113998],
                [3.9006898, 0.49423262, 3.96983569, 0.99681509, 0.19887933, 5.4918335],
                [0.09795126, 1.62064443, 4.87085747, 5.00686525, 2.58034068, -2.84740934],
                [3.06723145, 0.5500197, 3.21510038, 5.92467929, 5.00368782, 1.18141127],
                [3.39031162, 4.29102044, -0.03307682, 1.23546322, 3.6155225, -2.36958092],
                [0.40853772, 1.38062299, 2.63759679, 2.93761823, 7.97599415, 1.12855491],
                [-1.72065565, 3.45135358, 3.25454058, 3.16494832, 4.45190816, 0.73397134],
                [6.58610095, 0.99704001, 0.79797842, 3.45716561, 4.3673887, 2.73793233],
            ]
        ).ravel()

        self.assertTrue(np.allclose(a, ha, 1e-3, 1e-4))
        self.assertTrue(np.allclose(b, hb, 1e-3, 1e-4))
        self.assertTrue(np.allclose(c, hc, 1e-3, 1e-4))


class Test_pcg32_exponential(unittest.TestCase):
    """
    Exponential distribution.
    """

    def test_historic(self):

        a = prrng.pcg32().exponential([102])
        b = prrng.pcg32().exponential([102], 2.0)

        ha = np.array(
            [
                [1.14713794e-01, 2.37472711e+00, 5.22042264e-01],
                [2.08135211e+00, 1.18694941e+00, 1.36062718e+00],
                [1.83253677e-01, 9.71570105e-01, 2.50614928e-01],
                [2.26334415e+00, 5.12623847e-01, 5.05185100e-01],
                [4.26015851e-01, 1.06125450e+00, 1.56276964e+00],
                [2.16219151e+00, 9.57161540e-01, 1.26586110e-02],
                [4.73701551e-01, 6.27266364e-02, 2.74371475e-01],
                [2.08131193e-01, 3.97583830e-01, 3.25932903e-01],
                [2.11588450e+00, 4.55095511e-01, 9.59685281e-01],
                [2.22672073e-01, 1.81849039e+00, 5.33491248e-01],
                [1.64523355e+00, 2.21834998e+00, 2.36855459e+00],
                [3.34346348e-01, 6.40984073e-01, 2.72068988e-01],
                [4.05343893e-01, 4.47168321e-01, 6.95565652e-01],
                [4.92976361e-01, 6.97642147e-01, 1.33501794e+00],
                [1.24968428e+00, 2.47429748e+00, 1.95181036e+00],
                [5.86604354e-02, 2.82769296e-01, 2.42203374e+00],
                [2.82636859e+00, 1.16290285e+00, 5.25544117e-01],
                [7.69687771e-02, 3.34857966e-02, 6.69333183e-02],
                [1.44835274e-03, 3.15707741e+00, 6.95841517e-01],
                [1.67799498e+00, 4.20727898e-01, 1.09299627e+00],
                [1.76627586e+00, 2.55873254e-01, 1.81811034e+00],
                [3.68136252e-01, 2.03231942e-01, 3.20861095e+00],
                [1.87289078e-01, 5.53004121e-01, 2.58253075e+00],
                [2.71260417e+00, 9.52324825e-01, 7.71120177e-03],
                [1.21468426e+00, 2.66872157e-01, 1.30289880e+00],
                [3.69443577e+00, 2.70952059e+00, 4.17277336e-01],
                [1.41272527e+00, 2.07148185e+00, 1.68047753e-01],
                [4.32524660e-01, 1.56248745e+00, 1.45575559e-02],
                [2.39646684e-01, 4.75456748e-01, 9.80990074e-01],
                [1.14067555e+00, 6.56838637e+00, 4.02747340e-01],
                [3.19237636e-02, 1.45235525e+00, 1.32711629e+00],
                [1.27252409e+00, 2.20629378e+00, 3.05658880e-01],
                [4.51692586e+00, 3.68193423e-01, 3.20090119e-01],
                [1.45616569e+00, 2.13481018e+00, 1.03260976e+00],
            ]
        ).ravel()

        hb = np.array(
            [
                [2.29427588e-01, 4.74945422e+00, 1.04408453e+00],
                [4.16270422e+00, 2.37389882e+00, 2.72125437e+00],
                [3.66507354e-01, 1.94314021e+00, 5.01229856e-01],
                [4.52668830e+00, 1.02524769e+00, 1.01037020e+00],
                [8.52031703e-01, 2.12250901e+00, 3.12553929e+00],
                [4.32438303e+00, 1.91432308e+00, 2.53172220e-02],
                [9.47403101e-01, 1.25453273e-01, 5.48742951e-01],
                [4.16262386e-01, 7.95167660e-01, 6.51865805e-01],
                [4.23176900e+00, 9.10191022e-01, 1.91937056e+00],
                [4.45344145e-01, 3.63698077e+00, 1.06698250e+00],
                [3.29046710e+00, 4.43669996e+00, 4.73710919e+00],
                [6.68692697e-01, 1.28196815e+00, 5.44137975e-01],
                [8.10687787e-01, 8.94336642e-01, 1.39113130e+00],
                [9.85952722e-01, 1.39528429e+00, 2.67003588e+00],
                [2.49936856e+00, 4.94859496e+00, 3.90362072e+00],
                [1.17320871e-01, 5.65538592e-01, 4.84406748e+00],
                [5.65273718e+00, 2.32580569e+00, 1.05108823e+00],
                [1.53937554e-01, 6.69715932e-02, 1.33866637e-01],
                [2.89670548e-03, 6.31415483e+00, 1.39168303e+00],
                [3.35598995e+00, 8.41455796e-01, 2.18599254e+00],
                [3.53255171e+00, 5.11746508e-01, 3.63622069e+00],
                [7.36272505e-01, 4.06463885e-01, 6.41722190e+00],
                [3.74578156e-01, 1.10600824e+00, 5.16506151e+00],
                [5.42520834e+00, 1.90464965e+00, 1.54224035e-02],
                [2.42936852e+00, 5.33744313e-01, 2.60579760e+00],
                [7.38887154e+00, 5.41904118e+00, 8.34554671e-01],
                [2.82545055e+00, 4.14296369e+00, 3.36095506e-01],
                [8.65049321e-01, 3.12497490e+00, 2.91151118e-02],
                [4.79293367e-01, 9.50913495e-01, 1.96198015e+00],
                [2.28135110e+00, 1.31367727e+01, 8.05494680e-01],
                [6.38475272e-02, 2.90471050e+00, 2.65423259e+00],
                [2.54504819e+00, 4.41258756e+00, 6.11317761e-01],
                [9.03385173e+00, 7.36386846e-01, 6.40180237e-01],
                [2.91233137e+00, 4.26962036e+00, 2.06521953e+00],
            ]
        ).ravel()

        self.assertTrue(np.allclose(a, ha, 1e-3, 1e-4))
        self.assertTrue(np.allclose(b, hb, 1e-3, 1e-4))


class Test_pcg32_weibull(unittest.TestCase):
    """
    Weibull distribution.
    """

    def test_historic(self):

        gen = prrng.pcg32()

        a = gen.weibull([100])
        b = gen.weibull([100], 2.0)

        ha = np.array(
            [
                0.114714,
                2.374727,
                0.522042,
                2.081352,
                1.186949,
                1.360627,
                0.183254,
                0.97157,
                0.250615,
                2.263344,
                0.512624,
                0.505185,
                0.426016,
                1.061255,
                1.56277,
                2.162192,
                0.957162,
                0.012659,
                0.473702,
                0.062727,
                0.274371,
                0.208131,
                0.397584,
                0.325933,
                2.115884,
                0.455096,
                0.959685,
                0.222672,
                1.81849,
                0.533491,
                1.645234,
                2.21835,
                2.368555,
                0.334346,
                0.640984,
                0.272069,
                0.405344,
                0.447168,
                0.695566,
                0.492976,
                0.697642,
                1.335018,
                1.249684,
                2.474297,
                1.95181,
                0.05866,
                0.282769,
                2.422034,
                2.826369,
                1.162903,
                0.525544,
                0.076969,
                0.033486,
                0.066933,
                0.001448,
                3.157077,
                0.695842,
                1.677995,
                0.420728,
                1.092996,
                1.766276,
                0.255873,
                1.81811,
                0.368136,
                0.203232,
                3.208611,
                0.187289,
                0.553004,
                2.582531,
                2.712604,
                0.952325,
                0.007711,
                1.214684,
                0.266872,
                1.302899,
                3.694436,
                2.709521,
                0.417277,
                1.412725,
                2.071482,
                0.168048,
                0.432525,
                1.562487,
                0.014558,
                0.239647,
                0.475457,
                0.98099,
                1.140676,
                6.568386,
                0.402747,
                0.031924,
                1.452355,
                1.327116,
                1.272524,
                2.206294,
                0.305659,
                4.516926,
                0.368193,
                0.32009,
                1.456166,
            ]
        )

        hb = np.array(
            [
                1.461099,
                1.016174,
                0.63723,
                1.345621,
                1.202494,
                0.160559,
                0.407208,
                1.340296,
                1.863961,
                0.21003,
                0.728607,
                1.228798,
                1.139377,
                1.641695,
                0.739283,
                1.197968,
                0.78105,
                0.640222,
                0.424996,
                1.02282,
                1.755609,
                0.398027,
                1.85161,
                0.981798,
                0.479405,
                1.117769,
                1.219365,
                1.008502,
                0.875044,
                1.7715,
                1.581884,
                0.557566,
                0.646777,
                1.66085,
                0.558587,
                0.506768,
                0.530877,
                1.410221,
                0.766308,
                0.280472,
                0.179709,
                0.711375,
                0.912691,
                1.217811,
                1.268842,
                0.869746,
                1.43425,
                0.73892,
                0.232298,
                0.091539,
                0.484148,
                0.820966,
                1.009495,
                0.612865,
                1.253926,
                2.06628,
                0.982204,
                0.609027,
                0.74364,
                1.619779,
                0.441897,
                1.412394,
                0.740567,
                1.173888,
                0.347058,
                1.017462,
                1.395372,
                0.919926,
                0.287325,
                1.478055,
                1.971756,
                0.727748,
                0.222515,
                0.589475,
                1.662847,
                0.849125,
                0.673463,
                1.477411,
                1.68667,
                0.650129,
                1.075729,
                0.296702,
                0.200924,
                0.303833,
                0.85922,
                0.916668,
                1.08823,
                0.059829,
                0.781662,
                1.035956,
                0.980043,
                0.868404,
                1.283919,
                0.685628,
                0.417871,
                0.873931,
                1.93834,
                0.5638,
                1.111664,
                1.042235,
            ]
        )

        self.assertTrue(np.allclose(a, ha, 1e-3, 1e-4))
        self.assertTrue(np.allclose(b, hb, 1e-3, 1e-4))


class Test_pcg32_gamma(unittest.TestCase):
    """
    Gamma distribution.
    """

    def test_historic(self):

        gen = prrng.pcg32()

        a = gen.gamma([100])
        b = gen.gamma([100], 2.0)

        ha = np.array(
            [
                0.114714,
                2.374727,
                0.522042,
                2.081352,
                1.186949,
                1.360627,
                0.183254,
                0.97157,
                0.250615,
                2.263344,
                0.512624,
                0.505185,
                0.426016,
                1.061255,
                1.56277,
                2.162192,
                0.957162,
                0.012659,
                0.473702,
                0.062727,
                0.274371,
                0.208131,
                0.397584,
                0.325933,
                2.115884,
                0.455096,
                0.959685,
                0.222672,
                1.81849,
                0.533491,
                1.645234,
                2.21835,
                2.368555,
                0.334346,
                0.640984,
                0.272069,
                0.405344,
                0.447168,
                0.695566,
                0.492976,
                0.697642,
                1.335018,
                1.249684,
                2.474297,
                1.95181,
                0.05866,
                0.282769,
                2.422034,
                2.826369,
                1.162903,
                0.525544,
                0.076969,
                0.033486,
                0.066933,
                0.001448,
                3.157077,
                0.695842,
                1.677995,
                0.420728,
                1.092996,
                1.766276,
                0.255873,
                1.81811,
                0.368136,
                0.203232,
                3.208611,
                0.187289,
                0.553004,
                2.582531,
                2.712604,
                0.952325,
                0.007711,
                1.214684,
                0.266872,
                1.302899,
                3.694436,
                2.709521,
                0.417277,
                1.412725,
                2.071482,
                0.168048,
                0.432525,
                1.562487,
                0.014558,
                0.239647,
                0.475457,
                0.98099,
                1.140676,
                6.568386,
                0.402747,
                0.031924,
                1.452355,
                1.327116,
                1.272524,
                2.206294,
                0.305659,
                4.516926,
                0.368193,
                0.32009,
                1.456166,
            ]
        )

        hb = np.array(
            [
                3.677594,
                2.193831,
                1.189933,
                3.259955,
                2.774173,
                0.244567,
                0.691339,
                3.241253,
                5.317708,
                0.327136,
                1.410847,
                2.860826,
                2.571036,
                4.377361,
                1.437521,
                2.759383,
                1.543632,
                1.196957,
                0.727017,
                2.213531,
                4.848318,
                0.673109,
                5.263144,
                2.093101,
                0.839113,
                2.503039,
                2.829614,
                2.171181,
                1.792718,
                4.915848,
                4.139267,
                1.008037,
                1.212396,
                4.454944,
                1.010306,
                0.897186,
                0.949301,
                3.49074,
                1.50586,
                0.450735,
                0.276119,
                1.368172,
                1.896513,
                2.824487,
                2.995005,
                1.778297,
                3.578431,
                1.436611,
                0.365437,
                0.135102,
                0.849096,
                1.647663,
                2.174108,
                1.13325,
                2.944705,
                6.250598,
                2.09428,
                1.124407,
                1.44846,
                4.289387,
                0.761357,
                3.498629,
                1.440742,
                2.681273,
                0.57419,
                2.197642,
                3.437052,
                1.916725,
                0.463142,
                3.740866,
                5.805563,
                1.408708,
                0.348523,
                1.079707,
                4.463073,
                1.722601,
                1.275949,
                3.738455,
                4.560554,
                1.220319,
                2.372993,
                0.480228,
                0.311677,
                0.493308,
                1.749779,
                1.907614,
                2.411353,
                0.087015,
                1.545207,
                2.252687,
                2.088011,
                1.774651,
                3.046235,
                1.305294,
                0.71267,
                1.789683,
                5.652095,
                1.021917,
                2.483969,
                2.271504,
            ]
        )

        self.assertTrue(np.allclose(a, ha, 1e-3, 1e-4) or np.all(np.isnan(ha)))
        self.assertTrue(np.allclose(b, hb, 1e-3, 1e-4) or np.all(np.isnan(hb)))

        if np.all(np.isnan(ha)) and np.all(np.isnan(hb)):
            print("Warning: Compile without Gamma functions, skipping check")


class Test_pcg32_delta(unittest.TestCase):
    def test_array(self):

        seed = np.arange(10).reshape([2, -1])
        gen = prrng.pcg32_array(seed)
        state = gen.state()
        a = gen.delta([4, 5])
        self.assertTrue(np.allclose(a, np.ones_like(a)))
        self.assertTrue(np.all(np.equal(gen.state(), state)))


class Test_pcg32_array(unittest.TestCase):
    def test_list(self):

        seed = np.arange(5)
        gen = prrng.pcg32_array(seed)
        state = gen.state()
        a = gen.random([4, 5])
        b = gen.random([4, 5])
        self.assertTrue(not np.allclose(a, b))

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
        self.assertTrue(not np.allclose(a, b))

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

    def test_distance(self):

        seed = np.arange(10).reshape([2, -1])
        gen = prrng.pcg32_array(seed)
        regen = prrng.pcg32_array(seed)
        state = gen.state()

        gen.random([4, 5])

        self.assertTrue(np.all(gen.distance(state) == 4 * 5 * np.ones(seed.shape)))
        self.assertTrue(np.all(gen.distance(regen) == 4 * 5 * np.ones(seed.shape)))
        self.assertTrue(np.all(regen.distance(gen) == -4 * 5 * np.ones(seed.shape)))

    def test_decide(self):

        seed = np.arange(10).reshape([2, -1])
        gen = prrng.pcg32_array(seed)
        p = gen.random([])
        state = gen.state()

        decision = gen.decide(p)
        decision2 = np.empty_like(decision)
        gen.decide(p, decision2)

        gen.restore(state)
        self.assertTrue(np.all(np.equal(decision, gen.random([]) <= p)))
        self.assertTrue(np.all(np.equal(decision2, gen.random([]) <= p)))

        p = np.ones_like(p)
        gen.decide(p, decision)
        self.assertTrue(np.all(decision))

        p = np.zeros_like(p)
        gen.decide(p, decision)
        self.assertTrue(not np.any(decision))

        mask = np.empty_like(decision)
        gen.decide(0.5 * np.ones_like(p), mask)
        p = gen.random([])
        state = gen.state()

        decision = gen.decide_masked(p, mask)
        self.assertTrue(
            np.all(np.where(mask, np.equal(state, gen.state()), np.not_equal(state, gen.state())))
        )

        gen.decide_masked(p, mask, decision)
        self.assertTrue(
            np.all(np.where(mask, np.equal(state, gen.state()), np.not_equal(state, gen.state())))
        )

    def test_randint(self):

        seed = np.arange(10).reshape([2, -1])
        gen = prrng.pcg32_array(seed)

        a = gen.randint([10, 10], 10)
        self.assertTrue(np.all(a < 10))

        high = 1000
        a = gen.randint([100000], high)
        m = np.mean(a)
        self.assertTrue(np.all(a < high))
        self.assertLess((m - (high - 1)) / (high - 1), 1e-3)

        low = 500
        high = 1000
        a = gen.randint([100000], low, high)
        m = np.mean(a)
        c = 0.5 * (high - 1 + low)
        self.assertTrue(np.all(a >= low))
        self.assertTrue(np.all(a < high))
        self.assertLess((m - c) / c, 1e-3)

    def test_cumsum_random(self):

        seed = np.arange(10).reshape([2, -1])
        gen = prrng.pcg32_array(seed)
        state = gen.state()
        n = 10000
        offset = 0.1
        mean = 2.3

        gen.restore(state)
        a = np.cumsum(gen.random([n]), axis=-1)

        gen.restore(state)
        aprime = np.cumsum(offset + mean * gen.random([n]), axis=-1)

        gen.restore(state)
        b = gen.cumsum_random(n * np.ones_like(seed))

        gen.restore(state)
        bprime = offset * n + mean * gen.cumsum_random(n * np.ones_like(seed))

        self.assertTrue(np.allclose(a[..., -1], b))
        self.assertTrue(np.allclose(aprime[..., -1], bprime))

    def test_cumsum_normal(self):

        seed = np.arange(10).reshape([2, -1])
        gen = prrng.pcg32_array(seed)
        state = gen.state()
        n = 10000
        offset = 0.1
        mean = 2.3
        mu = 1.2
        sigma = 0.5

        gen.restore(state)
        a = np.cumsum(gen.normal([n]), axis=-1)

        gen.restore(state)
        aprime = np.cumsum(offset + mean * gen.normal([n]), axis=-1)

        gen.restore(state)
        acustom = np.cumsum(offset + mean * gen.normal([n], mu, sigma), axis=-1)

        gen.restore(state)
        b = gen.cumsum_normal(n * np.ones_like(seed))

        gen.restore(state)
        bprime = offset * n + mean * gen.cumsum_normal(n * np.ones_like(seed))

        gen.restore(state)
        bcustom = offset * n + mean * gen.cumsum_normal(n * np.ones_like(seed), mu, sigma)

        self.assertTrue(np.allclose(a[..., -1], b))
        self.assertTrue(np.allclose(aprime[..., -1], bprime))
        self.assertTrue(np.allclose(acustom[..., -1], bcustom))

    def test_cumsum_weibull(self):

        seed = np.arange(10).reshape([2, -1])
        gen = prrng.pcg32_array(seed)
        state = gen.state()
        n = 10000
        offset = 0.1
        mean = 2.3
        k = 1.2
        lam = 0.5

        gen.restore(state)
        a = np.cumsum(gen.weibull([n]), axis=-1)

        gen.restore(state)
        aprime = np.cumsum(offset + mean * gen.weibull([n]), axis=-1)

        gen.restore(state)
        acustom = np.cumsum(offset + mean * gen.weibull([n], k, lam), axis=-1)

        gen.restore(state)
        b = gen.cumsum_weibull(n * np.ones_like(seed))

        gen.restore(state)
        bprime = offset * n + mean * gen.cumsum_weibull(n * np.ones_like(seed))

        gen.restore(state)
        bcustom = offset * n + mean * gen.cumsum_weibull(n * np.ones_like(seed), k, lam)

        self.assertTrue(np.allclose(a[..., -1], b))
        self.assertTrue(np.allclose(aprime[..., -1], bprime))
        self.assertTrue(np.allclose(acustom[..., -1], bcustom))

    def test_cumsum_gamma(self):

        seed = np.arange(10).reshape([2, -1])
        gen = prrng.pcg32_array(seed)
        state = gen.state()
        n = 10000
        offset = 0.1
        mean = 2.3
        k = 1.2
        theta = 0.5

        gen.restore(state)
        a = np.cumsum(gen.gamma([n]), axis=-1)

        gen.restore(state)
        aprime = np.cumsum(offset + mean * gen.gamma([n]), axis=-1)

        gen.restore(state)
        acustom = np.cumsum(offset + mean * gen.gamma([n], k, theta), axis=-1)

        gen.restore(state)
        b = gen.cumsum_gamma(n * np.ones_like(seed))

        gen.restore(state)
        bprime = offset * n + mean * gen.cumsum_gamma(n * np.ones_like(seed))

        gen.restore(state)
        bcustom = offset * n + mean * gen.cumsum_gamma(n * np.ones_like(seed), k, theta)

        self.assertTrue(np.allclose(a[..., -1], b))
        self.assertTrue(np.allclose(aprime[..., -1], bprime))
        self.assertTrue(np.allclose(acustom[..., -1], bcustom))


if __name__ == "__main__":

    unittest.main()
