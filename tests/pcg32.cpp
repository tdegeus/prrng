#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <prrng.h>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>

template <class T>
inline xt::xtensor<uint32_t, 1> myget_n(T& generator, size_t n)
{
    xt::xtensor<uint32_t, 1> ret = xt::empty<uint32_t>({n});

    for (size_t i = 0; i < n; ++i) {
        ret(i) = generator();
    }

    return ret;
}

TEST_CASE("prrng::pgc32", "prrng.h")
{
    SECTION("basic - seed")
    {
        auto seed = std::time(0);

        prrng::pcg32 gen_a(seed);
        auto a = myget_n(gen_a, 100);

        prrng::pcg32 gen_b(seed);
        auto b = myget_n(gen_b, 100);

        REQUIRE(xt::all(xt::equal(a, b)));
    }

    SECTION("basic - restore")
    {
        prrng::pcg32 gen(std::time(0));

        auto f = myget_n(gen, 100);

        auto state = gen.state();
        auto a = myget_n(gen, 100);

        gen.restore(state);
        auto b = myget_n(gen, 100);

        REQUIRE(xt::all(xt::equal(a, b)));
    }

    SECTION("basic - distance")
    {
        size_t n = 345;
        auto seed = std::time(0);

        prrng::pcg32 gen_a(seed);
        prrng::pcg32 gen_b(seed);

        auto a = myget_n(gen_a, n);
        auto m = gen_a - gen_b;
        auto d = gen_a.distance(gen_b);
        auto b = myget_n(gen_b, m);

        REQUIRE(static_cast<decltype(m)>(n) == m);
        REQUIRE(static_cast<decltype(d)>(n) == d);
        REQUIRE(xt::all(xt::equal(a, b)));
    }

    SECTION("basic - distance using state")
    {
        size_t n = 345;
        auto seed = std::time(0);

        prrng::pcg32 gen(seed);

        auto state = gen.state();
        auto a = myget_n(gen, n);
        auto m = gen.distance(state);

        gen.restore(state);
        auto b = myget_n(gen, m);

        REQUIRE(static_cast<decltype(m)>(n) == m);
        REQUIRE(xt::all(xt::equal(a, b)));
    }

    SECTION("random - scalar")
    {
        prrng::pcg32 generator;
        auto a = generator.random<double>(std::array<size_t, 0>{});

        generator.advance(-1);
        auto b = generator.random({1});

        REQUIRE(a == b(0));
    }

    SECTION("randint - scalar")
    {
        prrng::pcg32 generator;
        auto a = generator.randint<int>(std::array<size_t, 0>{}, 10);

        generator.advance(-1);
        auto b = generator.randint({1}, (int)10);

        REQUIRE(a == b(0));
    }

    SECTION("random - return type")
    {
        prrng::pcg32 generator;

        std::array<size_t, 2> fixed_shape = {10, 20};
        std::vector<size_t> shape = {10, 20};

        {
            auto a = generator.random({10, 20});
            auto b = generator.random(fixed_shape);
            auto c = generator.random(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 2>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 2>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, shape));
            REQUIRE(xt::has_shape(b, shape));
            REQUIRE(xt::has_shape(c, shape));
        }

        {
            auto a = generator.random<xt::xtensor<double, 2>>({10, 20});
            auto b = generator.random<xt::xtensor<double, 2>>(fixed_shape);
            auto c = generator.random<xt::xtensor<double, 2>>(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 2>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 2>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xtensor<double, 2>>::value, "X");

            REQUIRE(xt::has_shape(a, shape));
            REQUIRE(xt::has_shape(b, shape));
            REQUIRE(xt::has_shape(c, shape));
        }

        {
            auto a = generator.random<xt::xarray<double>>({10, 20});
            auto b = generator.random<xt::xarray<double>>(fixed_shape);
            auto c = generator.random<xt::xarray<double>>(shape);

            static_assert(std::is_same<decltype(a), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, shape));
            REQUIRE(xt::has_shape(b, shape));
            REQUIRE(xt::has_shape(c, shape));
        }
    }

    SECTION("weibull - return type")
    {
        prrng::pcg32 generator;

        std::array<size_t, 2> fixed_shape = {10, 20};
        std::vector<size_t> shape = {10, 20};

        {
            auto a = generator.weibull({10, 20});
            auto b = generator.weibull(fixed_shape);
            auto c = generator.weibull(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 2>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 2>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, shape));
            REQUIRE(xt::has_shape(b, shape));
            REQUIRE(xt::has_shape(c, shape));
        }

        {
            auto a = generator.weibull<xt::xtensor<double, 2>>({10, 20});
            auto b = generator.weibull<xt::xtensor<double, 2>>(fixed_shape);
            auto c = generator.weibull<xt::xtensor<double, 2>>(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 2>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 2>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xtensor<double, 2>>::value, "X");

            REQUIRE(xt::has_shape(a, shape));
            REQUIRE(xt::has_shape(b, shape));
            REQUIRE(xt::has_shape(c, shape));
        }

        {
            auto a = generator.weibull<xt::xarray<double>>({10, 20});
            auto b = generator.weibull<xt::xarray<double>>(fixed_shape);
            auto c = generator.weibull<xt::xarray<double>>(shape);

            static_assert(std::is_same<decltype(a), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, shape));
            REQUIRE(xt::has_shape(b, shape));
            REQUIRE(xt::has_shape(c, shape));
        }
    }

    SECTION("random - seed")
    {
        auto seed = std::time(0);

        prrng::pcg32 gen_a(seed);
        auto a = gen_a.random({100000});

        prrng::pcg32 gen_b(seed);
        auto b = gen_b.random({100000});

        REQUIRE(xt::all(xt::equal(a, b)));
    }

    SECTION("random - restore")
    {
        prrng::pcg32 gen(std::time(0));

        auto state = gen.state();
        auto a = gen.random({100000});

        gen.restore(state);
        auto b = gen.random({100000});

        REQUIRE(xt::all(xt::equal(a, b)));
    }

    SECTION("random - mean")
    {
        prrng::pcg32 gen;

        auto a = gen.random({100000});
        double m = xt::mean(a)();
        REQUIRE(std::abs((m - 0.5) / 0.5) < 1e-3);
    }

    SECTION("randint - mean")
    {
        prrng::pcg32 gen;

        uint32_t high = 1000;
        auto a = gen.randint({1000000}, high);
        double m = xt::mean(xt::cast<double>(a))();
        double c = 0.5 * static_cast<double>(high - 1);
        REQUIRE(xt::all(a < high));
        REQUIRE(std::abs((m - c) / c) < 1e-3);
    }

    SECTION("randint - mean - low")
    {
        prrng::pcg32 gen;

        uint32_t low = 500;
        uint32_t high = 1000;
        auto a = gen.randint({1000000}, low, high);
        double m = xt::mean(xt::cast<double>(a))();
        double c = 0.5 * (static_cast<double>(low) + static_cast<double>(high - 1));
        REQUIRE(xt::all(a >= low));
        REQUIRE(xt::all(a < high));
        REQUIRE(std::abs((m - c) / c) < 1e-3);
    }

    SECTION("random - historic")
    {
        prrng::pcg32 gen;

        auto a = gen.random({100});

        xt::xtensor<double, 1> ha = {
            0.108379, 0.90696,  0.406692, 0.875239, 0.694849, 0.7435,   0.167443, 0.621512,
            0.221678, 0.895998, 0.401078, 0.396606, 0.346894, 0.653979, 0.790445, 0.884927,
            0.616019, 0.012579, 0.377307, 0.0608,   0.23995,  0.1879,   0.328058, 0.278146,
            0.879473, 0.365613, 0.616987, 0.199623, 0.837729, 0.413446, 0.807033, 0.891212,
            0.906384, 0.284194, 0.473226, 0.238198, 0.333253, 0.360564, 0.501208, 0.389194,
            0.502242, 0.736847, 0.713405, 0.915778, 0.857983, 0.056973, 0.246306, 0.911259,
            0.940772, 0.687423, 0.408766, 0.074081, 0.032931, 0.064742, 0.001447, 0.95745,
            0.501345, 0.813252, 0.343431, 0.664789, 0.829031, 0.22576,  0.837668, 0.307977,
            0.183911, 0.959587, 0.170796, 0.424781, 0.924418, 0.933636, 0.614157, 0.007682,
            0.703196, 0.234229, 0.728257, 0.975139, 0.933431, 0.341162, 0.756521, 0.874001,
            0.154687, 0.351131, 0.790386, 0.014452, 0.213094, 0.378399, 0.62506,  0.680397,
            0.998596, 0.331519, 0.03142,  0.765982, 0.734759, 0.719876, 0.889892, 0.263362,
            0.989077, 0.308017, 0.273916, 0.766872};

        REQUIRE(xt::allclose(a, ha, 1e-3, 1e-4));
    }

    SECTION("normal - historic")
    {
        auto a = prrng::pcg32().normal({102});
        auto b = prrng::pcg32().normal({102}, 2.0);
        auto c = prrng::pcg32().normal({102}, 2.0, 2.0);

        xt::xtensor<double, 1> ha = {
            -1.23519642, 1.32226552,  -0.23606174, 1.1515092,   0.50964317,  0.654174,
            -0.96431875, 0.30945346,  -0.7665385,  1.25907233,  -0.25055791, -0.26214164,
            -0.39371966, 0.39608421,  0.80796671,  1.1999846,   0.29504106,  -2.23897327,
            -0.31256146, -1.54809412, -0.70646239, -0.88566324, -0.44528089, -0.5883569,
            1.1723582,   -0.34349613, 0.29757597,  -0.84296958, 0.985169,    -0.21868835,
            0.86701283,  1.23299688,  1.31881226,  -0.57042665, -0.06716238, -0.71211011,
            -0.43094957, -0.35695237, 0.00302745,  -0.2814196,  0.00562099,  0.63365361,
            0.56335888,  1.37721989,  1.07130241,  -1.58070225, -0.68615923, 1.34854886,
            1.56129108,  0.48855752,  -0.23071934, -1.44605251, -1.83935692, -1.51613637,
            -2.97871394, 1.72183269,  0.00337231,  0.88994379,  -0.40311648, 0.42556999,
            0.9503449,   -0.75288369, 0.98491785,  -0.50159245, -0.90056034, 1.74591675,
            -0.95102437, -0.18967778, 1.43542874,  1.50343263,  0.29017034,  -2.42370467,
            0.53361573,  -0.72499015, 0.60755019,  1.96233964,  1.50184391,  -0.40929437,
            0.69515581,  1.14551022,  -1.01653841, -0.38226839, 0.80776125,  -2.18479046,
            -0.79573114, -0.3096885,  0.3187984,   0.46880912,  2.98799707,  -0.43572254,
            -1.86032782, 0.72567679,  0.62727029,  0.58247416,  1.22595408,  -0.63301433,
            2.29305048,  -0.50148,    -0.60101079, 0.7285828,   1.18369435,  0.36896617};

        xt::xtensor<double, 1> hb = {
            0.76480358,  3.32226552, 1.76393826, 3.1515092,  2.50964317, 2.654174, // nofo
            1.03568125,  2.30945346, 1.2334615,  3.25907233, 1.74944209, 1.73785836, // nofo
            1.60628034,  2.39608421, 2.80796671, 3.1999846,  2.29504106, -0.23897327, // nofo
            1.68743854,  0.45190588, 1.29353761, 1.11433676, 1.55471911, 1.4116431, // nofo
            3.1723582,   1.65650387, 2.29757597, 1.15703042, 2.985169,   1.78131165, // nofo
            2.86701283,  3.23299688, 3.31881226, 1.42957335, 1.93283762, 1.28788989, // nofo
            1.56905043,  1.64304763, 2.00302745, 1.7185804,  2.00562099, 2.63365361, // nofo
            2.56335888,  3.37721989, 3.07130241, 0.41929775, 1.31384077, 3.34854886, // nofo
            3.56129108,  2.48855752, 1.76928066, 0.55394749, 0.16064308, 0.48386363, // nofo
            -0.97871394, 3.72183269, 2.00337231, 2.88994379, 1.59688352, 2.42556999, // nofo
            2.9503449,   1.24711631, 2.98491785, 1.49840755, 1.09943966, 3.74591675, // nofo
            1.04897563,  1.81032222, 3.43542874, 3.50343263, 2.29017034, -0.42370467, // nofo
            2.53361573,  1.27500985, 2.60755019, 3.96233964, 3.50184391, 1.59070563, // nofo
            2.69515581,  3.14551022, 0.98346159, 1.61773161, 2.80776125, -0.18479046, // nofo
            1.20426886,  1.6903115,  2.3187984,  2.46880912, 4.98799707, 1.56427746, // nofo
            0.13967218,  2.72567679, 2.62727029, 2.58247416, 3.22595408, 1.36698567, // nofo
            4.29305048,  1.49852,    1.39898921, 2.7285828,  3.18369435, 2.36896617};

        xt::xtensor<double, 1> hc = {
            -0.47039285, 4.64453103,  1.52787652,  4.30301839,  3.01928635,  3.30834801,
            0.0713625,   2.61890692,  0.466923,    4.51814466,  1.49888419,  1.47571672,
            1.21256067,  2.79216842,  3.61593341,  4.39996919,  2.59008212,  -2.47794653,
            1.37487709,  -1.09618825, 0.58707522,  0.22867353,  1.10943822,  0.8232862,
            4.3447164,   1.31300774,  2.59515194,  0.31406084,  3.97033801,  1.5626233,
            3.73402566,  4.46599376,  4.63762453,  0.8591467,   1.86567523,  0.57577977,
            1.13810085,  1.28609526,  2.00605489,  1.43716079,  2.01124198,  3.26730721,
            3.12671776,  4.75443978,  4.14260483,  -1.1614045,  0.62768154,  4.69709773,
            5.12258217,  2.97711503,  1.53856132,  -0.89210502, -1.67871384, -1.03227275,
            -3.95742788, 5.44366538,  2.00674462,  3.77988757,  1.19376704,  2.85113998,
            3.9006898,   0.49423262,  3.96983569,  0.99681509,  0.19887933,  5.4918335,
            0.09795126,  1.62064443,  4.87085747,  5.00686525,  2.58034068,  -2.84740934,
            3.06723145,  0.5500197,   3.21510038,  5.92467929,  5.00368782,  1.18141127,
            3.39031162,  4.29102044,  -0.03307682, 1.23546322,  3.6155225,   -2.36958092,
            0.40853772,  1.38062299,  2.63759679,  2.93761823,  7.97599415,  1.12855491,
            -1.72065565, 3.45135358,  3.25454058,  3.16494832,  4.45190816,  0.73397134,
            6.58610095,  0.99704001,  0.79797842,  3.45716561,  4.3673887,   2.73793233};

        REQUIRE(xt::allclose(a, ha, 1e-3, 1e-4));
        REQUIRE(xt::allclose(b, hb, 1e-3, 1e-4));
        REQUIRE(xt::allclose(c, hc, 1e-3, 1e-4));
    }

    SECTION("weibull - historic")
    {
        prrng::pcg32 gen;

        auto a = gen.weibull({100});
        auto b = gen.weibull({100}, 2.0);

        xt::xtensor<double, 1> ha = {
            0.114714, 2.374727, 0.522042, 2.081352, 1.186949, 1.360627, 0.183254, 0.97157,
            0.250615, 2.263344, 0.512624, 0.505185, 0.426016, 1.061255, 1.56277,  2.162192,
            0.957162, 0.012659, 0.473702, 0.062727, 0.274371, 0.208131, 0.397584, 0.325933,
            2.115884, 0.455096, 0.959685, 0.222672, 1.81849,  0.533491, 1.645234, 2.21835,
            2.368555, 0.334346, 0.640984, 0.272069, 0.405344, 0.447168, 0.695566, 0.492976,
            0.697642, 1.335018, 1.249684, 2.474297, 1.95181,  0.05866,  0.282769, 2.422034,
            2.826369, 1.162903, 0.525544, 0.076969, 0.033486, 0.066933, 0.001448, 3.157077,
            0.695842, 1.677995, 0.420728, 1.092996, 1.766276, 0.255873, 1.81811,  0.368136,
            0.203232, 3.208611, 0.187289, 0.553004, 2.582531, 2.712604, 0.952325, 0.007711,
            1.214684, 0.266872, 1.302899, 3.694436, 2.709521, 0.417277, 1.412725, 2.071482,
            0.168048, 0.432525, 1.562487, 0.014558, 0.239647, 0.475457, 0.98099,  1.140676,
            6.568386, 0.402747, 0.031924, 1.452355, 1.327116, 1.272524, 2.206294, 0.305659,
            4.516926, 0.368193, 0.32009,  1.456166};

        xt::xtensor<double, 1> hb = {
            1.461099, 1.016174, 0.63723,  1.345621, 1.202494, 0.160559, 0.407208, 1.340296,
            1.863961, 0.21003,  0.728607, 1.228798, 1.139377, 1.641695, 0.739283, 1.197968,
            0.78105,  0.640222, 0.424996, 1.02282,  1.755609, 0.398027, 1.85161,  0.981798,
            0.479405, 1.117769, 1.219365, 1.008502, 0.875044, 1.7715,   1.581884, 0.557566,
            0.646777, 1.66085,  0.558587, 0.506768, 0.530877, 1.410221, 0.766308, 0.280472,
            0.179709, 0.711375, 0.912691, 1.217811, 1.268842, 0.869746, 1.43425,  0.73892,
            0.232298, 0.091539, 0.484148, 0.820966, 1.009495, 0.612865, 1.253926, 2.06628,
            0.982204, 0.609027, 0.74364,  1.619779, 0.441897, 1.412394, 0.740567, 1.173888,
            0.347058, 1.017462, 1.395372, 0.919926, 0.287325, 1.478055, 1.971756, 0.727748,
            0.222515, 0.589475, 1.662847, 0.849125, 0.673463, 1.477411, 1.68667,  0.650129,
            1.075729, 0.296702, 0.200924, 0.303833, 0.85922,  0.916668, 1.08823,  0.059829,
            0.781662, 1.035956, 0.980043, 0.868404, 1.283919, 0.685628, 0.417871, 0.873931,
            1.93834,  0.5638,   1.111664, 1.042235};

        REQUIRE(xt::allclose(a, ha, 1e-3, 1e-4));
        REQUIRE(xt::allclose(b, hb, 1e-3, 1e-4));
    }

    SECTION("gamma - historic")
    {
        prrng::pcg32 gen;

        auto a = gen.gamma({100});
        auto b = gen.gamma({100}, 2.0);

        xt::xtensor<double, 1> ha = {
            0.114714, 2.374727, 0.522042, 2.081352, 1.186949, 1.360627, 0.183254, 0.97157,
            0.250615, 2.263344, 0.512624, 0.505185, 0.426016, 1.061255, 1.56277,  2.162192,
            0.957162, 0.012659, 0.473702, 0.062727, 0.274371, 0.208131, 0.397584, 0.325933,
            2.115884, 0.455096, 0.959685, 0.222672, 1.81849,  0.533491, 1.645234, 2.21835,
            2.368555, 0.334346, 0.640984, 0.272069, 0.405344, 0.447168, 0.695566, 0.492976,
            0.697642, 1.335018, 1.249684, 2.474297, 1.95181,  0.05866,  0.282769, 2.422034,
            2.826369, 1.162903, 0.525544, 0.076969, 0.033486, 0.066933, 0.001448, 3.157077,
            0.695842, 1.677995, 0.420728, 1.092996, 1.766276, 0.255873, 1.81811,  0.368136,
            0.203232, 3.208611, 0.187289, 0.553004, 2.582531, 2.712604, 0.952325, 0.007711,
            1.214684, 0.266872, 1.302899, 3.694436, 2.709521, 0.417277, 1.412725, 2.071482,
            0.168048, 0.432525, 1.562487, 0.014558, 0.239647, 0.475457, 0.98099,  1.140676,
            6.568386, 0.402747, 0.031924, 1.452355, 1.327116, 1.272524, 2.206294, 0.305659,
            4.516926, 0.368193, 0.32009,  1.456166};

        xt::xtensor<double, 1> hb = {
            3.677594, 2.193831, 1.189933, 3.259955, 2.774173, 0.244567, 0.691339, 3.241253,
            5.317708, 0.327136, 1.410847, 2.860826, 2.571036, 4.377361, 1.437521, 2.759383,
            1.543632, 1.196957, 0.727017, 2.213531, 4.848318, 0.673109, 5.263144, 2.093101,
            0.839113, 2.503039, 2.829614, 2.171181, 1.792718, 4.915848, 4.139267, 1.008037,
            1.212396, 4.454944, 1.010306, 0.897186, 0.949301, 3.49074,  1.50586,  0.450735,
            0.276119, 1.368172, 1.896513, 2.824487, 2.995005, 1.778297, 3.578431, 1.436611,
            0.365437, 0.135102, 0.849096, 1.647663, 2.174108, 1.13325,  2.944705, 6.250598,
            2.09428,  1.124407, 1.44846,  4.289387, 0.761357, 3.498629, 1.440742, 2.681273,
            0.57419,  2.197642, 3.437052, 1.916725, 0.463142, 3.740866, 5.805563, 1.408708,
            0.348523, 1.079707, 4.463073, 1.722601, 1.275949, 3.738455, 4.560554, 1.220319,
            2.372993, 0.480228, 0.311677, 0.493308, 1.749779, 1.907614, 2.411353, 0.087015,
            1.545207, 2.252687, 2.088011, 1.774651, 3.046235, 1.305294, 0.71267,  1.789683,
            5.652095, 1.021917, 2.483969, 2.271504};

        REQUIRE((xt::allclose(a, ha, 1e-3, 1e-4) || xt::all(xt::isnan(ha))));
        REQUIRE((xt::allclose(b, hb, 1e-3, 1e-4) || xt::all(xt::isnan(hb))));

        if (xt::all(xt::isnan(a)) && xt::all(xt::isnan(hb))) {
            std::cout << "Warning: Compile without Gamma functions, skipping check" << std::endl;
        }
    }

    SECTION("pcg32_array - return type")
    {
        xt::xtensor<uint64_t, 2> seed = {{0, 1, 2}, {3, 4, 5}};
        prrng::pcg32_array generators(seed);

        auto state = generators.state();
        auto initstate = generators.initstate();
        auto initseq = generators.initseq();

        static_assert(std::is_same<decltype(state), xt::xarray<uint64_t>>::value, "X");
        static_assert(std::is_same<decltype(initstate), xt::xarray<uint64_t>>::value, "X");
        static_assert(std::is_same<decltype(initseq), xt::xarray<uint64_t>>::value, "X");

        std::array<size_t, 2> fixed_shape = {4, 5};
        std::vector<size_t> shape = {4, 5};
        std::vector<size_t> ret_shape = {2, 3, 4, 5};

        // random

        {
            auto a = generators.random({4, 5});
            auto b = generators.random(fixed_shape);
            auto c = generators.random(shape);

            static_assert(std::is_same<decltype(a), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        {
            auto a = generators.random<xt::xarray<double>>({4, 5});
            auto b = generators.random<xt::xarray<double>>(fixed_shape);
            auto c = generators.random<xt::xarray<double>>(shape);

            static_assert(std::is_same<decltype(a), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        {
            auto a = generators.random<xt::xtensor<double, 4>>({4, 5});
            auto b = generators.random<xt::xtensor<double, 4>>(fixed_shape);
            auto c = generators.random<xt::xtensor<double, 4>>(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xtensor<double, 4>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        // weibull

        {
            auto a = generators.weibull({4, 5});
            auto b = generators.weibull(fixed_shape);
            auto c = generators.weibull(shape);

            static_assert(std::is_same<decltype(a), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        {
            auto a = generators.weibull<xt::xarray<double>>({4, 5});
            auto b = generators.weibull<xt::xarray<double>>(fixed_shape);
            auto c = generators.weibull<xt::xarray<double>>(shape);

            static_assert(std::is_same<decltype(a), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        {
            auto a = generators.weibull<xt::xtensor<double, 4>>({4, 5});
            auto b = generators.weibull<xt::xtensor<double, 4>>(fixed_shape);
            auto c = generators.weibull<xt::xtensor<double, 4>>(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xtensor<double, 4>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }
    }

    SECTION("pcg32_tensor - return type")
    {
        xt::xtensor<uint64_t, 2> seed = {{0, 1, 2}, {3, 4, 5}};
        prrng::pcg32_tensor<2> generators(seed);

        auto state = generators.state();
        auto initstate = generators.initstate();
        auto initseq = generators.initseq();

        static_assert(std::is_same<decltype(state), xt::xtensor<uint64_t, 2>>::value, "X");
        static_assert(std::is_same<decltype(initstate), xt::xtensor<uint64_t, 2>>::value, "X");
        static_assert(std::is_same<decltype(initseq), xt::xtensor<uint64_t, 2>>::value, "X");

        std::array<size_t, 2> fixed_shape = {4, 5};
        std::vector<size_t> shape = {4, 5};
        std::vector<size_t> ret_shape = {2, 3, 4, 5};

        // random

        {
            auto a = generators.random({4, 5});
            auto b = generators.random(fixed_shape);
            auto c = generators.random(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        {
            auto a = generators.random<xt::xarray<double>>({4, 5});
            auto b = generators.random<xt::xarray<double>>(fixed_shape);
            auto c = generators.random<xt::xarray<double>>(shape);

            static_assert(std::is_same<decltype(a), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        {
            auto a = generators.random<xt::xtensor<double, 4>>({4, 5});
            auto b = generators.random<xt::xtensor<double, 4>>(fixed_shape);
            auto c = generators.random<xt::xtensor<double, 4>>(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xtensor<double, 4>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        // weibull

        {
            auto a = generators.weibull({4, 5});
            auto b = generators.weibull(fixed_shape);
            auto c = generators.weibull(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        {
            auto a = generators.weibull<xt::xarray<double>>({4, 5});
            auto b = generators.weibull<xt::xarray<double>>(fixed_shape);
            auto c = generators.weibull<xt::xarray<double>>(shape);

            static_assert(std::is_same<decltype(a), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xarray<double>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xarray<double>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }

        {
            auto a = generators.weibull<xt::xtensor<double, 4>>({4, 5});
            auto b = generators.weibull<xt::xtensor<double, 4>>(fixed_shape);
            auto c = generators.weibull<xt::xtensor<double, 4>>(shape);

            static_assert(std::is_same<decltype(a), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(b), xt::xtensor<double, 4>>::value, "X");
            static_assert(std::is_same<decltype(c), xt::xtensor<double, 4>>::value, "X");

            REQUIRE(xt::has_shape(a, ret_shape));
            REQUIRE(xt::has_shape(b, ret_shape));
            REQUIRE(xt::has_shape(c, ret_shape));
        }
    }

    SECTION("pcg32_array - list")
    {
        xt::xtensor<uint64_t, 1> seed = {0, 1, 2, 3, 4};
        prrng::pcg32_array gen(seed);
        auto state = gen.state();
        auto a = gen.random({4, 5});
        auto b = gen.random({4, 5});
        REQUIRE(!xt::allclose(a, b));

        // test "restore"

        gen.restore(state);
        REQUIRE(xt::allclose(a, gen.random({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random({4, 5})));

        // test "operator[]"

        for (size_t i = 0; i < gen.size(); ++i) {
            gen[i].restore(state(i));
        }

        REQUIRE(xt::allclose(a, gen.random({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random({4, 5})));

        // test "operator()"

        for (size_t i = 0; i < gen.size(); ++i) {
            gen(i).restore(state(i));
        }

        REQUIRE(xt::allclose(a, gen.random({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random({4, 5})));

        // test "initstate" and "initseq"

        auto initstate = gen.initstate();
        auto initseq = gen.initseq();

        for (size_t i = 0; i < gen.size(); ++i) {
            REQUIRE(gen[i].initstate() == initstate(i));
            REQUIRE(gen[i].initseq() == initseq(i));
        }

        for (size_t i = 0; i < gen.size(); ++i) {
            REQUIRE(gen(i).initstate() == initstate(i));
            REQUIRE(gen(i).initseq() == initseq(i));
        }
    }

    SECTION("pcg32_array - matrix")
    {
        xt::xtensor<uint64_t, 2> seed = {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}};
        prrng::pcg32_array gen(seed);
        auto state = gen.state();
        auto a = gen.random({4, 5});
        auto b = gen.random({4, 5});
        REQUIRE(!xt::allclose(a, b));

        // test "restore"

        gen.restore(state);
        REQUIRE(xt::allclose(a, gen.random({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random({4, 5})));

        // test "operator[]"

        for (size_t i = 0; i < gen.size(); ++i) {
            gen[i].restore(state.data()[i]);
        }

        REQUIRE(xt::allclose(a, gen.random({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random({4, 5})));

        // test "operator()"

        for (size_t i = 0; i < gen.shape(0); ++i) {
            for (size_t j = 0; j < gen.shape(1); ++j) {
                gen(i, j).restore(state(i, j));
            }
        }

        REQUIRE(xt::allclose(a, gen.random({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random({4, 5})));

        // test "flat_index"

        for (size_t i = 0; i < gen.shape(0); ++i) {
            for (size_t j = 0; j < gen.shape(1); ++j) {
                std::vector<size_t> index = {i, j};
                REQUIRE(gen[gen.flat_index(index)] == gen(i, j));
            }
        }

        // test "inbounds"

        for (size_t i = 0; i < 2 * gen.shape(0); ++i) {
            for (size_t j = 0; j < 2 * gen.shape(1); ++j) {
                std::vector<size_t> index = {i, j};
                REQUIRE(gen.inbounds(index) == (i < gen.shape(0) && j < gen.shape(1)));
            }
        }
    }

    SECTION("pcg32_array - matrix - state/restore/advance/distance")
    {
        xt::xtensor<uint64_t, 2> seed = {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}};

        prrng::pcg32_array gen(seed);
        prrng::pcg32_array regen(seed);

        auto i = gen.state();
        auto a = gen.random({4, 5});
        auto b = gen.random({4, 5});
        auto c = gen.random({4, 5});
        auto s = gen.state();
        auto e = gen.random({4, 5});
        xt::xtensor<int64_t, 2> unit = xt::ones<int64_t>(seed.shape());

        REQUIRE(xt::all(xt::equal(regen.distance(gen), -4 * 4 * 5 * unit)));
        REQUIRE(xt::all(xt::equal(gen.distance(regen), 4 * 4 * 5 * unit)));
        regen.restore(s);
        REQUIRE(xt::all(xt::equal(regen.distance(i), 3 * 4 * 5 * unit)));
        REQUIRE(xt::all(xt::equal(regen.random({4, 5}), e)));
        REQUIRE(xt::all(xt::equal(regen.distance(gen), 0 * unit)));
        REQUIRE(xt::all(xt::equal(gen.distance(regen), 0 * unit)));

        regen.restore(s);
        regen.advance(xt::eval(-4 * 5 * xt::ones<int>(regen.shape())));
        REQUIRE(xt::allclose(c, regen.random({4, 5})));

        regen.restore(s);
        regen.advance(xt::eval(-2 * 4 * 5 * xt::ones<int>(regen.shape())));
        REQUIRE(xt::allclose(b, regen.random({4, 5})));

        regen.restore(s);
        regen.advance(xt::eval(-3 * 4 * 5 * xt::ones<int>(regen.shape())));
        REQUIRE(xt::allclose(a, regen.random({4, 5})));
    }

    SECTION("pcg32_tensor - matrix")
    {
        xt::xtensor<uint64_t, 2> seed = {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}};
        prrng::pcg32_tensor<2> gen(seed);
        auto state = gen.state();
        auto a = gen.random({4, 5});
        auto b = gen.random({4, 5});
        REQUIRE(!xt::allclose(a, b));

        // test "restore"

        gen.restore(state);
        REQUIRE(xt::allclose(a, gen.random({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random({4, 5})));

        // test "operator[]"

        for (size_t i = 0; i < gen.size(); ++i) {
            gen[i].restore(state.data()[i]);
        }

        REQUIRE(xt::allclose(a, gen.random({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random({4, 5})));

        // test "operator()"

        for (size_t i = 0; i < gen.shape(0); ++i) {
            for (size_t j = 0; j < gen.shape(1); ++j) {
                gen(i, j).restore(state(i, j));
            }
        }

        REQUIRE(xt::allclose(a, gen.random({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random({4, 5})));

        // test "flat_index"

        for (size_t i = 0; i < gen.shape(0); ++i) {
            for (size_t j = 0; j < gen.shape(1); ++j) {
                std::vector<size_t> index = {i, j};
                REQUIRE(gen[gen.flat_index(index)] == gen(i, j));
            }
        }

        // test "inbounds"

        for (size_t i = 0; i < 2 * gen.shape(0); ++i) {
            for (size_t j = 0; j < 2 * gen.shape(1); ++j) {
                std::vector<size_t> index = {i, j};
                REQUIRE(gen.inbounds(index) == (i < gen.shape(0) && j < gen.shape(1)));
            }
        }
    }

    SECTION("auto_pcg32 - type")
    {
        auto a = prrng::auto_pcg32(0);
        auto b = prrng::auto_pcg32(xt::xarray<uint64_t>{0, 1, 2});
        auto c = prrng::auto_pcg32(xt::xtensor<uint64_t, 1>{0, 1, 2});
        auto d = prrng::auto_pcg32(xt::xtensor<uint64_t, 2>{{0, 1, 2}, {3, 4, 5}});

        static_assert(std::is_same<decltype(a), prrng::pcg32>::value, "X");
        static_assert(std::is_same<decltype(b), prrng::pcg32_array>::value, "X");
        static_assert(std::is_same<decltype(c), prrng::pcg32_tensor<1>>::value, "X");
        static_assert(std::is_same<decltype(d), prrng::pcg32_tensor<2>>::value, "X");

        REQUIRE(xt::has_shape(b.state(), {3}));
        REQUIRE(xt::has_shape(c.state(), {3}));
        REQUIRE(xt::has_shape(d.state(), {2, 3}));
    }

    SECTION("auto_pcg32 - pcg32")
    {
        auto seed = std::time(0);

        {
            auto a = prrng::auto_pcg32(seed);
            auto b = prrng::pcg32(seed);
            REQUIRE(xt::allclose(a.random({10, 20}), b.random({10, 20})));
        }

        {
            auto a = prrng::auto_pcg32(seed, seed + 1);
            auto b = prrng::pcg32(seed, seed + 1);
            REQUIRE(xt::allclose(a.random({10, 20}), b.random({10, 20})));
        }
    }

    SECTION("auto_pcg32 - pcg32_array")
    {
        auto s = std::time(0);
        xt::xarray<size_t> seed = s + xt::arange<size_t>(20);

        {
            auto a = prrng::auto_pcg32(seed);
            auto b = prrng::pcg32_array(seed);
            REQUIRE(xt::allclose(a.random({10, 20}), b.random({10, 20})));
        }

        {
            auto a = prrng::auto_pcg32(seed, xt::eval(seed + 1));
            auto b = prrng::pcg32_array(seed, xt::eval(seed + 1));
            REQUIRE(xt::allclose(a.random({10, 20}), b.random({10, 20})));
        }
    }

    SECTION("auto_pcg32 - pcg32_tensor")
    {
        auto s = std::time(0);
        xt::xtensor<size_t, 1> seed = s + xt::arange<size_t>(20);

        {
            auto a = prrng::auto_pcg32(seed);
            auto b = prrng::pcg32_tensor<1>(seed);
            REQUIRE(xt::allclose(a.random({10, 20}), b.random({10, 20})));
        }

        {
            auto a = prrng::auto_pcg32(seed, xt::eval(seed + 1));
            auto b = prrng::pcg32_tensor<1>(seed, xt::eval(seed + 1));
            REQUIRE(xt::allclose(a.random({10, 20}), b.random({10, 20})));
        }
    }
}
