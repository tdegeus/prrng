#define CATCH_CONFIG_MAIN // tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <prrng.h>

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

    SECTION("random - historic")
    {
        prrng::pcg32 gen;

        auto a = gen.random({100});

        xt::xtensor<double, 1> ha =
            { 0.108379,  0.90696 ,  0.406692,  0.875239,  0.694849,  0.7435  ,
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
              0.989077,  0.308017,  0.273916,  0.766872};

        REQUIRE(xt::allclose(a, ha, 1e-3, 1e-4));
    }

    SECTION("weibull - historic")
    {
        prrng::pcg32 gen;

        auto a = gen.weibull({100});
        auto b = gen.weibull({100}, 2.0);

        xt::xtensor<double, 1> ha =
            { 0.114714,  2.374727,  0.522042,  2.081352,  1.186949,  1.360627,
              0.183254,  0.97157 ,  0.250615,  2.263344,  0.512624,  0.505185,
              0.426016,  1.061255,  1.56277 ,  2.162192,  0.957162,  0.012659,
              0.473702,  0.062727,  0.274371,  0.208131,  0.397584,  0.325933,
              2.115884,  0.455096,  0.959685,  0.222672,  1.81849 ,  0.533491,
              1.645234,  2.21835 ,  2.368555,  0.334346,  0.640984,  0.272069,
              0.405344,  0.447168,  0.695566,  0.492976,  0.697642,  1.335018,
              1.249684,  2.474297,  1.95181 ,  0.05866 ,  0.282769,  2.422034,
              2.826369,  1.162903,  0.525544,  0.076969,  0.033486,  0.066933,
              0.001448,  3.157077,  0.695842,  1.677995,  0.420728,  1.092996,
              1.766276,  0.255873,  1.81811 ,  0.368136,  0.203232,  3.208611,
              0.187289,  0.553004,  2.582531,  2.712604,  0.952325,  0.007711,
              1.214684,  0.266872,  1.302899,  3.694436,  2.709521,  0.417277,
              1.412725,  2.071482,  0.168048,  0.432525,  1.562487,  0.014558,
              0.239647,  0.475457,  0.98099 ,  1.140676,  6.568386,  0.402747,
              0.031924,  1.452355,  1.327116,  1.272524,  2.206294,  0.305659,
              4.516926,  0.368193,  0.32009 ,  1.456166};

        xt::xtensor<double, 1> hb =
            { 1.461099,  1.016174,  0.63723 ,  1.345621,  1.202494,  0.160559,
              0.407208,  1.340296,  1.863961,  0.21003 ,  0.728607,  1.228798,
              1.139377,  1.641695,  0.739283,  1.197968,  0.78105 ,  0.640222,
              0.424996,  1.02282 ,  1.755609,  0.398027,  1.85161 ,  0.981798,
              0.479405,  1.117769,  1.219365,  1.008502,  0.875044,  1.7715  ,
              1.581884,  0.557566,  0.646777,  1.66085 ,  0.558587,  0.506768,
              0.530877,  1.410221,  0.766308,  0.280472,  0.179709,  0.711375,
              0.912691,  1.217811,  1.268842,  0.869746,  1.43425 ,  0.73892 ,
              0.232298,  0.091539,  0.484148,  0.820966,  1.009495,  0.612865,
              1.253926,  2.06628 ,  0.982204,  0.609027,  0.74364 ,  1.619779,
              0.441897,  1.412394,  0.740567,  1.173888,  0.347058,  1.017462,
              1.395372,  0.919926,  0.287325,  1.478055,  1.971756,  0.727748,
              0.222515,  0.589475,  1.662847,  0.849125,  0.673463,  1.477411,
              1.68667 ,  0.650129,  1.075729,  0.296702,  0.200924,  0.303833,
              0.85922 ,  0.916668,  1.08823 ,  0.059829,  0.781662,  1.035956,
              0.980043,  0.868404,  1.283919,  0.685628,  0.417871,  0.873931,
              1.93834 ,  0.5638  ,  1.111664,  1.042235};

        REQUIRE(xt::allclose(a, ha, 1e-3, 1e-4));
        REQUIRE(xt::allclose(b, hb, 1e-3, 1e-4));
    }

    SECTION("gamma - historic")
    {
        prrng::pcg32 gen;

        auto a = gen.gamma({100});
        auto b = gen.gamma({100}, 2.0);

        xt::xtensor<double, 1> ha =
            { 0.114714,  2.374727,  0.522042,  2.081352,  1.186949,  1.360627,
              0.183254,  0.97157 ,  0.250615,  2.263344,  0.512624,  0.505185,
              0.426016,  1.061255,  1.56277 ,  2.162192,  0.957162,  0.012659,
              0.473702,  0.062727,  0.274371,  0.208131,  0.397584,  0.325933,
              2.115884,  0.455096,  0.959685,  0.222672,  1.81849 ,  0.533491,
              1.645234,  2.21835 ,  2.368555,  0.334346,  0.640984,  0.272069,
              0.405344,  0.447168,  0.695566,  0.492976,  0.697642,  1.335018,
              1.249684,  2.474297,  1.95181 ,  0.05866 ,  0.282769,  2.422034,
              2.826369,  1.162903,  0.525544,  0.076969,  0.033486,  0.066933,
              0.001448,  3.157077,  0.695842,  1.677995,  0.420728,  1.092996,
              1.766276,  0.255873,  1.81811 ,  0.368136,  0.203232,  3.208611,
              0.187289,  0.553004,  2.582531,  2.712604,  0.952325,  0.007711,
              1.214684,  0.266872,  1.302899,  3.694436,  2.709521,  0.417277,
              1.412725,  2.071482,  0.168048,  0.432525,  1.562487,  0.014558,
              0.239647,  0.475457,  0.98099 ,  1.140676,  6.568386,  0.402747,
              0.031924,  1.452355,  1.327116,  1.272524,  2.206294,  0.305659,
              4.516926,  0.368193,  0.32009 ,  1.456166};

        xt::xtensor<double, 1> hb =
            { 3.677594,  2.193831,  1.189933,  3.259955,  2.774173,  0.244567,
              0.691339,  3.241253,  5.317708,  0.327136,  1.410847,  2.860826,
              2.571036,  4.377361,  1.437521,  2.759383,  1.543632,  1.196957,
              0.727017,  2.213531,  4.848318,  0.673109,  5.263144,  2.093101,
              0.839113,  2.503039,  2.829614,  2.171181,  1.792718,  4.915848,
              4.139267,  1.008037,  1.212396,  4.454944,  1.010306,  0.897186,
              0.949301,  3.49074 ,  1.50586 ,  0.450735,  0.276119,  1.368172,
              1.896513,  2.824487,  2.995005,  1.778297,  3.578431,  1.436611,
              0.365437,  0.135102,  0.849096,  1.647663,  2.174108,  1.13325 ,
              2.944705,  6.250598,  2.09428 ,  1.124407,  1.44846 ,  4.289387,
              0.761357,  3.498629,  1.440742,  2.681273,  0.57419 ,  2.197642,
              3.437052,  1.916725,  0.463142,  3.740866,  5.805563,  1.408708,
              0.348523,  1.079707,  4.463073,  1.722601,  1.275949,  3.738455,
              4.560554,  1.220319,  2.372993,  0.480228,  0.311677,  0.493308,
              1.749779,  1.907614,  2.411353,  0.087015,  1.545207,  2.252687,
              2.088011,  1.774651,  3.046235,  1.305294,  0.71267 ,  1.789683,
              5.652095,  1.021917,  2.483969,  2.271504};

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

    SECTION("pcg32_array - matrix - state/restore/advance")
    {
        xt::xtensor<uint64_t, 2> seed = {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}};

        prrng::pcg32_array gen(seed);
        prrng::pcg32_array regen(seed);

        auto a = gen.random({4, 5});
        auto b = gen.random({4, 5});
        auto c = gen.random({4, 5});
        auto s = gen.state();
        auto d = gen.random({4, 5});

        regen.restore(s);
        REQUIRE(xt::allclose(d, regen.random({4, 5})));

        regen.restore(s);
        regen.advance(xt::eval(- 4 * 5 * xt::ones<int>(regen.shape())));
        REQUIRE(xt::allclose(c, regen.random({4, 5})));

        regen.restore(s);
        regen.advance(xt::eval(- 2 * 4 * 5 * xt::ones<int>(regen.shape())));
        REQUIRE(xt::allclose(b, regen.random({4, 5})));

        regen.restore(s);
        regen.advance(xt::eval(- 3 * 4 * 5 * xt::ones<int>(regen.shape())));
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
