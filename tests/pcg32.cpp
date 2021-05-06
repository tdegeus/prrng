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

        xt::xtensor<double, 1> b =
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

        REQUIRE(xt::allclose(a, b, 1e-3, 1e-4));
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
        auto state = gen.state<xt::xtensor<uint64_t, 1>>();
        auto a = gen.random<xt::xtensor<double, 3>>({4, 5});
        auto b = gen.random<xt::xtensor<double, 3>>({4, 5});
        REQUIRE(!xt::allclose(a, b));

        // test "restore"

        gen.restore(state);
        REQUIRE(xt::allclose(a, gen.random<xt::xtensor<double, 3>>({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random<xt::xtensor<double, 3>>({4, 5})));

        // test "operator[]"

        for (size_t i = 0; i < gen.size(); ++i) {
            gen[i].restore(state(i));
        }

        REQUIRE(xt::allclose(a, gen.random<xt::xtensor<double, 3>>({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random<xt::xtensor<double, 3>>({4, 5})));

        // test "operator()"

        for (size_t i = 0; i < gen.size(); ++i) {
            gen(i).restore(state(i));
        }

        REQUIRE(xt::allclose(a, gen.random<xt::xtensor<double, 3>>({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random<xt::xtensor<double, 3>>({4, 5})));

        // test "initstate" and "initseq"

        auto initstate = gen.initstate<xt::xtensor<uint64_t, 1>>();
        auto initseq = gen.initseq<xt::xtensor<uint64_t, 1>>();

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
        auto state = gen.state<xt::xtensor<uint64_t, 2>>();
        auto a = gen.random<xt::xtensor<double, 4>>({4, 5});
        auto b = gen.random<xt::xtensor<double, 4>>({4, 5});
        REQUIRE(!xt::allclose(a, b));

        // test "restore"

        gen.restore(state);
        REQUIRE(xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));

        // test "operator[]"

        for (size_t i = 0; i < gen.size(); ++i) {
            gen[i].restore(state.data()[i]);
        }

        REQUIRE(xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));

        // test "operator()"

        for (size_t i = 0; i < gen.shape(0); ++i) {
            for (size_t j = 0; j < gen.shape(1); ++j) {
                gen(i, j).restore(state(i, j));
            }
        }

        REQUIRE(xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));

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
        auto state = gen.state<xt::xtensor<uint64_t, 2>>();
        auto a = gen.random<xt::xtensor<double, 4>>({4, 5});
        auto b = gen.random<xt::xtensor<double, 4>>({4, 5});
        REQUIRE(!xt::allclose(a, b));

        // test "restore"

        gen.restore(state);
        REQUIRE(xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));

        // test "operator[]"

        for (size_t i = 0; i < gen.size(); ++i) {
            gen[i].restore(state.data()[i]);
        }

        REQUIRE(xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));

        // test "operator()"

        for (size_t i = 0; i < gen.shape(0); ++i) {
            for (size_t j = 0; j < gen.shape(1); ++j) {
                gen(i, j).restore(state(i, j));
            }
        }

        REQUIRE(xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));
        REQUIRE(!xt::allclose(a, gen.random<xt::xtensor<double, 4>>({4, 5})));

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
