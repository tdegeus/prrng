
#include <catch2/catch.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <prrng.h>

template <class T>
inline xt::xtensor<uint32_t, 1> get(T& generator, size_t n)
{
    xt::xtensor<uint32_t, 1> ret = xt::empty<uint32_t>({n});

    for (size_t i = 0; i < n; ++i) {
        ret(i) = generator();
    }

    return ret;
}

TEST_CASE("prrng::pgc32", "prrng.h")
{
    SECTION("basic")
    {
        auto seed = std::time(0);

        prrng::pcg32 first(seed);
        auto a = get(first, 100);

        prrng::pcg32 second(seed);
        auto b = get(second, 100);

        REQUIRE(xt::all(xt::equal(a, b)));
    }

    SECTION("restore")
    {
        prrng::pcg32 draw(std::time(0));

        auto state = draw.state<>();
        auto a = get(draw, 100);

        draw.restore(state);
        auto b = get(draw, 100);

        REQUIRE(xt::all(xt::equal(a, b)));
    }

    SECTION("random")
    {
        prrng::pcg32 draw;

        auto a = draw.random<xt::xtensor<double, 1>>({100000});
        double m = xt::mean(a)();
        REQUIRE(std::abs((m - 0.5) / 0.5) < 1e-3);
    }
}
