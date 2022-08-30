#include <catch2/catch_all.hpp>
#include <iostream>
#include <prrng.h>

TEST_CASE("prrng::version", "prrng.h")
{

    SECTION("basic")
    {
        std::cout << prrng::version() << std::endl;
    }
}
