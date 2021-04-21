
#include <catch2/catch.hpp>
#include <prrng.h>
#include <iostream>

TEST_CASE("prrng::version", "prrng.h")
{

    SECTION("basic")
    {
        std::cout << prrng::version() << std::endl;
    }

}
