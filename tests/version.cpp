#define CATCH_CONFIG_MAIN // tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <iostream>
#include <prrng.h>

TEST_CASE("prrng::version", "prrng.h")
{

    SECTION("basic")
    {
        std::cout << prrng::version() << std::endl;
    }
}
