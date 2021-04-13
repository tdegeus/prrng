
#include <catch2/catch.hpp>
#include <prrng/version.h>
#include <iostream>

TEST_CASE("prrng::version", "version.h")
{

    SECTION("basic")
    {
        std::cout << prrng::version() << std::endl;

        auto deps = prrng::version_dependencies();

        for (auto& i : deps) {
            std::cout << i << std::endl;
        }
    }

}
