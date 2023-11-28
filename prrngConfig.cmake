# prrng cmake module
#
# This module sets the target:
#
#   prrng
#
# In addition, it sets the following variables:
#
#   prrng_FOUND - true if prrng found
#   prrng_VERSION - prrng's version
#   prrng_INCLUDE_DIRS - the directory containing prrng headers
#
# The following support targets are defined to simplify things:
#
#   prrng::compiler_warnings - enable compiler warnings
#   prrng::assert - enable prrng assertions
#   prrng::debug - enable all assertions (slow)

include(CMakeFindDependencyMacro)

# Define target "prrng"

if(NOT TARGET prrng)
    include("${CMAKE_CURRENT_LIST_DIR}/prrngTargets.cmake")
    get_target_property(prrng_INCLUDE_DIRS prrng INTERFACE_INCLUDE_DIRECTORIES)
endif()

# Find dependencies

find_dependency(xtensor REQUIRED)
find_dependency(Boost REQUIRED COMPONENTS math)

# Define support target "prrng::compiler_warnings"

if(NOT TARGET prrng::compiler_warnings)
    add_library(prrng::compiler_warnings INTERFACE IMPORTED)
    if(MSVC)
        set_property(
            TARGET prrng::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            /W4)
    else()
        set_property(
            TARGET prrng::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            -Wall -Wextra -pedantic -Wno-unknown-pragmas)
    endif()
endif()

# Define support target "prrng::assert"

if(NOT TARGET prrng::assert)
    add_library(prrng::assert INTERFACE IMPORTED)
    set_property(
        TARGET prrng::assert
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        PRRNG_ENABLE_ASSERT)
endif()

# Define support target "prrng::debug"

if(NOT TARGET prrng::debug)
    add_library(prrng::debug INTERFACE IMPORTED)
    set_property(
        TARGET prrng::debug
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        XTENSOR_ENABLE_ASSERT
        PRRNG_ENABLE_ASSERT
        PRRNG_ENABLE_DEBUG)
endif()
