
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(prrng-test-basic)
    find_package(prrng REQUIRED CONFIG)
endif()

set(ASSERT ON)
set(DEBUG ON)

set(test_name "test-basic")

find_package(Catch2 REQUIRED)

add_executable(${test_name}
    main.cpp
    pcg32.cpp
    version.cpp)

target_link_libraries(${test_name} PRIVATE
    Catch2::Catch2
    prrng
    prrng::compiler_warnings)

if(ASSERT)
    target_link_libraries(${test_name} PRIVATE prrng::assert)
endif()

if(DEBUG)
    target_link_libraries(${test_name} PRIVATE prrng::debug)
endif()

add_test(NAME ${test_name} COMMAND ${test_name})
