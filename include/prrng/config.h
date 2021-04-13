/**
Basic configuration:

-   Include general dependencies.
-   Define assertions.

\file config.h
\copyright Copyright 2021. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef PRRNG_CONFIG_H
#define PRRNG_CONFIG_H

/**
\cond
*/
#define Q(x) #x
#define QUOTE(x) Q(x)

#define UNUSED(p) ((void)(p))

#define PRRNG_WARNING_IMPL(message, file, line) \
    std::cout << \
        std::string(file) + ':' + std::to_string(line) + \
        ": " message ") \n\t"; \

#define PRRNG_ASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

/**
\endcond
*/

/**
All assertions are implementation as::

    PRRNG_ASSERT(...)

They can be enabled by::

    #define PRRNG_ENABLE_ASSERT

(before including prrng).
The advantage is that:

-   File and line-number are displayed if the assertion fails.
-   prrng's assertions can be enabled/disabled independently from those of other libraries.

\throw std::runtime_error
*/
#ifdef PRRNG_ENABLE_ASSERT
#define PRRNG_ASSERT(expr) PRRNG_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define PRRNG_ASSERT(expr)
#endif

/**
Assertion that cannot be switched off. Implement assertion by::

    PRRNG_CHECK(...)

\throw std::runtime_error
*/
#define PRRNG_CHECK(expr) PRRNG_ASSERT_IMPL(expr, __FILE__, __LINE__)

/**
Assertion that concerns temporary implementation limitations.
Implement assertion by::

    PRRNG_WIP_ASSERT(...)

\throw std::runtime_error
*/
#define PRRNG_WIP_ASSERT(expr) PRRNG_ASSERT_IMPL(expr, __FILE__, __LINE__)

/**
All warnings are implemented as::

    PRRNG_WARNING(...)

They can be disabled by::

    #define PRRNG_DISABLE_WARNING
*/
#ifdef PRRNG_DISABLE_WARNING
#define PRRNG_WARNING(message)
#else
#define PRRNG_WARNING(message) PRRNG_WARNING_IMPL(message, __FILE__, __LINE__)
#endif

/**
All warnings specific to the Python API are implemented as::

    PRRNG_WARNING_PYTHON(...)

They can be enabled by::

    #define PRRNG_ENABLE_WARNING_PYTHON
*/
#ifdef PRRNG_ENABLE_WARNING_PYTHON
#define PRRNG_WARNING_PYTHON(message) PRRNG_WARNING_IMPL(message, __FILE__, __LINE__)
#else
#define PRRNG_WARNING_PYTHON(message)
#endif

/**
Toolbox to perform finite element computations.
*/
namespace prrng {}

#endif
