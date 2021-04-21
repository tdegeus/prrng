/**
Portable Reconstructible Random Number Generator.
The idea is that a random sequence can be restored independent of platform or compiler.
In addition, this library allows you to store a point in the sequence, and then later restore
the sequence exactly from this point (in both directions actually).

Note that the core of this code is taken from
https://github.com/imneme/pcg-c-basic
All the credits goes to those developers.
This is just a wrapper.

\file prrng.h
\copyright Copyright 2021. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef PRRNG_H
#define PRRNG_H

/**
Multiplicative factor for pcg32
*/
#define PRRNG_PCG32_MULT 6364136223846793005ULL

#include <array>
#include <xtensor/xtensor.hpp>

/**
\cond
*/
#define Q(x) #x
#define QUOTE(x) Q(x)

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
Library version.

Either:

-   Configure using CMake at install time. Internally uses::

        python -c "from setuptools_scm import get_version; print(get_version())"

-   Define externally using::

        -DPRRNG_VERSION="`python -c "from setuptools_scm import get_version; print(get_version())"`"

    From the root of this project. This is what ``setup.py`` does.

Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using ``setuptools_scm``.
Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION``
to overwrite the automatic version.
*/
#ifndef PRRNG_VERSION
#define PRRNG_VERSION "@prrng_VERSION@"
#endif

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
Portable Reconstructible (Approximate!) Random Number Generator
*/
namespace prrng {

namespace detail {

    inline std::string unquote(const std::string& arg)
    {
        std::string ret = arg;
        ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
        return ret;
    };

}

/**
Return version string, for example `"0.1.0"`

\return std::string
*/
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(PRRNG_VERSION)));
};

class Generator
{
public:

    Generator() = default;

    template <class R, class S>
    R random(const S& shape)
    {
        return this->random_impl<R>(shape);
    }

    template <class R, class I, std::size_t L>
    R random(const I (&shape)[L])
    {
        return this->random_impl<R>(shape);
    }

    template <class R, class S>
    R weibull(const S& shape, double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl(shape, k, lambda);
    };

    template <class R, class I, std::size_t L>
    R weibull(const I (&shape)[L], double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl(shape, k, lambda);
    };

private:

    template <class R, class S>
    R random_impl(const S& shape)
    {
        R ret = xt::empty<double>(shape);
        this->draw_list(&ret.data()[0], ret.size());
        return ret;
    };

    template <class R, class S>
    R weibull_impl(const S& shape, double k, double lambda)
    {
        R r = this->random(shape);
        return lambda * xt::pow(- xt::log(r - 1.0), k);
    };

protected:

    virtual void draw_list(double* data, size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            data[i] = 0.5;
        }
    }
};

/**
The original code was licensed as follows:

    The PCG random number generator was developed by Melissa O'Neill
    <oneill@pcg-random.org>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    For additional information about the PCG random number generation scheme,
    including its license and other licensing options, visit

        http://www.pcg-random.org

Furthermore bits are code are taken from another wrapper:

    Wenzel Jakob, February 2015
    https://github.com/wjakob/pcg32
*/
class pcg32 : public Generator
{
public:

    pcg32()
    {
        this->init();
    };

    template <typename T>
    pcg32(T initstate)
    {
        static_assert(sizeof(uint64_t) >= sizeof(T), "Down-casting not allowed.");
        this->init(static_cast<uint64_t>(initstate));
    };

    template <typename T, typename S>
    pcg32(T initstate, S initseq)
    {
        static_assert(sizeof(uint64_t) >= sizeof(T), "Down-casting not allowed.");
        static_assert(sizeof(uint64_t) >= sizeof(S), "Down-casting not allowed.");
        this->init(static_cast<uint64_t>(initstate), static_cast<uint64_t>(initseq));
    };

    uint32_t operator()()
    {
        uint64_t oldstate = m_state;
        m_state = oldstate * PRRNG_PCG32_MULT + m_inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    };

    uint32_t next_uint32()
    {
        return (*this)();
    }

    /**
    Generate a double precision floating point value on the interval [0, 1)

    \remark Since the underlying random number generator produces 32 bit output,
    only the first 32 mantissa bits will be filled (however, the resolution is still
    finer than in \ref nextFloat(), which only uses 23 mantissa bits).

    \note Taken from Wenzel Jakob, February 2015, https://github.com/wjakob/pcg32.

    \return Random number.
    */
    double next_double()
    {
        union {
            uint64_t u;
            double d;
        } x;

        x.u = ((uint64_t) next_uint32() << 20) | 0x3ff0000000000000ULL;

        return x.d - 1.0;
    }

    template <typename R = uint64_t>
    R state()
    {
        static_assert(std::numeric_limits<R>::max >= std::numeric_limits<decltype(m_state)>::max,
            "Down-casting not allowed.");

        static_assert(std::numeric_limits<R>::min <= std::numeric_limits<decltype(m_state)>::min,
            "Down-casting not allowed.");

        return static_cast<R>(m_state);
    };

    template <typename R = uint64_t>
    R initstate()
    {
        static_assert(std::numeric_limits<R>::max >= std::numeric_limits<decltype(m_initstate)>::max,
            "Down-casting not allowed.");

        static_assert(std::numeric_limits<R>::min <= std::numeric_limits<decltype(m_initstate)>::min,
            "Down-casting not allowed.");

        return static_cast<R>(m_initstate);
    };

    template <typename R = uint64_t>
    R initseq()
    {
        static_assert(std::numeric_limits<R>::max >= std::numeric_limits<decltype(m_initseq)>::max,
            "Down-casting not allowed.");

        static_assert(std::numeric_limits<R>::min <= std::numeric_limits<decltype(m_initseq)>::min,
            "Down-casting not allowed.");

        return static_cast<R>(m_initseq);
    };

    template <typename T>
    void restore(T state)
    {
        static_assert(sizeof(uint64_t) >= sizeof(T), "Down-casting not allowed.");
        m_state = static_cast<uint64_t>(state);
    };

    /**
    Compute the distance between two PCG32 pseudorandom number generators.
    */
    int64_t operator-(const pcg32 &other) const
    {
        PRRNG_ASSERT(m_inc == other.m_inc);

        uint64_t
            cur_mult = PRRNG_PCG32_MULT,
            cur_plus = m_inc,
            cur_state = other.m_state,
            the_bit = 1u,
            distance = 0u;

        while (m_state != cur_state) {
            if ((m_state & the_bit) != (cur_state & the_bit)) {
                cur_state = cur_state * cur_mult + cur_plus;
                distance |= the_bit;
            }
            assert((m_state & the_bit) == (cur_state & the_bit));
            the_bit <<= 1;
            cur_plus = (cur_mult + 1ULL) * cur_plus;
            cur_mult *= cur_mult;
        }

        return (int64_t) distance;
    };

    /**
    Compute the distance between two PCG32 pseudorandom number generators.
    */
    template <typename R = int64_t>
    R distance(const pcg32 &other)
    {
        static_assert(sizeof(R) >= sizeof(int64_t), "Down-casting not allowed.");

        int64_t r = this->operator-(other);

        #ifdef PRRNG_ENABLE_ASSERT
        bool u = std::is_unsigned<R>::value;
        PRRNG_ASSERT((r < 0 && !u) || r >= 0);
        #endif

        return static_cast<R>(r);
    }

    /**
    Compute the distance between two states.

    \warning The increment of used to generate must be the same. There is no way of checking here!
    */
    template <typename R = int64_t, typename T>
    R distance(T other_state)
    {
        static_assert(sizeof(R) >= sizeof(int64_t), "Down-casting not allowed.");

        uint64_t
            cur_mult = PRRNG_PCG32_MULT,
            cur_plus = m_inc,
            cur_state = other_state,
            the_bit = 1u,
            distance = 0u;

        while (m_state != cur_state) {
            if ((m_state & the_bit) != (cur_state & the_bit)) {
                cur_state = cur_state * cur_mult + cur_plus;
                distance |= the_bit;
            }
            assert((m_state & the_bit) == (cur_state & the_bit));
            the_bit <<= 1;
            cur_plus = (cur_mult + 1ULL) * cur_plus;
            cur_mult *= cur_mult;
        }

        int64_t r = (int64_t) distance;

        #ifdef PRRNG_ENABLE_ASSERT
        bool u = std::is_unsigned<R>::value;
        PRRNG_ASSERT((r < 0 && !u) || r >= 0);
        #endif

        return static_cast<R>(r);
    }

    /**
    Equality operator.
    */
    bool operator==(const pcg32 &other) const
    {
        return m_state == other.m_state && m_inc == other.m_inc;
    };

    /**
    Inequality operator.
    */
    bool operator!=(const pcg32 &other) const
    {
        return m_state != other.m_state || m_inc != other.m_inc;
    };

protected:

    void draw_list(double* data, size_t n) override
    {
        for (size_t i = 0; i < n; ++i) {
            data[i] = next_double();
        }
    }

private:

    void init(uint64_t initstate = 0x853c49e6748fea9bULL, uint64_t initseq = 0xda3e39cb94b95bdbULL)
    {
        PRRNG_ASSERT(initstate >= 0);
        PRRNG_ASSERT(initseq >= 0);

        m_initstate = initstate;
        m_initseq = initseq;

        m_state = 0U;
        m_inc = (initseq << 1u) | 1u;
        (*this)();
        m_state += initstate;
        (*this)();
    };

private:

    uint64_t m_initstate; ///< State initiator
    uint64_t m_initseq;   ///< Sequence initiator
    uint64_t m_state;     ///< RNG state. All values are possible.
    uint64_t m_inc;       ///< Controls which RNG sequence (stream) is selected. Must *always* be odd.
};

} // namespace prrng

#endif
