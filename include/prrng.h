/**
Portable Reconstructible (Approximate!) Random Number Generator.

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
        m_state = oldstate * 6364136223846793005ULL + m_inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    };

    template <typename R = uint64_t>
    R state()
    {
        static_assert(sizeof(R) >= sizeof(uint64_t), "Down-casting not allowed.");
        return static_cast<R>(m_state);
    };

    template <typename T>
    void restore(T state)
    {
        static_assert(sizeof(uint64_t) >= sizeof(T), "Down-casting not allowed.");
        m_state = static_cast<uint64_t>(state);
    };

protected:

    void draw_list(double* data, size_t n) override
    {
        for (size_t i = 0; i < n; ++i) {
            data[i] = ((*this)()) * (1.0 / std::numeric_limits<uint32_t>::max());
        }
    }


private:

    void init(uint64_t initstate = 0x853c49e6748fea9bULL, uint64_t initseq = 0xda3e39cb94b95bdbULL)
    {
        m_state = 0U;
        m_inc = (initseq << 1u) | 1u;
        (*this)();
        m_state += initstate;
        (*this)();
    };

private:

    uint64_t m_state; ///< RNG state. All values are possible.
    uint64_t m_inc;   ///< Controls which RNG sequence (stream) is selected. Must *always* be odd.
};

} // namespace prrng

#endif
