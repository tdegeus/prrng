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
Default initialisation state for pcg32()
(used as constructor parameter that can be overwritten at run-time).
*/
#define PRRNG_PCG32_INITSTATE 0x853c49e6748fea9bULL

/**
Default initialisation sequence for pcg32()
(used as constructor parameter that can be overwritten at run-time).
*/
#define PRRNG_PCG32_INITSEQ 0xda3e39cb94b95bdbULL

/**
Multiplicative factor for pcg32()
(used internally, cannot be overwritten at run-time).
*/
#define PRRNG_PCG32_MULT 6364136223846793005ULL

#include <array>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

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

    template <class S, class T>
    inline std::vector<size_t> concatenate(const S& s, const T& t)
    {
        std::vector<size_t> r;
        r.insert(r.begin(), s.begin(), s.end());
        r.insert(r.end(), t.begin(), t.end());
        return r;
    }

    template <class S>
    inline size_t size(const S& shape)
    {
        using ST = typename S::value_type;
        return std::accumulate(shape.cbegin(), shape.cend(), ST(1), std::multiplies<ST>());
    }

    template <class I, std::size_t L>
    std::array<I, L> to_array(const I (&shape)[L])
    {
        std::array<I, L> r;
        std::copy(&shape[0], &shape[0] + L, r.begin());
        return r;
    }
}

/**
Return version string, for example `"0.1.0"`

\return std::string
*/
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(PRRNG_VERSION)));
};

namespace detail
{

    template <class S, class T, typename = void>
    struct shapes
    {
        static std::vector<size_t> concatenate(const T& t, const S& s)
        {
            std::vector<size_t> r(t.cbegin(), t.cend());
            r.insert(r.end(), s.begin(), s.end());
            return r;
        }
    };

    template <class S, class T>
    struct shapes<T, S, typename std::enable_if<xt::has_fixed_rank_t<T>::value &&
                                                xt::has_fixed_rank_t<S>::value>::type>
    {
        constexpr static size_t rank = std::rank<T>::value + std::rank<S>::value;

        static std::array<size_t, rank> concatenate(const T& t, const S& s)
        {
            std::array<size_t, rank> r;
            std::copy(t.cbegin(), t.cend(), r.begin());
            std::copy(s.cbegin(), s.cend(), r.begin() + std::rank<T>::value);
            return r;
        }
    };

    // template <class S>
    // inline size_t size(const S& shape)
    // {
    //     using ST = typename S::value_type;
    //     return std::accumulate(shape.cbegin(), shape.cend(), ST(1), std::multiplies<ST>());
    // }

} // namespace detail


/**
Base class of the pseudorandom number generators.
This class provides common methods, but itself does not really do much.
*/
class Generator
{
public:

    Generator() = default;

    virtual ~Generator() = default;

    /**
    Generate an nd-array of random numbers \f$ 0 \leq r \leq 1 \f$.

    \tparam R The type of the output array, e.g. `xt::xtensor<double, 3>`.
    \tparam S The type of `shape`, e.g. `std::array<size_t, 3>`.

    \param shape The shape of the nd-array.
    \return The sample of shape `shape`.
    */
    template <class R, class S>
    R random(const S& shape)
    {
        return this->random_impl<R>(shape);
    }

    /**
    Generate an nd-array of random numbers \f$ 0 \leq r \leq 1 \f$.

    \tparam R The type of the output array, e.g. `xt::xtensor<double, 3>`.

    \param shape The shape of the nd-array (brace input allowed, e.g. `{2, 3, 4}`.
    \return The sample of shape `shape`.
    */
    template <class R, class I, std::size_t L>
    R random(const I (&shape)[L])
    {
        // size_t size = std::accumulate(&shape[0], &shape[L], 1, std::multiplies<size_t>());
        // std::cout << size << std::endl;

        return this->random_impl<R>(shape);
    }

    /**
    Generate an nd-array of random numbers distributed according to Weibull distribution.
    Internally, the output of random() is converted using the cumulative density

    \f$ \Phi(x) = 1 - e^{-(x / \lambda)^k} \f$

    such that the output `r` from random() leads to

    \f$ x = \lambda (- \ln (1 - r))^{1 / k}) \f$

    \tparam R The type of the output array, e.g. `xt::xtensor<double, 3>`.
    \tparam S The type of `shape`, e.g. `std::array<size_t, 3>`.

    \param shape The shape of the nd-array.
    \param k The "shape" parameter.
    \param lambda The "scale" parameter.
    \return The sample of shape `shape`.
    */
    template <class R, class S>
    R weibull(const S& shape, double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl<R>(shape, k, lambda);
    };

    /**
    Generate an nd-array of random numbers distributed according to Weibull distribution.
    Internally, the output of random() is converted using the cumulative density

    \f$ \Phi(x) = 1 - e^{-(x / \lambda)^k} \f$

    such that the output `r` from random() leads to

    \f$ x = \lambda (- \ln (1 - r))^{1 / k}) \f$

    \tparam R The type of the output array, e.g. `xt::xtensor<double, 3>`.

    \param shape The shape of the nd-array (brace input allowed, e.g. `{2, 3, 4}`.
    \param k The "shape" parameter.
    \param lambda The "scale" parameter.
    \return The sample of shape `shape`.
    */
    template <class R, class I, std::size_t L>
    R weibull(const I (&shape)[L], double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl<R>(shape, k, lambda);
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
        R r = this->random_impl<R>(shape);
        return lambda * xt::pow(- xt::log(1.0 - r), 1.0 / k);
    };

protected:

    /**
    Draw `n` random numbers and write them to list (input as pointer `data`).

    \param data Pointer to the data (no bounds-check).
    \param n Size of `data`.
    */
    virtual void draw_list(double* data, size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            data[i] = 0.5;
        }
    }
};

/**
Random number generate using the pcg32 algorithm.
The class generate random 32-bit random numbers (of type `uint32_t`).
In addition, they can be converted to nd-arrays of random floating-point numbers (according)
using derived methods from Generate().

The algorithm is full based on:

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

Whereby most code is taken from the follow wrapper:

    Wenzel Jakob, February 2015
    https://github.com/wjakob/pcg32
*/
class pcg32 : public Generator
{
public:

    /**
    Constructor.

    \param initstate State initiator.
    \param initseq Sequence initiator.
    */
    template <typename T = uint64_t, typename S = uint64_t>
    pcg32(T initstate = PRRNG_PCG32_INITSTATE, S initseq = PRRNG_PCG32_INITSEQ)
    {
        static_assert(sizeof(uint64_t) >= sizeof(T), "Down-casting not allowed.");
        static_assert(sizeof(uint64_t) >= sizeof(S), "Down-casting not allowed.");
        this->seed(static_cast<uint64_t>(initstate), static_cast<uint64_t>(initseq));
    };

    /**
    \return Draw new random number (uniformly distributed, `0 <= r <= max(uint32_t)`).
    This advances the state of the generator by one increment.

    \author Melissa O'Neill, http://www.pcg-random.org.
    */
    uint32_t operator()()
    {
        uint64_t oldstate = m_state;
        m_state = oldstate * PRRNG_PCG32_MULT + m_inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    };

    /**
    \return Draw new random number (uniformly distributed, `0 <= r <= max(uint32_t)`).
    This advances the state of the generator by one increment.

    \note Wrapper around operator().
    */
    uint32_t next_uint32()
    {
        return (*this)();
    }

    /**
    \param bound Bound on the return.
    \return Draw new random number (uniformly distributed, `0 <= r <= bound`).

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    uint32_t next_uint32(uint32_t bound)
    {
        // To avoid bias, we need to make the range of the RNG a multiple of
        // bound, which we do by dropping output less than a threshold.
        // A naive scheme to calculate the threshold would be to do
        //
        //     uint32_t threshold = 0x100000000ull % bound;
        //
        // but 64-bit div/mod is slower than 32-bit div/mod (especially on
        // 32-bit platforms).  In essence, we do
        //
        //     uint32_t threshold = (0x100000000ull-bound) % bound;
        //
        // because this version will calculate the same modulus, but the LHS
        // value is less than 2^32.

        uint32_t threshold = (~bound+1u) % bound;

        // Uniformity guarantees that this loop will terminate.  In practice, it
        // should usually terminate quickly; on average (assuming all bounds are
        // equally likely), 82.25% of the time, we can expect it to require just
        // one iteration.  In the worst case, someone passes a bound of 2^31 + 1
        // (i.e., 2147483649), which invalidates almost 50% of the range.  In
        // practice, bounds are typically small and only a tiny amount of the range
        // is eliminated.
        for (;;) {
            uint32_t r = next_uint32();
            if (r >= threshold) {
                return r % bound;
            }
        }
    }

    /**
    \return Generate a single precision floating point value on the interval [0, 1).

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    float next_float() {
        /* Trick from MTGP: generate an uniformly distributed
           single precision number in [1,2) and subtract 1. */
        union {
            uint32_t u;
            float f;
        } x;
        x.u = (next_uint32() >> 9) | 0x3f800000u;
        return x.f - 1.0f;
    }

    /**
    \return Generate a double precision floating point value on the interval [0, 1)

    \remark Since the underlying random number generator produces 32 bit output,
    only the first 32 mantissa bits will be filled (however, the resolution is still
    finer than in next_float(), which only uses 23 mantissa bits).

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
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

    /**
    \return The current "state" of the generator. If the same initseq() is used, this exact point
    in the sequence can be restored with restore().

    \tparam R use a different return-type. There are some internal checks if the type is able to
    store the internal state of type `uint64_t`.
    */
    template <typename R = uint64_t>
    R state()
    {
        static_assert(std::numeric_limits<R>::max >= std::numeric_limits<decltype(m_state)>::max,
            "Down-casting not allowed.");

        static_assert(std::numeric_limits<R>::min <= std::numeric_limits<decltype(m_state)>::min,
            "Down-casting not allowed.");

        return static_cast<R>(m_state);
    };

    /**
    \return The state initiator that was used upon construction.

    \tparam R use a different return-type. There are some internal checks if the type is able to
    store the internal state of type `uint64_t`.
    */
    template <typename R = uint64_t>
    R initstate()
    {
        static_assert(std::numeric_limits<R>::max >= std::numeric_limits<decltype(m_initstate)>::max,
            "Down-casting not allowed.");

        static_assert(std::numeric_limits<R>::min <= std::numeric_limits<decltype(m_initstate)>::min,
            "Down-casting not allowed.");

        return static_cast<R>(m_initstate);
    };

    /**
    \return The sequence initiator that was used upon construction.

    \tparam R use a different return-type. There are some internal checks if the type is able to
    store the internal state of type `uint64_t`.
    */
    template <typename R = uint64_t>
    R initseq()
    {
        static_assert(std::numeric_limits<R>::max >= std::numeric_limits<decltype(m_initseq)>::max,
            "Down-casting not allowed.");

        static_assert(std::numeric_limits<R>::min <= std::numeric_limits<decltype(m_initseq)>::min,
            "Down-casting not allowed.");

        return static_cast<R>(m_initseq);
    };

    /**
    Restore a given state in the sequence. See state().

    \tparam R use a different return-type. There are some internal checks if the type is able to
    store the internal state of type `uint64_t`.
    */
    template <typename T>
    void restore(T state)
    {
        static_assert(sizeof(uint64_t) >= sizeof(T), "Down-casting not allowed.");
        m_state = static_cast<uint64_t>(state);
    };

    /**
    \return The distance between two PCG32 pseudorandom number generators.

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
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
    \return The distance between two PCG32 pseudorandom number generators.

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.

    \tparam R use a different return-type. There are some internal checks if the type is able to
    store the internal state of type `int64_t`.
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
    \return The distance between two states.

    \warning The increment of used to generate must be the same. There is no way of checking here!

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.

    \tparam R use a different return-type. There are some internal checks if the type is able to
    store the internal state of type `int64_t`.
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
    Multi-step advance function (jump-ahead, jump-back).

    \param distance Distance to jump ahead or jump back (depending on the sign).
    This changes that state of the generator by the appropriate number of increments.

    \note The method used here is based on Brown, "Random Number Generation
    with Arbitrary Stride", Transactions of the American Nuclear Society (Nov. 1994).
    The algorithm is very similar to fast exponentiation.

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    template <typename T>
    void advance(T distance)
    {
        static_assert(sizeof(int64_t) >= sizeof(T), "Down-casting not allowed.");

        int64_t delta_ = static_cast<int64_t>(distance);

        uint64_t
            cur_mult = PRRNG_PCG32_MULT,
            cur_plus = m_inc,
            acc_mult = 1u,
            acc_plus = 0u;

        /* Even though delta is an unsigned integer, we can pass a signed
           integer to go backwards, it just goes "the long way round". */
        uint64_t delta = (uint64_t) delta_;

        while (delta > 0) {
            if (delta & 1) {
                acc_mult *= cur_mult;
                acc_plus = acc_plus * cur_mult + cur_plus;
            }
            cur_plus = (cur_mult + 1) * cur_plus;
            cur_mult *= cur_mult;
            delta /= 2;
        }
        m_state = acc_mult * m_state + acc_plus;
    }

    /**
    Draw uniformly distributed permutation and permute the given STL container.

    \param begin
    \param end

    \note From: Knuth, TAoCP Vol. 2 (3rd 3d), Section 3.4.2

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    template <typename Iterator>
    void shuffle(Iterator begin, Iterator end)
    {
        for (Iterator it = end - 1; it > begin; --it) {
            std::iter_swap(it, begin + next_uint32((uint32_t) (it - begin + 1)));
        }
    }

    /**
    Equality operator.

    \param other The generator to which to compare.

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    bool operator==(const pcg32 &other) const
    {
        return m_state == other.m_state && m_inc == other.m_inc;
    };

    /**
    Inequality operator.

    \param other The generator to which to compare.

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
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

    void seed(uint64_t initstate, uint64_t initseq)
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

/**
Grid of independent generators.
*/
class nd_Generator
{
public:

    nd_Generator() = default;

    virtual ~nd_Generator() = default;

    /**
    Return the size.
    */
    size_t size() const
    {
        return m_size;
    }

    /**
    Check in bounds.
    */
    template <class T>
    bool inbounds(const T& index)
    {
        if (index.size() != m_strides.size()) {
            return false;
        }

        for (size_t i = 0; i < m_strides.size(); ++i) {
            if (index[i] >= m_shape[i]) {
                return false;
            }
        }

        return true;
    }

    /**
    Generate an nd-array of random numbers \f$ 0 \leq r \leq 1 \f$.

    \tparam R The type of the output array, e.g. `xt::xtensor<double, 3>`.
    \tparam S The type of `shape`, e.g. `std::array<size_t, 3>`.

    \param shape The shape of the nd-array.
    \return The sample of shape `shape`.
    */
    template <class R, class S>
    R random(const S& shape)
    {
        // size_t size = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<size_t>());
        return this->random_impl<R>(shape);
    }

    /**
    Generate an nd-array of random numbers \f$ 0 \leq r \leq 1 \f$.

    \tparam R The type of the output array, e.g. `xt::xtensor<double, 3>`.

    \param shape The shape of the nd-array (brace input allowed, e.g. `{2, 3, 4}`.
    \return The sample of shape `shape`.
    */
    template <class R, class I, std::size_t L>
    R random(const I (&shape)[L])
    {
        return this->random_impl<R>(detail::to_array(shape));
    }

    /**
    Generate an nd-array of random numbers distributed according to Weibull distribution.
    Internally, the output of random() is converted using the cumulative density

    \f$ \Phi(x) = 1 - e^{-(x / \lambda)^k} \f$

    such that the output `r` from random() leads to

    \f$ x = \lambda (- \ln (1 - r))^{1 / k}) \f$

    \tparam R The type of the output array, e.g. `xt::xtensor<double, 3>`.
    \tparam S The type of `shape`, e.g. `std::array<size_t, 3>`.

    \param shape The shape of the nd-array.
    \param k The "shape" parameter.
    \param lambda The "scale" parameter.
    \return The sample of shape `shape`.
    */
    template <class R, class S>
    R weibull(const S& shape, double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl<R>(shape, k, lambda);
    };

    /**
    Generate an nd-array of random numbers distributed according to Weibull distribution.
    Internally, the output of random() is converted using the cumulative density

    \f$ \Phi(x) = 1 - e^{-(x / \lambda)^k} \f$

    such that the output `r` from random() leads to

    \f$ x = \lambda (- \ln (1 - r))^{1 / k}) \f$

    \tparam R The type of the output array, e.g. `xt::xtensor<double, 3>`.

    \param shape The shape of the nd-array (brace input allowed, e.g. `{2, 3, 4}`.
    \param k The "shape" parameter.
    \param lambda The "scale" parameter.
    \return The sample of shape `shape`.
    */
    template <class R, class I, std::size_t L>
    R weibull(const I (&shape)[L], double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl<R>(detail::to_array(shape), k, lambda);
    };

private:

    template <class R, class S>
    R random_impl(const S& shape)
    {
        auto n = detail::size(shape);
        R ret = xt::empty<double>(detail::concatenate(m_shape, shape));
        this->draw_list(&ret.data()[0], n);
        return ret;
    };

    template <class R, class S>
    R weibull_impl(const S& shape, double k, double lambda)
    {
        R r = this->random_impl<R>(shape);
        return lambda * xt::pow(- xt::log(1.0 - r), 1.0 / k);
    };

protected:

    /**
    Draw `n` random numbers and write them to list (input as pointer `data`).

    \param data Pointer to the data (no bounds-check).
    \param n Size of `data`.
    */
    virtual void draw_list(double* data, size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            data[i] = 0.5;
        }
    }

protected:

    size_t m_size = 0;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
};

/**
Grid of independent generators.
*/
class nd_pcg32 : public nd_Generator
{
public:

    template <class T>
    nd_pcg32(const T& initstate)
    {
        m_shape = std::vector<size_t>(initstate.shape().cbegin(), initstate.shape().cend());
        m_strides = std::vector<size_t>(initstate.strides().cbegin(), initstate.strides().cend());
        m_size = initstate.size();
        m_gen.reserve(m_size);

        for (size_t i = 0; i < m_size; ++i) {
            m_gen.push_back(pcg32(initstate.data()[i]));
        }
    }

    template <class T, class U>
    nd_pcg32(const T& initstate, const U& initseq)
    {
        PRRNG_ASSERT(xt::has_shape(initstate, initseq));

        m_shape = std::vector<size_t>(initstate.shape().cbegin(), initstate.shape().cend());
        m_strides = std::vector<size_t>(initstate.strides().cbegin(), initstate.strides().cend());
        m_size = initstate.size();
        m_gen.reserve(m_size);

        for (size_t i = 0; i < m_size; ++i) {
            m_gen.push_back(pcg32(initstate.data()[i], initseq.data()[i]));
        }
    }

    pcg32& operator[](size_t i)
    {
        PRRNG_ASSERT(i < m_size);

        return m_gen[i];
    }

    template <class T>
    pcg32& get(const T& index)
    {
        PRRNG_ASSERT(this->inbounds(index));
        return m_gen[std::inner_product(index.begin(), index.end(), m_strides.begin(), 0)];
    }

    template <class R>
    R state()
    {
        using value_type = typename R::value_type;
        R ret = xt::empty<value_type>(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            ret.data()[i] = m_gen[i].state<value_type>();
        }

        return ret;
    }

    template <class R>
    R initstate()
    {
        using value_type = typename R::value_type;
        R ret = xt::empty<value_type>(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            ret.data()[i] = m_gen[i].initstate<value_type>();
        }

        return ret;
    }

    template <class R>
    R initseq()
    {
        using value_type = typename R::value_type;
        R ret = xt::empty<value_type>(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            ret.data()[i] = m_gen[i].initseq<value_type>();
        }

        return ret;
    }

    template <class T>
    void restore(const T& arg)
    {
        for (size_t i = 0; i < m_size; ++i) {
            m_gen[i].restore(arg.data()[i]);
        }
    }

protected:

    void draw_list(double* data, size_t n) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            for (size_t j = 0; j < n; ++j) {
                data[i * n + j] = m_gen[i].next_double();
            }
        }
    }

private:

    std::vector<pcg32> m_gen;
};



} // namespace prrng

#endif
