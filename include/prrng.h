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
\par license This project is released under the MIT License.
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
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

#ifndef PRRNG_USE_BOOST
/**
If this macro is defined before including prrng:

    #define PRRNG_USE_BOOST
    #include <prrng.h>

then Boost is used to compute the inverse Gamma function.
Without it, the Gamma distribution does not work and returns NaNs.

Note that one can switch-off Boost explicitly by:

    #define PRRNG_USE_BOOST 0
    #include <prrng.h>

Likewise one can be explicit in enabling it:

    #define PRRNG_USE_BOOST 1
    #include <prrng.h>
*/
#define PRRNG_USE_BOOST 0
#elif PRRNG_USE_BOOST != 0
#define PRRNG_USE_BOOST 1
#endif

#if PRRNG_USE_BOOST
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <xtensor/xvectorize.hpp>
#endif

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

-   Configure using CMake at install time. Internally uses:

        python -c "from setuptools_scm import get_version; print(get_version())"

-   Define externally using:

        -DPRRNG_VERSION="`python -c "from setuptools_scm import get_version; print(get_version())"`"

    From the root of this project. This is what ``setup.py`` does.

Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using
``setuptools_scm``. Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION`` to
overwrite the automatic version.
*/
#ifndef PRRNG_VERSION
#define PRRNG_VERSION "@PROJECT_VERSION@"
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
Portable Reconstructible (Pseudo!) Random Number Generator
*/
namespace prrng {

namespace detail {

/**
Remove " from string.

\param arg Input string.
\return String without "
*/
inline std::string unquote(const std::string& arg)
{
    std::string ret = arg;
    ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
    return ret;
}

template <class T>
struct is_std_array : std::false_type {
};

template <class T, size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {
};

template <class T>
struct is_xtensor : std::false_type {
};

template <class T, size_t N>
struct is_xtensor<xt::xtensor<T, N>> : std::true_type {
};

/**
Check that an object has a certain fixed rank.
*/
template <size_t N, class T, typename = void>
struct check_fixed_rank {
    constexpr static bool value = false;
};

template <size_t N, class T>
struct check_fixed_rank<N, T, typename std::enable_if_t<is_xtensor<T>::value>> {
    constexpr static bool value = (N == xt::get_rank<T>::value);
};

template <size_t N, class T>
struct check_fixed_rank<N, T, typename std::enable_if_t<is_std_array<T>::value>> {
    constexpr static bool value = (N == std::tuple_size<T>::value);
};

/**
Get default return type
*/
template <typename R, size_t N>
struct return_type_fixed {
    using type = typename xt::xtensor<R, N>;
};

/**
Get default return type
*/
template <typename R, class S, typename = void>
struct return_type {
    using type = typename xt::xarray<R>;
};

template <typename R, class S>
struct return_type<R, S, typename std::enable_if_t<is_std_array<S>::value>> {
    using type = typename xt::xtensor<R, std::tuple_size<S>::value>;
};

/**
Get default return type
*/
template <typename R, class S, class T, typename = void>
struct composite_return_type {
    using type = typename xt::xarray<R>;
};

template <typename R, class S, class T>
struct composite_return_type<
    R,
    S,
    T,
    typename std::enable_if_t<is_std_array<S>::value && is_std_array<T>::value>> {
    constexpr static size_t N = std::tuple_size<S>::value + std::tuple_size<T>::value;
    using type = typename xt::xtensor<R, N>;
};

/**
Concatenate two objects that have `begin()` and `end()` methods.

\param s First object (e.g. std::vector).
\param t Second object (e.g. std::vector).
\return Concatenated [t, s]
*/
template <class S, class T, typename = void>
struct concatenate {
    static auto two(const S& s, const T& t)
    {
        std::vector<size_t> r;
        r.insert(r.begin(), s.cbegin(), s.cend());
        r.insert(r.end(), t.cbegin(), t.cend());
        return r;
    }
};

template <class S, class T>
struct concatenate<
    S,
    T,
    typename std::enable_if_t<is_std_array<S>::value && is_std_array<T>::value>> {
    static auto two(const S& s, const T& t)
    {
        std::array<size_t, std::tuple_size<S>::value + std::tuple_size<T>::value> r;
        std::copy(s.cbegin(), s.cend(), r.begin());
        std::copy(t.cbegin(), t.cend(), r.begin() + std::tuple_size<S>::value);
        return r;
    }
};

/**
Compute 'size' from 'shape'.

\param shape Shape array.
\return Size
*/
template <class S>
inline size_t size(const S& shape)
{
    using ST = typename S::value_type;
    return std::accumulate(shape.cbegin(), shape.cend(), ST(1), std::multiplies<ST>());
}

/**
Return as std::array.
*/
template <class I, std::size_t L>
std::array<I, L> to_array(const I (&shape)[L])
{
    std::array<I, L> r;
    std::copy(&shape[0], &shape[0] + L, r.begin());
    return r;
}
} // namespace detail

/**
Return version string, for example `"0.1.0"`

\return std::string
*/
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(PRRNG_VERSION)));
}

/**
Normal distribution.

References:

-   https://en.wikipedia.org/wiki/Normal_distribution
-   https://www.boost.org/doc/libs/1_63_0/libs/math/doc/html/math_toolkit/sf_erf/error_inv.html
-   https://www.boost.org/doc/libs/1_78_0/boost/math/special_functions/detail/erf_inv.hpp
*/
class normal_distribution {
public:
    /**
    Constructor.

    \param mu Average.
    \param sigma Standard deviation.
    */
    normal_distribution(double mu = 0.0, double sigma = 1.0)
    {
        m_mu = mu;
        m_sigma = sigma;
        m_sigma_sqrt2 = m_sigma * std::sqrt(2.0);
    }

    /**
    Return the probability density function.

    \param x Coordinates.
    \return probability density for each `x`.
    */
    template <class T>
    T pdf(const T& x)
    {
        return xt::exp(-0.5 * xt::pow((x - m_mu) / m_sigma, 2.0)) /
               (m_sigma_sqrt2 * std::sqrt(xt::numeric_constants<double>::PI));
    }

    /**
    Return the cumulative density function.

    \param x Coordinates.
    \return cumulative density for each `x`.
    */
    template <class T>
    T cdf(const T& x)
    {
        return 0.5 * (1.0 + xt::erf((x - m_mu) / m_sigma_sqrt2));
    }

    /**
    Return the quantile (the inverse of the cumulative density function).

    \param p Probability [0, 1].
    \return quantile for each `p`.
    */
    template <class T>
    T quantile(const T& p)
    {
        using value_type = typename T::value_type;

#if PRRNG_USE_BOOST
        auto f = xt::vectorize(boost::math::erf_inv<value_type>);
        return m_mu + m_sigma_sqrt2 * f(2.0 * p - 1.0);
#else
        auto ret = p;
        ret.fill(std::numeric_limits<value_type>::quiet_NaN());
        return ret;
#endif
    }

private:
    double m_mu;
    double m_sigma;
    double m_sigma_sqrt2;
};

/**
Weibull distribution.

References:

-   https://en.wikipedia.org/wiki/Weibull_distribution
-   https://github.com/boostorg/math/blob/develop/include/boost/math/distributions/weibull.hpp
*/
class weibull_distribution {
public:
    /**
    Constructor.

    \param k Shape parameter.
    \param lambda Scale parameter.
    */
    weibull_distribution(double k = 1.0, double lambda = 1.0)
    {
        m_shape = k;
        m_scale = lambda;
    }

    /**
    Return the probability density function.

    \param x Coordinates.
    \return probability density for each `x`.
    */
    template <class T>
    T pdf(const T& x)
    {
        T ret = xt::exp(-xt::pow(x / m_scale, m_shape));
        ret *= xt::pow(x / m_scale, m_shape - 1.0) * m_shape / m_scale;

        if (xt::any(xt::equal(x, 0))) {
            if (m_shape == 1) {
                ret = xt::where(xt::equal(x, 0), 1.0 / m_scale, ret);
            }
            else if (m_shape > 1) {
                ret = xt::where(xt::equal(x, 0), 0.0, ret);
            }
            else {
                throw std::runtime_error("[prrng::weibull_distribution::pdf] Overflow error");
            }
        }

        return ret;
    }

    /**
    Return the cumulative density function.

    \param x Coordinates.
    \return cumulative density for each `x`.
    */
    template <class T>
    T cdf(const T& x)
    {
        return -xt::expm1(-xt::pow(x / m_scale, m_shape));
    }

    /**
    Return the quantile (the inverse of the cumulative density function).

    \param p Probability [0, 1].
    \return quantile for each `p`.
    */
    template <class T>
    T quantile(const T& p)
    {
        return m_scale * xt::pow(-xt::log1p(-p), 1.0 / m_shape);
    }

private:
    double m_shape;
    double m_scale;
};

/**
Gamma distribution.

References:

-   https://en.wikipedia.org/wiki/Gamma_distribution
-   https://github.com/boostorg/math/blob/develop/include/boost/math/distributions/gamma.hpp
*/
class gamma_distribution {
public:
    /**
    Constructor.

    \param k Shape parameter.
    \param theta Scale parameter.
    */
    gamma_distribution(double k, double theta)
    {
        m_shape = k;
        m_scale = theta;
    }

    /**
    Return the probability density function.
    Only available when compiled with PRRNG_USE_BOOST
    (e.g. using the CMake target `prrng::use_boost`).

    \param x Coordinates.
    \return probability density for each `x`.
    */
    template <class T>
    T pdf(const T& x)
    {
        using value_type = typename T::value_type;

#if PRRNG_USE_BOOST
        auto f = xt::vectorize(boost::math::gamma_p_derivative<value_type, value_type>);
        T ret = f(m_shape, x / m_scale);
        return xt::where(xt::equal(x, 0), 0.0, ret);
#else
        T ret = x;
        ret.fill(std::numeric_limits<value_type>::quiet_NaN());
        return ret;
#endif
    }

    /**
    Return the cumulative density function.
    Only available when compiled with PRRNG_USE_BOOST
    (e.g. using the CMake target `prrng::use_boost`).

    \param x Coordinates.
    \return cumulative density for each `x`.
    */
    template <class T>
    T cdf(const T& x)
    {
        using value_type = typename T::value_type;

#if PRRNG_USE_BOOST
        auto f = xt::vectorize(boost::math::gamma_p<value_type, value_type>);
        return f(m_shape, x / m_scale);
#else
        auto ret = x;
        ret.fill(std::numeric_limits<value_type>::quiet_NaN());
        return ret;
#endif
    }

    /**
    Return the quantile (the inverse of the cumulative density function).
    Only available when compiled with PRRNG_USE_BOOST
    (e.g. using the CMake target `prrng::use_boost`).

    \param p Probability [0, 1].
    \return quantile for each `p`.
    */
    template <class T>
    T quantile(const T& p)
    {
        using value_type = typename T::value_type;

#if PRRNG_USE_BOOST
        auto f = xt::vectorize(boost::math::gamma_p_inv<value_type, value_type>);
        return m_scale * f(m_shape, p);
#else
        auto ret = p;
        ret.fill(std::numeric_limits<value_type>::quiet_NaN());
        return ret;
#endif
    }

private:
    double m_shape;
    double m_scale;
};

/**
Base class of the pseudorandom number generators.
This class provides common methods, but itself does not really do much.
*/
class GeneratorBase {
public:
    GeneratorBase() = default;

    virtual ~GeneratorBase() = default;

    /**
    Generate an nd-array of random numbers \f$ 0 \leq r \leq 1 \f$.

    \param shape The shape of the nd-array.
    \return The sample of shape `shape`.
    */
    template <class S>
    auto random(const S& shape) -> typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->random_impl<R>(shape);
    }

    /**
    \copydoc random(const S&)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class S>
    R random(const S& shape)
    {
        return this->random_impl<R>(shape);
    }

    /**
    \copydoc random(const S&)
    */
    template <class I, std::size_t L>
    auto random(const I (&shape)[L]) -> typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->random_impl<R>(shape);
    }

    /**
    \copydoc random(const S&)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class I, std::size_t L>
    R random(const I (&shape)[L])
    {
        return this->random_impl<R>(shape);
    }

    /**
    Generate an nd-array of random numbers distributed according to a normal distribution.
    Internally, the output of random() is converted using the cumulative density

    \f$ \Phi(x) = \frac{1}{2} \left[
        1 + \mathrm{erf}\left( \frac{x - \mu}{\sigma \sqrt{2}} \right)
    \right]\f$

    such that the output `r` from random() leads to

    \f$ x = \mu + \sigma \sqrt{2} \mathrm{erf}^{-1} (2r - 1) \f$

    \param shape The shape of the nd-array.
    \param mu The average.
    \param sigma The standard deviation.
    \return The sample of shape `shape`.
    */
    template <class S>
    auto normal(const S& shape, double mu = 0.0, double sigma = 1.0) ->
        typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->normal_impl<R>(shape, mu, sigma);
    }

    /**
    \copydoc normal(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class S>
    R normal(const S& shape, double mu = 0.0, double sigma = 1.0)
    {
        return this->normal_impl<R>(shape, mu, sigma);
    }

    /**
    \copydoc normal(const S&, double, double)
    */
    template <class I, std::size_t L>
    auto normal(const I (&shape)[L], double mu = 0.0, double sigma = 1.0) ->
        typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->normal_impl<R>(shape, mu, sigma);
    }

    /**
    \copydoc normal(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class I, std::size_t L>
    R normal(const I (&shape)[L], double mu = 0.0, double sigma = 1.0)
    {
        return this->normal_impl<R>(shape, mu, sigma);
    }

    /**
    Generate an nd-array of random numbers distributed according to a Weibull distribution.
    Internally, the output of random() is converted using the cumulative density

    \f$ \Phi(x) = 1 - e^{-(x / \lambda)^k} \f$

    such that the output `r` from random() leads to

    \f$ x = \lambda (- \ln (1 - r))^{1 / k}) \f$

    \param shape The shape of the nd-array.
    \param k The "shape" parameter.
    \param lambda The "scale" parameter.
    \return The sample of shape `shape`.
    */
    template <class S>
    auto weibull(const S& shape, double k = 1.0, double lambda = 1.0) ->
        typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->weibull_impl<R>(shape, k, lambda);
    }

    /**
    \copydoc weibull(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class S>
    R weibull(const S& shape, double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl<R>(shape, k, lambda);
    }

    /**
    \copydoc weibull(const S&, double, double)
    */
    template <class I, std::size_t L>
    auto weibull(const I (&shape)[L], double k = 1.0, double lambda = 1.0) ->
        typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->weibull_impl<R>(shape, k, lambda);
    }

    /**
    \copydoc weibull(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class I, std::size_t L>
    R weibull(const I (&shape)[L], double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl<R>(shape, k, lambda);
    }

    /**
    Generate an nd-array of random numbers distributed according to a Gamma distribution.
    Only available when compiled with PRRNG_USE_BOOST
    (e.g. using the CMake target `prrng::use_boost`).

    \param shape The shape of the nd-array.
    \param k The "shape" parameter.
    \param theta The "scale" parameter.
    \return The sample of shape `shape`.
    */
    template <class S>
    auto gamma(const S& shape, double k = 1.0, double theta = 1.0) ->
        typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->gamma_impl<R>(shape, k, theta);
    }

    /**
    \copydoc gamma(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class S>
    R gamma(const S& shape, double k = 1.0, double theta = 1.0)
    {
        return this->gamma_impl<R>(shape, k, theta);
    }

    /**
    \copydoc gamma(const S&, double, double)
    */
    template <class I, std::size_t L>
    auto gamma(const I (&shape)[L], double k = 1.0, double theta = 1.0) ->
        typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->gamma_impl<R>(shape, k, theta);
    }

    /**
    \copydoc gamma(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class I, std::size_t L>
    R gamma(const I (&shape)[L], double k = 1.0, double theta = 1.0)
    {
        return this->gamma_impl<R>(shape, k, theta);
    }

private:
    template <class R, class S>
    R random_impl(const S& shape)
    {
        static_assert(
            std::is_same<typename R::value_type, double>::value,
            "Return value_type must be double");

        R ret = xt::empty<typename R::value_type>(shape);
        this->draw_list(&ret.front(), ret.size());
        return ret;
    }

    template <class R, class S>
    R normal_impl(const S& shape, double mu, double sigma)
    {
        R r = this->random_impl<R>(shape);
        return normal_distribution(mu, sigma).quantile(r);
    }

    template <class R, class S>
    R weibull_impl(const S& shape, double k, double lambda)
    {
        R r = this->random_impl<R>(shape);
        return weibull_distribution(k, lambda).quantile(r);
    }

    template <class R, class S>
    R gamma_impl(const S& shape, double k, double theta)
    {
        R r = this->random_impl<R>(shape);
        return gamma_distribution(k, theta).quantile(r);
    }

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
class pcg32 : public GeneratorBase {
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
    }

    /**
    Draw new random number (uniformly distributed, `0 <= r <= max(uint32_t)`).
    This advances the state of the generator by one increment.

    \return Next random number in sequence.

    \author Melissa O'Neill, http://www.pcg-random.org.
    */
    uint32_t operator()()
    {
        uint64_t oldstate = m_state;
        m_state = oldstate * PRRNG_PCG32_MULT + m_inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    /**
    Draw new random number (uniformly distributed, `0 <= r <= max(uint32_t)`).
    This advances the state of the generator by one increment.

    \return Next random number in sequence.

    \note Wrapper around operator().
    */
    uint32_t next_uint32()
    {
        return (*this)();
    }

    /**
    Draw new random number (uniformly distributed, `0 <= r <= bound`).

    \param bound Bound on the return.
    \return Next random number in sequence.

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

        uint32_t threshold = (~bound + 1u) % bound;

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
    Generate a single precision floating point value on the interval [0, 1).

    \return Next random number in sequence.

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    float next_float()
    {
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
    Generate a double precision floating point value on the interval [0, 1).

    \return Next random number in sequence.

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

        x.u = ((uint64_t)next_uint32() << 20) | 0x3ff0000000000000ULL;

        return x.d - 1.0;
    }

    /**
    The current "state" of the generator. If the same initseq() is used, this exact point
    in the sequence can be restored with restore().

    \return State of the generator.
    */
    uint64_t state()
    {
        return m_state;
    }

    /**
    \copydoc state()

    \tparam R use a different return-type. There are some internal checks if the type is able to
    store the internal state of type `uint64_t`.
    */
    template <typename R>
    R state()
    {
        static_assert(
            std::numeric_limits<R>::max() >= std::numeric_limits<decltype(m_state)>::max(),
            "Down-casting not allowed.");

        static_assert(
            std::numeric_limits<R>::min() <= std::numeric_limits<decltype(m_state)>::min(),
            "Down-casting not allowed.");

        return static_cast<R>(m_state);
    }

    /**
    The state initiator that was used upon construction.

    \return initiator.
    */
    uint64_t initstate()
    {
        return m_initstate;
    }

    /**
    \copydoc initstate()

    \tparam R use a different return-type. There are some internal checks if the type is able to
    store the internal state of type `uint64_t`.
    */
    template <typename R>
    R initstate()
    {
        static_assert(
            std::numeric_limits<R>::max() >= std::numeric_limits<decltype(m_initstate)>::max(),
            "Down-casting not allowed.");

        static_assert(
            std::numeric_limits<R>::min() <= std::numeric_limits<decltype(m_initstate)>::min(),
            "Down-casting not allowed.");

        return static_cast<R>(m_initstate);
    }

    /**
    The sequence initiator that was used upon construction.

    \return initiator.
    */
    uint64_t initseq()
    {
        return m_initseq;
    }

    /**
    \copydoc initseq()

    \tparam R use a different return-type. There are some internal checks if the type is able to
    store the internal state of type `uint64_t`.
    */
    template <typename R>
    R initseq()
    {
        static_assert(
            std::numeric_limits<R>::max() >= std::numeric_limits<decltype(m_initseq)>::max(),
            "Down-casting not allowed.");

        static_assert(
            std::numeric_limits<R>::min() <= std::numeric_limits<decltype(m_initseq)>::min(),
            "Down-casting not allowed.");

        return static_cast<R>(m_initseq);
    }

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
    }

    /**
    \copydoc distance(const pcg32 &)
    */
    int64_t operator-(const pcg32& other) const
    {
        PRRNG_ASSERT(m_inc == other.m_inc);

        uint64_t cur_mult = PRRNG_PCG32_MULT;
        uint64_t cur_plus = m_inc;
        uint64_t cur_state = other.m_state;
        uint64_t the_bit = 1u;
        uint64_t distance = 0u;

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

        return (int64_t)distance;
    }

    /**
    The distance between two PCG32 pseudorandom number generators.

    \tparam R
        Return-type.
        `static_assert` against down-casting, #PRRNG_ASSERT against loss of signedness.

    \return Distance.

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    template <typename R = int64_t>
    R distance(const pcg32& other)
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
    The distance between two states.

    \tparam R
        Return-type.
        `static_assert` against down-casting, #PRRNG_ASSERT against loss of signedness.

    \return Distance.

    \warning The increment of used to generate must be the same. There is no way of checking here!

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    template <
        typename R = int64_t,
        typename T,
        std::enable_if_t<std::is_integral<T>::value, bool> = true>
    R distance(T other_state)
    {
        static_assert(sizeof(R) >= sizeof(int64_t), "Down-casting not allowed.");

        uint64_t cur_mult = PRRNG_PCG32_MULT;
        uint64_t cur_plus = m_inc;
        uint64_t cur_state = other_state;
        uint64_t the_bit = 1u;
        uint64_t distance = 0u;

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

        int64_t r = (int64_t)distance;

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

        uint64_t cur_mult = PRRNG_PCG32_MULT;
        uint64_t cur_plus = m_inc;
        uint64_t acc_mult = 1u;
        uint64_t acc_plus = 0u;

        /* Even though delta is an unsigned integer, we can pass a signed
           integer to go backwards, it just goes "the long way round". */
        uint64_t delta = (uint64_t)delta_;

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
            std::iter_swap(it, begin + next_uint32((uint32_t)(it - begin + 1)));
        }
    }

    /**
    Equality operator.

    \param other The generator to which to compare.

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    bool operator==(const pcg32& other) const
    {
        return m_state == other.m_state && m_inc == other.m_inc;
    }

    /**
    Inequality operator.

    \param other The generator to which to compare.

    \author Wenzel Jakob, https://github.com/wjakob/pcg32.
    */
    bool operator!=(const pcg32& other) const
    {
        return m_state != other.m_state || m_inc != other.m_inc;
    }

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
    }

private:
    uint64_t m_initstate; ///< State initiator
    uint64_t m_initseq; ///< Sequence initiator
    uint64_t m_state; ///< RNG state. All values are possible.
    uint64_t m_inc; ///< Controls which RNG sequence (stream) is selected. Must *always* be odd.
};

/**
Base class of an array of pseudorandom number generators.
This class provides common methods, but itself does not really do much.
See the description of derived classed for information.

\tparam M Type to use storage of the shape and array vectors. E.g. `std::vector` or `std::array`
*/
template <class M>
class GeneratorBase_array {
public:
    GeneratorBase_array() = default;

    virtual ~GeneratorBase_array() = default;

    /**
    Return the size of the array of generators.

    \return unsigned int
    */
    size_t size() const
    {
        return m_size;
    }

    /**
    Return the shape of the array of generators.

    \return vector of unsigned ints
    */
    M shape() const
    {
        return m_shape;
    }

    /**
    Return the shape of the array of generators along a specific axis.

    \param axis The axis.
    \return vector of unsigned ints
    */
    template <class T>
    size_t shape(T axis) const
    {
        return m_shape[axis];
    }

    /**
    Return a flat index based on an array index specified as a list.

    \param index Array index, e.g. as std::vector.
    \return Flat index.
    */
    template <class T>
    size_t flat_index(const T& index) const
    {
        PRRNG_ASSERT(this->inbounds(index));
        return std::inner_product(index.cbegin(), index.cend(), m_strides.cbegin(), 0);
    }

    /**
    Check if an index is in bounds (and of the correct rank).

    \return `false` if out-of-bounds, `true` otherwise.
    */
    template <class T>
    bool inbounds(const T& index) const
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
    Per generator, generate an nd-array of random numbers \f$ 0 \leq r \leq 1 \f$.

    \param ishape The shape of the nd-array drawn per generator.
    \return The array of arrays of samples: [#shape, `ishape`]
    */
    template <class S>
    auto random(const S& ishape) -> typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->random_impl<R>(ishape);
    }

    /**
    \copydoc random(const S&)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class S>
    R random(const S& ishape)
    {
        return this->random_impl<R>(ishape);
    }

    /**
    \copydoc random(const S&)
    */
    template <class I, std::size_t L>
    auto random(const I (&ishape)[L]) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->random_impl<R>(detail::to_array(ishape));
    }

    /**
    \copydoc random(const S&)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class I, std::size_t L>
    R random(const I (&ishape)[L])
    {
        return this->random_impl<R>(detail::to_array(ishape));
    }

    /**
    Per generator, generate an nd-array of random numbers distributed
    according to a normal distribution.
    Internally, the output of random() is converted using the cumulative density

    \f$ \Phi(x) = \frac{1}{2} \left[
        1 + \mathrm{erf}\left( \frac{x - \mu}{\sigma \sqrt{2}} \right)
    \right]\f$

    such that the output `r` from random() leads to

    \f$ x = \mu + \sigma \sqrt{2} \mathrm{erf}^{-1} (2r - 1) \f$

    \param ishape The shape of the nd-array drawn per generator.
    \param mu The average.
    \param sigma The standard deviation.
    \return The array of arrays of samples: [#shape, `ishape`]
    */
    template <class S>
    auto normal(const S& ishape, double mu = 0.0, double sigma = 1.0) ->
        typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->normal_impl<R>(ishape, mu, sigma);
    }

    /**
    \copydoc normal(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class S>
    R normal(const S& ishape, double mu = 0.0, double sigma = 1.0)
    {
        return this->normal_impl<R>(ishape, mu, sigma);
    }

    /**
    \copydoc normal(const S&, double, double)
    */
    template <class I, std::size_t L>
    auto normal(const I (&ishape)[L], double mu = 0.0, double sigma = 1.0) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->normal_impl<R>(detail::to_array(ishape), mu, sigma);
    }

    /**
    \copydoc normal(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class I, std::size_t L>
    R normal(const I (&ishape)[L], double mu = 0.0, double sigma = 1.0)
    {
        return this->normal_impl<R>(detail::to_array(ishape), mu, sigma);
    }

    /**
    Per generator, generate an nd-array of random numbers distributed
    according to a Weibull distribution.
    Internally, the output of random() is converted using the cumulative density

    \f$ \Phi(x) = 1 - e^{-(x / \lambda)^k} \f$

    such that the output `r` from random() leads to

    \f$ x = \lambda (- \ln (1 - r))^{1 / k}) \f$

    \param ishape The shape of the nd-array drawn per generator.
    \param k The "shape" parameter.
    \param lambda The "scale" parameter.
    \return The array of arrays of samples: [#shape, `ishape`]
    */
    template <class S>
    auto weibull(const S& ishape, double k = 1.0, double lambda = 1.0) ->
        typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->weibull_impl<R>(ishape, k, lambda);
    }

    /**
    \copydoc weibull(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class S>
    R weibull(const S& ishape, double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl<R>(ishape, k, lambda);
    }

    /**
    \copydoc weibull(const S&, double, double)
    */
    template <class I, std::size_t L>
    auto weibull(const I (&ishape)[L], double k = 1.0, double lambda = 1.0) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->weibull_impl<R>(detail::to_array(ishape), k, lambda);
    }

    /**
    \copydoc weibull(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class I, std::size_t L>
    R weibull(const I (&ishape)[L], double k = 1.0, double lambda = 1.0)
    {
        return this->weibull_impl<R>(detail::to_array(ishape), k, lambda);
    }

    /**
    Per generator, generate an nd-array of random numbers distributed
    according to a Gamma distribution.
    Only available when compiled with PRRNG_USE_BOOST
    (e.g. using the CMake target `prrng::use_boost`).

    \param ishape The shape of the nd-array drawn per generator.
    \param k The "shape" parameter.
    \param theta The "scale" parameter.
    \return The array of arrays of samples: [#shape, `ishape`]
    */
    template <class S>
    auto gamma(const S& ishape, double k = 1.0, double theta = 1.0) ->
        typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->gamma_impl<R>(ishape, k, theta);
    }

    /**
    \copydoc gamma(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class S>
    R gamma(const S& ishape, double k = 1.0, double theta = 1.0)
    {
        return this->gamma_impl<R>(ishape, k, theta);
    }

    /**
    \copydoc gamma(const S&, double, double)
    */
    template <class I, std::size_t L>
    auto gamma(const I (&ishape)[L], double k = 1.0, double theta = 1.0) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->gamma_impl<R>(detail::to_array(ishape), k, theta);
    }

    /**
    \copydoc gamma(const S&, double, double)
    \tparam R return type, e.g. `xt::xtensor<double, 1>`
    */
    template <class R, class I, std::size_t L>
    R gamma(const I (&ishape)[L], double k = 1.0, double theta = 1.0)
    {
        return this->gamma_impl<R>(detail::to_array(ishape), k, theta);
    }

private:
    template <class R, class S>
    R random_impl(const S& ishape)
    {
        static_assert(
            std::is_same<typename R::value_type, double>::value,
            "Return value_type must be double");

        auto n = detail::size(ishape);
        R ret = R::from_shape(detail::concatenate<M, S>::two(m_shape, ishape));
        this->draw_list(&ret.front(), n);
        return ret;
    }

    template <class R, class S>
    R normal_impl(const S& ishape, double mu, double sigma)
    {
        R r = this->random_impl<R>(ishape);
        return normal_distribution(mu, sigma).quantile(r);
    }

    template <class R, class S>
    R weibull_impl(const S& ishape, double k, double lambda)
    {
        R r = this->random_impl<R>(ishape);
        return weibull_distribution(k, lambda).quantile(r);
    }

    template <class R, class S>
    R gamma_impl(const S& ishape, double k, double theta)
    {
        R r = this->random_impl<R>(ishape);
        return gamma_distribution(k, theta).quantile(r);
    }

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
    size_t m_size = 0; ///< See size().
    M m_shape; ///< See shape().
    M m_strides; ///< The strides of the array of generators.
};

/**
Base class, see pcg32_array for description.
*/
template <class M>
class pcg32_arrayBase : public GeneratorBase_array<M> {
public:
    pcg32_arrayBase() = default;

    virtual ~pcg32_arrayBase() = default;

    /**
    Return a reference to one generator, using an array index.

    \param args Array index (number of arguments should correspond to the rank of the array).
    \return Reference to underlying generator.
    */
    template <class... Args>
    pcg32& operator()(Args... args)
    {
        return m_gen[this->get_item(0, 0, args...)];
    }

    /**
    Return a constant reference to one generator, using an array index.

    \param args Array index (number of arguments should correspond to the rank of the array).
    \return Reference to underlying generator.
    */
    template <class... Args>
    const pcg32& operator()(Args... args) const
    {
        return m_gen[this->get_item(0, 0, args...)];
    }

    /**
    Return a reference to one generator, using a flat index.

    \param i Flat index.
    \return Reference to underlying generator.
    */
    pcg32& operator[](size_t i)
    {
        PRRNG_ASSERT(i < m_size);
        return m_gen[i];
    }

    /**
    Return a constant reference to one generator, using a flat index.

    \param i Flat index.
    \return Reference to underlying generator.
    */
    const pcg32& operator[](size_t i) const
    {
        PRRNG_ASSERT(i < m_size);
        return m_gen[i];
    }

    /**
    Return a reference to one generator, using a flat index.

    \param i Flat index.
    \return Reference to underlying generator.
    */
    pcg32& flat(size_t i)
    {
        PRRNG_ASSERT(i < m_size);
        return m_gen[i];
    }

    /**
    Return a constant reference to one generator, using a flat index.

    \param i Flat index.
    \return Reference to underlying generator.
    */
    const pcg32& flat(size_t i) const
    {
        PRRNG_ASSERT(i < m_size);
        return m_gen[i];
    }

    /**
    Return the state of all generators.
    See pcg32::state().

    \return The state of each generator.
    */
    auto state() -> typename detail::return_type<uint64_t, M>::type
    {
        using R = typename detail::return_type<uint64_t, M>::type;
        return this->state<R>();
    }

    /**
    \copydoc state()

    \tparam R The type of the return array, e.g. `xt::array<uint64_t>` or `xt::xtensor<uint64_t, N>`
    */
    template <class R>
    R state()
    {
        using value_type = typename R::value_type;
        R ret = R::from_shape(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            ret.flat(i) = m_gen[i].template state<value_type>();
        }

        return ret;
    }

    /**
    Return the state initiator of all generators.
    See pcg32::initstate().

    \return The state initiator of each generator.
    */
    auto initstate() -> typename detail::return_type<uint64_t, M>::type
    {
        using R = typename detail::return_type<uint64_t, M>::type;
        return this->initstate<R>();
    }

    /**
    \copydoc initstate()

    \return The state initiator of each generator.
    */
    template <class R>
    R initstate()
    {
        using value_type = typename R::value_type;
        R ret = R::from_shape(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            ret.flat(i) = m_gen[i].template initstate<value_type>();
        }

        return ret;
    }

    /**
    Return the sequence initiator of all generators.
    See pcg32::initseq().

    \return The sequence initiator of each generator.
    */
    auto initseq() -> typename detail::return_type<uint64_t, M>::type
    {
        using R = typename detail::return_type<uint64_t, M>::type;
        return this->initseq<R>();
    }

    /**
    \copydoc initseq()

    \tparam R The type of the return array, e.g. `xt::array<uint64_t>` or `xt::xtensor<uint64_t, N>`
    */
    template <class R>
    R initseq()
    {
        using value_type = typename R::value_type;
        R ret = R::from_shape(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            ret.flat(i) = m_gen[i].template initseq<value_type>();
        }

        return ret;
    }

    /**
    Distance between two states.
    See pcg32::distance().

    \tparam T Array, e.g. `xt::array<int64_t>` or `xt::xtensor<int64_t, N>`.
    \param arg The state to which to compare.
    */
    template <class T>
    auto distance(const T& arg) -> typename detail::return_type<int64_t, M>::type
    {
        using R = typename detail::return_type<int64_t, M>::type;
        return this->distance<R, T>(arg);
    }

    /**
    Distance between two states.
    See pcg32::distance().

    \tparam R Array, e.g. `xt::array<int64_t>` or `xt::xtensor<int64_t, N>`.
    \tparam T Array, e.g. `xt::array<int64_t>` or `xt::xtensor<int64_t, N>`.
    \param arg The state to which to compare.
    */
    template <class R, class T>
    R distance(const T& arg)
    {
        using value_type = typename R::value_type;
        R ret = R::from_shape(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            ret.flat(i) = m_gen[i].template distance<value_type>(arg.flat(i));
        }

        return ret;
    }

    /**
    Advance generators.
    See pcg32::advance().

    \tparam T The type of the input array, e.g. `xt::array<int64_t>` or `xt::xtensor<int64_t, N>`

    \param arg The distance (positive or negative) by which to advance each generator.
    */
    template <class T>
    void advance(const T& arg)
    {
        for (size_t i = 0; i < m_size; ++i) {
            m_gen[i].advance(arg.flat(i));
        }
    }

    /**
    Restore generators from a state.
    See pcg32::restore().

    \tparam T The type of the input array, e.g. `xt::array<uint64_t>` or `xt::xtensor<uint64_t, N>`

    \param arg The state of each generator.
    */
    template <class T>
    void restore(const T& arg)
    {
        for (size_t i = 0; i < m_size; ++i) {
            m_gen[i].restore(arg.flat(i));
        }
    }

protected:
    /**
    Draw `n` random numbers per array item, and write them to the correct position in `data`
    (assuming row-major storage!).

    \param data Pointer to the data (no bounds-check).
    \param n The number of random numbers per generator.
    */
    void draw_list(double* data, size_t n) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            for (size_t j = 0; j < n; ++j) {
                data[i * n + j] = m_gen[i].next_double();
            }
        }
    }

private:
    /**
    implementation of `operator()`.
    (Last call in recursion).
    */
    template <class T>
    size_t get_item(size_t sum, size_t d, T arg)
    {
        return sum + arg * m_strides[d];
    }

    /**
    implementation of `operator()`.
    (Called recursively).
    */
    template <class T, class... Args>
    size_t get_item(size_t sum, size_t d, T arg, Args... args)
    {
        return get_item(sum + arg * m_strides[d], d + 1, args...);
    }

protected:
    std::vector<pcg32> m_gen; ///< Underlying storage: one generator per array item
    using GeneratorBase_array<M>::m_size;
    using GeneratorBase_array<M>::m_shape;
    using GeneratorBase_array<M>::m_strides;
};

/**
Array of independent generators.
The idea is that each array-entry has its own random sequence, initiated by its own seed.
An array of random numbers can then be generated whose shape if composed of the #shape,
the shape of the array of generators, followed by the desired shape of the random sequence
draw per generator.
Let us consider an example. Suppose that we have a list of n = 5 generators,
and we want to generate i = 8 random numbers for each generator.
Then the output will be collected in a matrix of shape [n, i] = [5, 8] where each row
corresponds to a generator and the columns for that row are the random sequence generated
by that generator.
Since this class is general, you can also imagine an array of [m, n, o, p] generators with
a random sequence reshaped in a (row-major) array of shape [a, b, c, d, e].
The output is then collected in an array of shape [m, n, o, p, a, b, c, d, e].

Note that a reference to each generator can be obtained using the `[]` and `()` operators,
e.g. `generators[flat_index]` and `generators(i, j, k, ...)`. All functions of pcg32()
can be used for each reference.
In addition, convenience functions state(), initstate(), initseq(), restore() are provided
here to store/restore the state of the entire array of generators.
*/
class pcg32_array : public pcg32_arrayBase<std::vector<size_t>> {
public:
    /**
    Constructor.

    \param initstate State initiator for every item (accept default sequence initiator).
    The shape of the argument determines the shape of the generator array.
    */
    template <class T>
    pcg32_array(const T& initstate)
    {
        m_shape = std::vector<size_t>(initstate.shape().cbegin(), initstate.shape().cend());
        m_strides = std::vector<size_t>(initstate.strides().cbegin(), initstate.strides().cend());
        m_size = initstate.size();
        m_gen.reserve(m_size);

        for (size_t i = 0; i < m_size; ++i) {
            m_gen.push_back(pcg32(initstate.flat(i)));
        }
    }

    /**
    Constructor.

    \param initstate State initiator for every item.
    \param initseq Sequence initiator for every item.
    The shape of these argument determines the shape of the generator array.
    */
    template <class T, class U>
    pcg32_array(const T& initstate, const U& initseq)
    {
        PRRNG_ASSERT(xt::same_shape(initstate.shape(), initseq.shape()));

        m_shape = std::vector<size_t>(initstate.shape().cbegin(), initstate.shape().cend());
        m_strides = std::vector<size_t>(initstate.strides().cbegin(), initstate.strides().cend());
        m_size = initstate.size();
        m_gen.reserve(m_size);

        for (size_t i = 0; i < m_size; ++i) {
            m_gen.push_back(pcg32(initstate.flat(i), initseq.flat(i)));
        }
    }

protected:
    using pcg32_arrayBase<std::vector<size_t>>::m_gen;
    using GeneratorBase_array<std::vector<size_t>>::m_size;
    using GeneratorBase_array<std::vector<size_t>>::m_shape;
    using GeneratorBase_array<std::vector<size_t>>::m_strides;
};

/**
Fixed rank version of pcg32_array
*/
template <size_t N>
class pcg32_tensor : public pcg32_arrayBase<std::array<size_t, N>> {
public:
    /**
    Constructor.

    \param initstate State initiator for every item (accept default sequence initiator).
    The shape of the argument determines the shape of the generator array.
    */
    template <class T>
    pcg32_tensor(const T& initstate)
    {
        static_assert(detail::check_fixed_rank<N, T>::value, "Ranks to not match");

        std::copy(initstate.shape().cbegin(), initstate.shape().cend(), m_shape.begin());
        std::copy(initstate.strides().cbegin(), initstate.strides().cend(), m_strides.begin());
        m_size = initstate.size();
        m_gen.reserve(m_size);

        for (size_t i = 0; i < m_size; ++i) {
            m_gen.push_back(pcg32(initstate.flat(i)));
        }
    }

    /**
    Constructor.

    \param initstate State initiator for every item.
    \param initseq Sequence initiator for every item.
    The shape of these argument determines the shape of the generator array.
    */
    template <class T, class U>
    pcg32_tensor(const T& initstate, const U& initseq)
    {
        static_assert(detail::check_fixed_rank<N, T>::value, "Ranks to not match");
        PRRNG_ASSERT(xt::same_shape(initstate.shape(), initseq.shape()));

        std::copy(initstate.shape().cbegin(), initstate.shape().cend(), m_shape.begin());
        std::copy(initstate.strides().cbegin(), initstate.strides().cend(), m_strides.begin());
        m_size = initstate.size();
        m_gen.reserve(m_size);

        for (size_t i = 0; i < m_size; ++i) {
            m_gen.push_back(pcg32(initstate.flat(i), initseq.flat(i)));
        }
    }

protected:
    using pcg32_arrayBase<std::array<size_t, N>>::m_gen;
    using GeneratorBase_array<std::array<size_t, N>>::m_size;
    using GeneratorBase_array<std::array<size_t, N>>::m_shape;
    using GeneratorBase_array<std::array<size_t, N>>::m_strides;
};

namespace detail {

template <class T, typename = void>
struct auto_pcg32 {
    static auto get(const T& initseq)
    {
        return pcg32_array(initseq);
    }

    template <class S>
    static auto get(const T& initseq, const S& initstate)
    {
        return pcg32_array(initseq, initstate);
    }
};

template <class T>
struct auto_pcg32<T, typename std::enable_if_t<xt::has_fixed_rank_t<T>::value>> {
    static auto get(const T& initseq)
    {
        return pcg32_tensor<xt::get_rank<T>::value>(initseq);
    }

    template <class S>
    static auto get(const T& initseq, const S& initstate)
    {
        return pcg32_tensor<xt::get_rank<T>::value>(initseq, initstate);
    }
};

template <class T>
struct auto_pcg32<T, typename std::enable_if_t<std::is_integral<T>::value>> {
    static auto get(const T& initseq)
    {
        return pcg32(initseq);
    }

    template <class S>
    static auto get(const T& initseq, const S& initstate)
    {
        return pcg32(initseq, initstate);
    }
};

} // namespace detail

/**
Return a pcg32, a pcg32_array, or a pcg32_tensor based on input.

\param initstate The sequence initiator.
\return The allocated generator.
*/
template <class T>
inline auto auto_pcg32(const T& initstate)
{
    return detail::auto_pcg32<T>::get(initstate);
}

/**
Return a pcg32, a pcg32_array, or a pcg32_tensor based on input.

\param initstate The sequence initiator.
\param initseq The sequence initiator.
\return The allocated generator.
*/
template <class T, class S>
inline auto auto_pcg32(const T& initstate, const S& initseq)
{
    return detail::auto_pcg32<T>::get(initstate, initseq);
}

} // namespace prrng

#endif
