/**
 * Portable Reconstructible Random Number Generator.
 * The idea is that a random sequence can be restored independent of platform or compiler.
 * In addition, this library allows you to store a point in the sequence, and then later restore
 * the sequence exactly from this point (in both directions actually).
 *
 * Note that the core of this code is taken from
 * https://github.com/imneme/pcg-c-basic
 * All the credits goes to those developers.
 * This is just a wrapper.
 *
 * @file prrng.h
 * @copyright Copyright 2021. Tom de Geus. All rights reserved.
 * @par license This project is released under the MIT License.
 */

#ifndef PRRNG_H
#define PRRNG_H

/**
 * Default initialisation state for pcg32()
 * (used as constructor parameter that can be overwritten at run-time).
 */
#define PRRNG_PCG32_INITSTATE 0x853c49e6748fea9bULL

/**
 * Default initialisation sequence for pcg32()
 * (used as constructor parameter that can be overwritten at run-time).
 */
#define PRRNG_PCG32_INITSEQ 0xda3e39cb94b95bdbULL

/**
 * Multiplicative factor for pcg32()
 * (used internally, cannot be overwritten at run-time).
 */
#define PRRNG_PCG32_MULT 6364136223846793005ULL

#include <array>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

#ifndef PRRNG_USE_BOOST
/**
 * To use prrng without Boost
 *
 *     #define PRRNG_USE_BOOST 0
 *     #include <prrng.h>
 *
 * You will loose the normal and Gamma distributions.
 */
#define PRRNG_USE_BOOST 1
#endif

#if PRRNG_USE_BOOST
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <xtensor/xvectorize.hpp>
#endif

/**
 * \cond
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
 * \endcond
 */

/**
 * Library version.
 *
 * Either:
 *
 * -   Configure using CMake at install time. Internally uses:
 *
 *         python -c "from setuptools_scm import get_version; print(get_version())"
 *
 * -   Define externally using:
 *
 *         -DPRRNG_VERSION="`python -c "from setuptools_scm import get_version;
 * print(get_version())"`"
 *
 *     From the root of this project. This is what ``setup.py`` does.
 *
 * Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using
 * ``setuptools_scm``. Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION`` to
 * overwrite the automatic version.
 */
#ifndef PRRNG_VERSION
#define PRRNG_VERSION "@PROJECT_VERSION@"
#endif

/**
 * All assertions are implementation as:
 *
 *     PRRNG_ASSERT(...)
 *
 * They can be enabled by:
 *
 *     #define PRRNG_ENABLE_ASSERT
 *
 * (before including prrng).
 * The advantage is that:
 *
 * -   File and line-number are displayed if the assertion fails.
 * -   prrng's assertions can be enabled/disabled independently from those of other libraries.
 *
 * \throw std::runtime_error
 */
#ifdef PRRNG_ENABLE_ASSERT
#define PRRNG_ASSERT(expr) PRRNG_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define PRRNG_ASSERT(expr)
#endif

/**
 * All debug assertions are implementation as:
 *
 *     PRRNG_DEBUG(...)
 *
 * They can be enabled by:
 *
 *     #define PRRNG_ENABLE_DEBUG
 *
 * (before including prrng).
 * The advantage is that:
 *
 * -   File and line-number are displayed if the assertion fails.
 * -   prrng's assertions can be enabled/disabled independently from those of other libraries.
 *
 * \throw std::runtime_error
 */
#ifdef PRRNG_ENABLE_DEBUG
#define PRRNG_DEBUG(expr) PRRNG_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define PRRNG_DEBUG(expr)
#endif

/**
 * @brief Portable Reconstructible (Pseudo!) Random Number Generator
 */
namespace prrng {

namespace detail {

/**
 * Remove " from string.
 *
 * @param arg Input string.
 * @return String without "
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

/**
 * Check that an object has a certain fixed rank.
 */
template <size_t N, class T, typename = void>
struct check_fixed_rank {
    constexpr static bool value = false;
};

template <size_t N, class T>
struct check_fixed_rank<N, T, typename std::enable_if_t<xt::get_rank<T>::value != SIZE_MAX>> {
    constexpr static bool value = (N == xt::get_rank<T>::value);
};

template <size_t N, class T>
struct check_fixed_rank<N, T, typename std::enable_if_t<is_std_array<T>::value>> {
    constexpr static bool value = (N == std::tuple_size<T>::value);
};

/**
 * Get value type
 */
template <typename R, typename = void>
struct get_value_type {
    using type = typename R::value_type;
};

template <typename R>
struct get_value_type<R, typename std::enable_if_t<std::is_arithmetic<R>::value>> {
    using type = R;
};

/**
 * Get default return type
 */
template <typename R, size_t N>
struct return_type_fixed {
    using type = typename xt::xtensor<R, N>;
};

/**
 * Get default return type
 */
template <typename R, class S, typename = void>
struct return_type {
    using type = typename xt::xarray<R>;
};

template <typename R, class S>
struct return_type<R, S, typename std::enable_if_t<is_std_array<S>::value>> {
    using type = typename xt::xtensor<R, std::tuple_size<S>::value>;
};

template <typename R, class S>
struct return_type<R, S, typename std::enable_if_t<xt::has_fixed_rank_t<S>::value>> {
    using type = typename xt::xtensor<R, xt::get_rank<S>::value>;
};

/**
 * Get default return type
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
 * Allocate return: scalar or array
 */
template <typename R, typename = void>
struct allocate_return {

    using value_type = typename R::value_type;
    R value;

    template <class S>
    allocate_return(const S& shape)
    {
        value = xt::empty<typename R::value_type>(shape);
    }

    typename R::value_type* data()
    {
        return &value.front();
    }

    size_t size() const
    {
        return value.size();
    }
};

template <typename R>
struct allocate_return<R, typename std::enable_if_t<std::is_arithmetic<R>::value>> {

    using value_type = R;
    R value;

    template <class S>
    allocate_return(const S& shape)
    {
#ifdef PRRNG_ENABLE_ASSERT
        PRRNG_ASSERT(shape.size() <= 1);
        if (shape.size() == 1) {
            PRRNG_ASSERT(shape[0] == 1);
        }
#else
        (void)(shape);
#endif
    }

    R* data()
    {
        return &value;
    }

    size_t size() const
    {
        return 1;
    }
};

/**
 * Concatenate two objects that have `begin()` and `end()` methods.
 *
 * @param s First object (e.g. std::vector).
 * @param t Second object (e.g. std::vector).
 * @return Concatenated [t, s]
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
 * Compute 'size' from 'shape'.
 *
 * @param shape Shape array.
 * @return Size
 */
template <class S>
inline size_t size(const S& shape)
{
    using ST = typename S::value_type;
    return std::accumulate(shape.cbegin(), shape.cend(), ST(1), std::multiplies<ST>());
}

/**
 * Return as std::array.
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
 * Return version string, for example `"0.1.0"`
 *
 * @return std::string
 */
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(PRRNG_VERSION)));
}

/**
 * Normal distribution.
 *
 * References:
 *
 * -   https://en.wikipedia.org/wiki/Normal_distribution
 * -   https://www.boost.org/doc/libs/1_63_0/libs/math/doc/html/math_toolkit/sf_erf/error_inv.html
 * -   https://www.boost.org/doc/libs/1_78_0/boost/math/special_functions/detail/erf_inv.hpp
 */
class normal_distribution {
public:
    /**
     * Constructor.
     *
     * @param mu Average.
     * @param sigma Standard deviation.
     */
    normal_distribution(double mu = 0, double sigma = 1)
    {
        m_mu = mu;
        m_sigma = sigma;
        m_sigma_sqrt2 = m_sigma * std::sqrt(2.0);
    }

    /**
     * Probability density function.
     *
     * @param x Coordinates.
     * @return Probability density for each `x`.
     */
    template <class T>
    T pdf(const T& x)
    {
        return xt::exp(-0.5 * xt::pow((x - m_mu) / m_sigma, 2.0)) /
               (m_sigma_sqrt2 * std::sqrt(xt::numeric_constants<double>::PI));
    }

    /**
     * Cumulative density function.
     *
     * @param x Coordinates.
     * @return Cumulative density for each `x`.
     */
    template <class T>
    T cdf(const T& x)
    {
        return 0.5 * (1.0 + xt::erf((x - m_mu) / m_sigma_sqrt2));
    }

    /**
     * Quantile (the inverse of the cumulative density function).
     *
     * @param p Probability [0, 1].
     * @return Quantile for each `p`.
     */
    template <class T>
    T quantile(const T& p)
    {
        using value_type = typename detail::get_value_type<T>::type;

#if PRRNG_USE_BOOST
        auto f = xt::vectorize(boost::math::erf_inv<value_type>);
        return m_mu + m_sigma_sqrt2 * f(2.0 * p - 1.0);
#else
        static_assert(xt::is_xexpression<T>::value, "T must be an xexpression");
        auto ret = p;
        ret.fill(std::numeric_limits<value_type>::quiet_NaN());
        return ret;
#endif
    }

    /**
     * @brief Apply the quantile in place.
     * @param p Probability [0, 1], modified in place.
     */
    template <class T>
    void apply_quantile(T* p)
    {
#if PRRNG_USE_BOOST
        *p = m_mu + m_sigma_sqrt2 * boost::math::erf_inv<T>(2.0 * (*p) - 1.0);
#else
        *p = std::numeric_limits<T>::quiet_NaN();
#endif
    }

private:
    double m_mu;
    double m_sigma;
    double m_sigma_sqrt2;
};

/**
 * Exponential distribution.
 *
 * References:
 *
 * -   https://en.wikipedia.org/wiki/Exponential_distribution
 */
class exponential_distribution {
public:
    /**
     * Constructor.
     *
     * @param scale Scale (inverse rate).
     */
    exponential_distribution(double scale = 1)
    {
        m_scale = scale;
    }

    /**
     * Probability density function.
     *
     * @param x Coordinates.
     * @return Probability density for each `x`.
     */
    template <class T>
    T pdf(const T& x)
    {
        double rate = 1.0 / m_scale;
        return rate * xt::exp(-rate * x);
    }

    /**
     * Cumulative density function.
     *
     * @param x Coordinates.
     * @return Cumulative density for each `x`.
     */
    template <class T>
    T cdf(const T& x)
    {
        double rate = 1.0 / m_scale;
        return 1.0 - xt::exp(-rate * x);
    }

    /**
     * Quantile (the inverse of the cumulative density function).
     *
     * @param p Probability [0, 1].
     * @return Quantile for each `p`.
     */
    template <class T>
    T quantile(const T& p)
    {
        return -xt::log(1.0 - p) * m_scale;
    }

    /**
     * @brief Apply the quantile in place.
     * @param p Probability [0, 1], modified in place.
     */
    template <class T>
    void apply_quantile(T* p)
    {
        *p = -std::log(1.0 - (*p)) * m_scale;
    }

private:
    double m_scale;
};

/**
 * Weibull distribution.
 *
 * References:
 *
 * -   https://en.wikipedia.org/wiki/Weibull_distribution
 * -   https://github.com/boostorg/math/blob/develop/include/boost/math/distributions/weibull.hpp
 */
class weibull_distribution {
public:
    /**
     * Constructor.
     *
     * @param k Shape parameter \f$ k \f$.
     * @param scale Scale parameter \f$ \lambda \f$.
     */
    weibull_distribution(double k = 1, double scale = 1)
    {
        m_shape = k;
        m_scale = scale;
    }

    /**
     * Probability density function.
     *
     * @param x Coordinates.
     * @return Probability density for each `x`.
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
     * Cumulative density function.
     *
     * \f$ \Phi(x) = 1 - e^{-(x / \lambda)^k} \f$
     *
     * @param x Coordinates.
     * @return Cumulative density for each `x`.
     */
    template <class T>
    T cdf(const T& x)
    {
        return -xt::expm1(-xt::pow(x / m_scale, m_shape));
    }

    /**
     * Quantile (the inverse of the cumulative density function).
     * For a given probability \f$ p \f$ the output is
     *
     * \f$ x = \lambda (- \ln (1 - p))^{1 / k}) \f$
     *
     * @param p Probability [0, 1].
     * @return Quantile for each `p`.
     */
    template <class T>
    T quantile(const T& p)
    {
        return m_scale * xt::pow(-xt::log1p(-p), 1.0 / m_shape);
    }

    /**
     * @brief Apply the quantile in place.
     * @param p Probability [0, 1], modified in place.
     */
    template <class T>
    void apply_quantile(T* p)
    {
        *p = m_scale * std::pow(-std::log(1 - (*p)), 1.0 / m_shape);
    }

private:
    double m_shape;
    double m_scale;
};

/**
 * Gamma distribution.
 *
 * References:
 *
 * -   https://en.wikipedia.org/wiki/Gamma_distribution
 * -   https://github.com/boostorg/math/blob/develop/include/boost/math/distributions/gamma.hpp
 */
class gamma_distribution {
public:
    /**
     * Constructor.
     *
     * @param k Shape parameter \f$ \k \f$.
     * @param scale Scale parameter \f$ \theta \f$.
     */
    gamma_distribution(double k = 1, double scale = 1)
    {
        m_shape = k;
        m_scale = scale;
    }

    /**
     * Probability density function.
     * Only available when compiled with PRRNG_USE_BOOST
     * (e.g. using the CMake target `prrng::use_boost`).
     *
     * @param x Coordinates.
     * @return Probability density for each `x`.
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
     * Cumulative density function.
     * Only available when compiled with PRRNG_USE_BOOST
     * (e.g. using the CMake target `prrng::use_boost`).
     *
     * @param x Coordinates.
     * @return Cumulative density for each `x`.
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
     * Quantile (the inverse of the cumulative density function).
     * Only available when compiled with PRRNG_USE_BOOST
     * (e.g. using the CMake target `prrng::use_boost`).
     *
     * @param p Probability [0, 1].
     * @return Quantile for each `p`.
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

    /**
     * @brief Apply the quantile in place.
     * @param p Probability [0, 1], modified in place.
     */
    template <class T>
    void apply_quantile(T* p)
    {
#if PRRNG_USE_BOOST
        *p = m_scale * boost::math::gamma_p_inv<T, T>(m_shape, *p);
#else
        *p = std::numeric_limits<T>::quiet_NaN();
#endif
    }

private:
    double m_shape;
    double m_scale;
};

/**
 * Base class of the pseudorandom number generators.
 * This class provides common methods, but itself does not really do much.
 */
class GeneratorBase {
public:
    GeneratorBase() = default;

    virtual ~GeneratorBase() = default;

    /**
     * @brief Result of the cumulative sum of `n` random numbers.
     * @param n Number of steps.
     * @return Cumulative sum.
     */
    double cumsum_random(size_t n)
    {
        double ret = 0.0;
        for (size_t i = 0; i < n; ++i) {
            ret += draw_double();
        }
        return ret;
    }

    /**
     * @brief Result of the cumulative sum of `n` random numbers, distributed according to a
     * normal distribution, see normal_distribution(),
     * @param n Number of steps.
     * @param mu Mean.
     * @param sigma Standard deviation.
     * @return Cumulative sum.
     */
    double cumsum_normal(size_t n, double mu = 0, double sigma = 1)
    {
        double ret = 0.0;
        auto tranform = normal_distribution(mu, sigma);
        for (size_t i = 0; i < n; ++i) {
            auto r = draw_double();
            tranform.apply_quantile(&r);
            ret += r;
        }
        return ret;
    }

    /**
     * @brief Result of the cumulative sum of `n` random numbers, distributed according to an
     * exponential distribution, see exponential_distribution(),
     * @param n Number of steps.
     * @param scale Scale.
     * @return Cumulative sum.
     */
    double cumsum_exponential(size_t n, double scale = 1)
    {
        double ret = 0.0;
        auto tranform = exponential_distribution(scale);
        for (size_t i = 0; i < n; ++i) {
            auto r = draw_double();
            tranform.apply_quantile(&r);
            ret += r;
        }
        return ret;
    }

    /**
     * @brief Result of the cumulative sum of `n` random numbers, distributed according to a
     * weibull distribution, see weibull_distribution(),
     * @param n Number of steps.
     * @param k Shape.
     * @param scale Scale.
     * @return Cumulative sum.
     */
    double cumsum_weibull(size_t n, double k = 1, double scale = 1)
    {
        double ret = 0.0;
        auto tranform = weibull_distribution(k, scale);
        for (size_t i = 0; i < n; ++i) {
            auto r = draw_double();
            tranform.apply_quantile(&r);
            ret += r;
        }
        return ret;
    }

    /**
     * @brief Result of the cumulative sum of `n` random numbers, distributed according to a
     * gamma distribution, see gamma_distribution(),
     * @param n Number of steps.
     * @param k Shape.
     * @param scale Scale.
     * @return Cumulative sum.
     */
    double cumsum_gamma(size_t n, double k = 1, double scale = 1)
    {
        double ret = 0.0;
        auto tranform = gamma_distribution(k, scale);
        for (size_t i = 0; i < n; ++i) {
            auto r = draw_double();
            tranform.apply_quantile(&r);
            ret += r;
        }
        return ret;
    }

    /**
     * @brief Decide based on probability per value.
     * This is fully equivalent to `generator.random(p.shape) < p`,
     * but avoids the memory allocation of `random`.
     *
     * @param p Probability per value [0, 1].
     * @return Decision for each `p`.
     */
    template <class P>
    auto decide(const P& p) -> typename detail::return_type<bool, P>::type
    {
        using R = typename detail::return_type<bool, P>::type;
        R ret = xt::empty<bool>(p.shape());
        this->decide(p, ret);
        return ret;
    }

    /**
     * @copydoc prrng::GeneratorBase::decide(const P&)
     */
    template <class P, class R>
    R decide(const P& p)
    {
        using value_type = typename detail::get_value_type<R>::type;
        R ret = xt::empty<value_type>(p.shape());
        this->decide(p, ret);
        return ret;
    }

    /**
     * @brief Decide based on probability per value.
     * This is fully equivalent to `generator.random(p.shape) < p`,
     * but avoids the memory allocation of `random`.
     *
     * @param p Probability per value [0, 1].
     * @param ret Decision for each ``p``.
     */
    template <class P, class R>
    void decide(const P& p, R& ret)
    {
        PRRNG_ASSERT(xt::has_shape(p, ret.shape()));
        using value_type = typename detail::get_value_type<R>::type;

        for (size_t i = 0; i < p.size(); ++i) {
            if (draw_double() < p.flat(i)) {
                ret.flat(i) = static_cast<value_type>(true);
            }
            else {
                ret.flat(i) = static_cast<value_type>(false);
            }
        }
    }

    /**
     * @brief Decide based on probability per value.
     * This is fully equivalent to `generator.random(p.shape) < p`,
     * but avoids the memory allocation of `random`.
     *
     * @param p Probability per value [0, 1].
     * @param mask Where `true` the decision is `false` (no random number is drawn there).
     * @return Decision for each `p`.
     */
    template <class P, class T>
    auto decide_masked(const P& p, const T& mask) -> typename detail::return_type<bool, P>::type
    {
        using R = typename detail::return_type<bool, P>::type;
        R ret = xt::empty<bool>(p.shape());
        this->decide_masked(p, mask, ret);
        return ret;
    }

    /**
     * @copydoc prrng::GeneratorBase::decide_masked(const P&, const T&)
     */
    template <class P, class T, class R>
    R decide_masked(const P& p, const T& mask)
    {
        using value_type = typename detail::get_value_type<R>::type;
        R ret = xt::empty<value_type>(p.shape());
        this->decide_masked(p, mask, ret);
        return ret;
    }

    /**
     * @brief Decide based on probability per value.
     * This is fully equivalent to `generator.random(p.shape) < p`,
     * but avoids the memory allocation of `random`.
     *
     * @param p Probability per value [0, 1].
     * @param mask Where `true` the decision is `false` (no random number is drawn there).
     * @param ret Decision for each ``p``.
     */
    template <class P, class T, class R>
    void decide_masked(const P& p, const T& mask, R& ret)
    {
        PRRNG_ASSERT(xt::has_shape(p, ret.shape()));
        using value_type = typename detail::get_value_type<R>::type;

        for (size_t i = 0; i < p.size(); ++i) {
            if (mask.flat(i)) {
                ret.flat(i) = static_cast<value_type>(false);
            }
            else if (draw_double() < p.flat(i)) {
                ret.flat(i) = static_cast<value_type>(true);
            }
            else {
                ret.flat(i) = static_cast<value_type>(false);
            }
        }
    }

    /**
     * Generate an nd-array of random numbers \f$ 0 \leq r \leq 1 \f$.
     *
     * @param shape The shape of the nd-array.
     * @return The sample of shape `shape`.
     */
    template <class S>
    auto random(const S& shape) -> typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->random_impl<R>(shape);
    }

    /**
     * @copydoc prrng::GeneratorBase::random(const S&)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R random(const S& shape)
    {
        return this->random_impl<R>(shape);
    }

    /**
     * @copydoc prrng::GeneratorBase::random(const S&)
     */
    template <class I, std::size_t L>
    auto random(const I (&shape)[L]) -> typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->random_impl<R>(shape);
    }

    /**
     * @copydoc prrng::GeneratorBase::random(const S&)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R random(const I (&shape)[L])
    {
        return this->random_impl<R>(shape);
    }

    /**
     * Generate an nd-array of random integers \f$ 0 \leq r \leq bound \f$.
     *
     * @param shape The shape of the nd-array.
     * @param high The upper bound of the random integers.
     * @return The sample of shape `shape`.
     */
    template <class S, typename T>
    auto randint(const S& shape, T high) -> typename detail::return_type<T, S>::type
    {
        using R = typename detail::return_type<T, S>::type;
        return this->randint_impl<R>(shape, high);
    }

    /**
     * @copydoc prrng::GeneratorBase::randint(const S&, T)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S, typename T>
    R randint(const S& shape, T high)
    {
        return this->randint_impl<R>(shape, high);
    }

    /**
     * @copydoc prrng::GeneratorBase::randint(const S&, T)
     */
    template <class I, std::size_t L, typename T>
    auto randint(const I (&shape)[L], T high) -> typename detail::return_type_fixed<T, L>::type
    {
        using R = typename detail::return_type_fixed<T, L>::type;
        return this->randint_impl<R>(shape, high);
    }

    /**
     * @copydoc prrng::GeneratorBase::randint(const S&, T)
     * @tparam R return type, e.g. `xt::xtensor<uint32_t, 1>`
     */
    template <class R, class I, std::size_t L, typename T>
    R randint(const I (&shape)[L], T high)
    {
        return this->randint_impl<R>(shape, high);
    }

    /**
     * Generate an nd-array of random integers \f$ 0 \leq r \leq bound \f$.
     *
     * @param shape The shape of the nd-array.
     * @param low The lower bound of the random integers.
     * @param high The upper bound of the random integers.
     * @return The sample of shape `shape`.
     */
    template <class S, typename T, typename U>
    auto randint(const S& shape, T low, U high) -> typename detail::return_type<T, S>::type
    {
        using R = typename detail::return_type<T, S>::type;
        return this->randint_impl<R>(shape, low, high);
    }

    /**
     * @copydoc prrng::GeneratorBase::randint(const S&, T, U)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S, typename T, typename U>
    R randint(const S& shape, T low, U high)
    {
        return this->randint_impl<R>(shape, low, high);
    }

    /**
     * @copydoc prrng::GeneratorBase::randint(const S&, T, U)
     */
    template <class I, std::size_t L, typename T, typename U>
    auto randint(const I (&shape)[L], T low, U high) ->
        typename detail::return_type_fixed<T, L>::type
    {
        using R = typename detail::return_type_fixed<T, L>::type;
        return this->randint_impl<R>(shape, low, high);
    }

    /**
     * @copydoc prrng::GeneratorBase::randint(const S&, T, U)
     * @tparam R return type, e.g. `xt::xtensor<uint32_t, 1>`
     */
    template <class R, class I, std::size_t L, typename T, typename U>
    R randint(const I (&shape)[L], T low, U high)
    {
        return this->randint_impl<R>(shape, low, high);
    }

    /**
     * Generate an nd-array of random numbers distributed according to a normal distribution.
     * Internally, the output of random() is converted using the cumulative density
     *
     * \f$
     *      \Phi(x) = \frac{1}{2} \left[
     *          1 + \mathrm{erf}\left( \frac{x - \mu}{\sigma \sqrt{2}} \right)
     *      \right]
     * \f$
     *
     * such that the output `r` from random() leads to
     *
     * \f$ x = \mu + \sigma \sqrt{2} \mathrm{erf}^{-1} (2r - 1) \f$
     *
     * @param shape The shape of the nd-array.
     * @param mu The average.
     * @param sigma The standard deviation.
     * @return The sample of shape `shape`.
     */
    template <class S>
    auto normal(const S& shape, double mu = 0, double sigma = 1) ->
        typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->normal_impl<R>(shape, mu, sigma);
    }

    /**
     * @copydoc prrng::GeneratorBase::normal(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R normal(const S& shape, double mu = 0, double sigma = 1)
    {
        return this->normal_impl<R>(shape, mu, sigma);
    }

    /**
     * @copydoc prrng::GeneratorBase::normal(const S&, double, double)
     */
    template <class I, std::size_t L>
    auto normal(const I (&shape)[L], double mu = 0, double sigma = 1) ->
        typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->normal_impl<R>(shape, mu, sigma);
    }

    /**
     * @copydoc prrng::GeneratorBase::normal(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R normal(const I (&shape)[L], double mu = 0, double sigma = 1)
    {
        return this->normal_impl<R>(shape, mu, sigma);
    }

    /**
     * Generate an nd-array of random numbers distributed according to an exponential distribution.
     *
     * @param shape The shape of the nd-array.
     * @param scale The scale.
     * @return The sample of shape `shape`.
     */
    template <class S>
    auto exponential(const S& shape, double scale = 1) ->
        typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->exponential_impl<R>(shape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::exponential(const S&, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R exponential(const S& shape, double scale = 1)
    {
        return this->exponential_impl<R>(shape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::exponential(const S&, double)
     */
    template <class I, std::size_t L>
    auto exponential(const I (&shape)[L], double scale = 1) ->
        typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->exponential_impl<R>(shape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::exponential(const S&, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R exponential(const I (&shape)[L], double scale = 1)
    {
        return this->exponential_impl<R>(shape, scale);
    }

    /**
     * Generate an nd-array of random numbers distributed according to a Weibull distribution.
     *
     * @param shape The shape of the nd-array.
     * @param k Shape parameter.
     * @param scale Scale parameter.
     * @return The sample of shape `shape`.
     */
    template <class S>
    auto weibull(const S& shape, double k = 1, double scale = 1) ->
        typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->weibull_impl<R>(shape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::weibull(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R weibull(const S& shape, double k = 1, double scale = 1)
    {
        return this->weibull_impl<R>(shape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::weibull(const S&, double, double)
     */
    template <class I, std::size_t L>
    auto weibull(const I (&shape)[L], double k = 1, double scale = 1) ->
        typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->weibull_impl<R>(shape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::weibull(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R weibull(const I (&shape)[L], double k = 1, double scale = 1)
    {
        return this->weibull_impl<R>(shape, k, scale);
    }

    /**
     * Generate an nd-array of random numbers distributed according to a Gamma distribution.
     * Only available when compiled with PRRNG_USE_BOOST
     * (e.g. using the CMake target `prrng::use_boost`).
     *
     * @param shape The shape of the nd-array.
     * @param k Shape parameter.
     * @param scale Scale parameter.
     * @return The sample of shape `shape`.
     */
    template <class S>
    auto gamma(const S& shape, double k = 1, double scale = 1) ->
        typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->gamma_impl<R>(shape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::gamma(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R gamma(const S& shape, double k = 1, double scale = 1)
    {
        return this->gamma_impl<R>(shape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::gamma(const S&, double, double)
     */
    template <class I, std::size_t L>
    auto gamma(const I (&shape)[L], double k = 1, double scale = 1) ->
        typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->gamma_impl<R>(shape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::gamma(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R gamma(const I (&shape)[L], double k = 1, double scale = 1)
    {
        return this->gamma_impl<R>(shape, k, scale);
    }

    /**
     * Generate an nd-array of numbers that are delta distribution.
     * These numbers are not random; calling this function does not change the state of the
     * generators.
     *
     * @param shape The shape of the nd-array.
     * @param scale The value of the 'peak' of the delta distribution.
     * @return The sample of shape `shape`.
     */
    template <class S>
    auto delta(const S& shape, double scale = 1) -> typename detail::return_type<double, S>::type
    {
        using R = typename detail::return_type<double, S>::type;
        return this->delta_impl<R>(shape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::delta(const S&, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R delta(const S& shape, double scale = 1)
    {
        return this->delta_impl<R>(shape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::delta(const S&, double)
     */
    template <class I, std::size_t L>
    auto delta(const I (&shape)[L], double scale = 1) ->
        typename detail::return_type_fixed<double, L>::type
    {
        using R = typename detail::return_type_fixed<double, L>::type;
        return this->delta_impl<R>(shape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase::delta(const S&, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R delta(const I (&shape)[L], double scale = 1)
    {
        return this->delta_impl<R>(shape, scale);
    }

private:
    template <class R, class S>
    R random_impl(const S& shape)
    {
        static_assert(
            std::is_same<typename detail::allocate_return<R>::value_type, double>::value,
            "Return value_type must be double");

        detail::allocate_return<R> ret(shape);
        this->draw_list(ret.data(), ret.size());
        return ret.value;
    }

    template <class R, class S, typename T>
    R randint_impl(const S& shape, T high)
    {
        static_assert(
            std::numeric_limits<typename detail::allocate_return<R>::value_type>::max() >=
                std::numeric_limits<T>::max(),
            "Return value_type must must be able to accommodate the bound");

        PRRNG_ASSERT(high >= 0);
        PRRNG_ASSERT(static_cast<uint32_t>(high) < std::numeric_limits<uint32_t>::max());

        detail::allocate_return<R> ret(shape);
        std::vector<uint32_t> tmp(ret.size());
        this->draw_list_uint32(&tmp.front(), static_cast<uint32_t>(high), ret.size());
        std::copy(tmp.begin(), tmp.end(), ret.data());
        return ret.value;
    }

    template <class R, class S, typename T, typename U>
    R randint_impl(const S& shape, T low, U high)
    {
        static_assert(
            std::numeric_limits<typename detail::allocate_return<R>::value_type>::min() >=
                std::numeric_limits<T>::min(),
            "Return value_type must must be able to accommodate the bound");

        static_assert(
            std::numeric_limits<typename detail::allocate_return<R>::value_type>::max() >=
                std::numeric_limits<T>::max(),
            "Return value_type must must be able to accommodate the bound");

        static_assert(
            std::numeric_limits<typename detail::allocate_return<R>::value_type>::min() >=
                std::numeric_limits<U>::min(),
            "Return value_type must must be able to accommodate the bound");

        static_assert(
            std::numeric_limits<typename detail::allocate_return<R>::value_type>::max() >=
                std::numeric_limits<U>::max(),
            "Return value_type must must be able to accommodate the bound");

        PRRNG_ASSERT(high - low >= 0);
        PRRNG_ASSERT(static_cast<uint32_t>(high - low) < std::numeric_limits<uint32_t>::max());

        detail::allocate_return<R> ret(shape);
        std::vector<uint32_t> tmp(ret.size());
        this->draw_list_uint32(&tmp.front(), static_cast<uint32_t>(high - low), ret.size());
        std::copy(tmp.begin(), tmp.end(), ret.data());
        return ret.value + low;
    }

    template <class R, class S>
    R normal_impl(const S& shape, double mu, double sigma)
    {
        R r = this->random_impl<R>(shape);
        return normal_distribution(mu, sigma).quantile(r);
    }

    template <class R, class S>
    R exponential_impl(const S& shape, double scale)
    {
        R r = this->random_impl<R>(shape);
        return exponential_distribution(scale).quantile(r);
    }

    template <class R, class S>
    R weibull_impl(const S& shape, double k, double scale)
    {
        R r = this->random_impl<R>(shape);
        return weibull_distribution(k, scale).quantile(r);
    }

    template <class R, class S>
    R gamma_impl(const S& shape, double k, double scale)
    {
        R r = this->random_impl<R>(shape);
        return gamma_distribution(k, scale).quantile(r);
    }

    template <class R, class S>
    R delta_impl(const S& shape, double scale)
    {
        R ret = xt::empty<typename R::value_type>(shape);
        ret.fill(scale);
        return ret;
    }

protected:
    /**
     * @brief Draw one random double.
     * @return double
     */
    virtual double draw_double()
    {
        return 0.5;
    }

    /**
     * Draw `n` random numbers and write them to list (input as pointer `data`).
     *
     * @param data Pointer to the data (no bounds-check).
     * @param n Size of `data`.
     */
    virtual void draw_list(double* data, size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            data[i] = 0.5;
        }
    }

    /**
     * Draw `n` random numbers and write them to list (input as pointer `data`).
     *
     * @param data Pointer to the data (no bounds-check).
     * @param bound Upper bound of the random numbers.
     * @param n Size of `data`.
     */
    virtual void draw_list_uint32(uint32_t* data, uint32_t bound, size_t n)
    {
        (void)(bound);

        for (size_t i = 0; i < n; ++i) {
            data[i] = 0;
        }
    }
};

/**
 * Random number generate using the pcg32 algorithm.
 * The class generate random 32-bit random numbers (of type `uint32_t`).
 * In addition, they can be converted to nd-arrays of random floating-point numbers (according)
 * using derived methods from Generate().
 *
 * The algorithm is full based on:
 *
 *     The PCG random number generator was developed by Melissa O'Neill
 *     <oneill@pcg-random.org>
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 *     For additional information about the PCG random number generation scheme,
 *     including its license and other licensing options, visit
 *
 *         http://www.pcg-random.org
 *
 * Whereby most code is taken from the follow wrapper:
 *
 *     Wenzel Jakob, February 2015
 *     https://github.com/wjakob/pcg32
 */
class pcg32 : public GeneratorBase {
public:
    /**
     * Constructor.
     *
     * @param initstate State initiator.
     * @param initseq Sequence initiator.
     */
    template <typename T = uint64_t, typename S = uint64_t>
    pcg32(T initstate = PRRNG_PCG32_INITSTATE, S initseq = PRRNG_PCG32_INITSEQ)
    {
        static_assert(sizeof(uint64_t) >= sizeof(T), "Down-casting not allowed.");
        static_assert(sizeof(uint64_t) >= sizeof(S), "Down-casting not allowed.");
        this->seed(static_cast<uint64_t>(initstate), static_cast<uint64_t>(initseq));
    }

    /**
     * Draw new random number (uniformly distributed, `0 <= r <= max(uint32_t)`).
     * This advances the state of the generator by one increment.
     *
     * @return Next random number in sequence.
     *
     * @author Melissa O'Neill, http://www.pcg-random.org.
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
     * Draw new random number (uniformly distributed, `0 <= r <= max(uint32_t)`).
     * This advances the state of the generator by one increment.
     *
     * @return Next random number in sequence.
     *
     * \note Wrapper around operator().
     */
    uint32_t next_uint32()
    {
        return (*this)();
    }

    /**
     * Draw new random number (uniformly distributed, `0 <= r <= bound`).
     *
     * @param bound Bound on the return.
     * @return Next random number in sequence.
     *
     * @author Wenzel Jakob, https://github.com/wjakob/pcg32.
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
     * Generate a single precision floating point value on the interval [0, 1).
     *
     * @return Next random number in sequence.
     *
     * @author Wenzel Jakob, https://github.com/wjakob/pcg32.
     */
    float next_float()
    {
        // Trick from MTGP: generate an uniformly distributed
        // single precision number in [1,2) and subtract 1.
        union {
            uint32_t u;
            float f;
        } x;
        x.u = (next_uint32() >> 9) | 0x3f800000u;
        return x.f - 1.0f;
    }

    /**
     * Generate a double precision floating point value on the interval [0, 1).
     *
     * @return Next random number in sequence.
     *
     * @remark Since the underlying random number generator produces 32 bit output,
     * only the first 32 mantissa bits will be filled (however, the resolution is still
     * finer than in next_float(), which only uses 23 mantissa bits).
     *
     * @author Wenzel Jakob, https://github.com/wjakob/pcg32.
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
     * The current "state" of the generator. If the same initseq() is used, this exact point
     * in the sequence can be restored with restore().
     *
     * @return State of the generator.
     */
    uint64_t state() const
    {
        return m_state;
    }

    /**
     * @copydoc prrng::pcg32::state() const
     *
     * @tparam R use a different return-type. There are some internal checks if the type is able to
     * store the internal state of type `uint64_t`.
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
     * The state initiator that was used upon construction.
     *
     * @return initiator.
     */
    uint64_t initstate() const
    {
        return m_initstate;
    }

    /**
     * @copydoc initstate()
     *
     * @tparam R use a different return-type. There are some internal checks if the type is able to
     * store the internal state of type `uint64_t`.
     */
    template <typename R>
    R initstate() const
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
     * The sequence initiator that was used upon construction.
     *
     * @return initiator.
     */
    uint64_t initseq() const
    {
        return m_initseq;
    }

    /**
     * @copydoc initseq()
     *
     * @tparam R use a different return-type. There are some internal checks if the type is able to
     * store the internal state of type `uint64_t`.
     */
    template <typename R>
    R initseq() const
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
     * Restore a given state in the sequence. See state().
     *
     * @tparam R use a different return-type. There are some internal checks if the type is able to
     * store the internal state of type `uint64_t`.
     */
    template <typename T>
    void restore(T state)
    {
        static_assert(sizeof(uint64_t) >= sizeof(T), "Down-casting not allowed.");
        m_state = static_cast<uint64_t>(state);
    }

    /**
     * @copydoc prrng::pcg32::distance(const pcg32&) const
     */
    int64_t operator-(const pcg32& other) const
    {
        PRRNG_DEBUG(m_inc == other.m_inc);

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
            PRRNG_DEBUG((m_state & the_bit) == (cur_state & the_bit));
            the_bit <<= 1;
            cur_plus = (cur_mult + 1ULL) * cur_plus;
            cur_mult *= cur_mult;
        }

        return (int64_t)distance;
    }

    /**
     * The distance between two PCG32 pseudorandom number generators.
     *
     * @tparam R
     *     Return-type.
     *     `static_assert` against down-casting, #PRRNG_DEBUG against loss of signedness.
     *
     * @return Distance.
     *
     * @author Wenzel Jakob, https://github.com/wjakob/pcg32.
     */
    template <typename R = int64_t>
    R distance(const pcg32& other) const
    {
        static_assert(sizeof(R) >= sizeof(int64_t), "Down-casting not allowed.");
        int64_t r = this->operator-(other);

#ifdef PRRNG_ENABLE_DEBUG
        bool u = std::is_unsigned<R>::value;
        PRRNG_DEBUG((r < 0 && !u) || r >= 0);
#endif

        return static_cast<R>(r);
    }

    /**
     * The distance between two states.
     *
     * @tparam R
     *     Return-type.
     *     `static_assert` against down-casting, #PRRNG_DEBUG against loss of signedness.
     *
     * @return Distance.
     *
     * @warning The increment of used to generate must be the same. There is no way of checking
     * here!
     *
     * @author Wenzel Jakob, https://github.com/wjakob/pcg32.
     */
    template <
        typename R = int64_t,
        typename T,
        std::enable_if_t<std::is_integral<T>::value, bool> = true>
    R distance(T other_state) const
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
            PRRNG_DEBUG((m_state & the_bit) == (cur_state & the_bit));
            the_bit <<= 1;
            cur_plus = (cur_mult + 1ULL) * cur_plus;
            cur_mult *= cur_mult;
        }

        int64_t r = (int64_t)distance;

#ifdef PRRNG_ENABLE_DEBUG
        bool u = std::is_unsigned<R>::value;
        PRRNG_DEBUG((r < 0 && !u) || r >= 0);
#endif

        return static_cast<R>(r);
    }

    /**
     * Multi-step advance function (jump-ahead, jump-back).
     *
     * @param distance Distance to jump ahead or jump back (depending on the sign).
     * This changes that state of the generator by the appropriate number of increments.
     *
     * \note The method used here is based on Brown, "Random Number Generation
     * with Arbitrary Stride", Transactions of the American Nuclear Society (Nov. 1994).
     * The algorithm is very similar to fast exponentiation.
     *
     * @author Wenzel Jakob, https://github.com/wjakob/pcg32.
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

        // Even though delta is an unsigned integer, we can pass a signed
        // integer to go backwards, it just goes "the long way round".
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
     * Draw uniformly distributed permutation and permute the given STL container.
     *
     * @param begin
     * @param end
     *
     * \note From: Knuth, TAoCP Vol. 2 (3rd 3d), Section 3.4.2
     *
     * @author Wenzel Jakob, https://github.com/wjakob/pcg32.
     */
    template <typename Iterator>
    void shuffle(Iterator begin, Iterator end)
    {
        for (Iterator it = end - 1; it > begin; --it) {
            std::iter_swap(it, begin + next_uint32((uint32_t)(it - begin + 1)));
        }
    }

    /**
     * Equality operator.
     *
     * @param other The generator to which to compare.
     *
     * @author Wenzel Jakob, https://github.com/wjakob/pcg32.
     */
    bool operator==(const pcg32& other) const
    {
        return m_state == other.m_state && m_inc == other.m_inc;
    }

    /**
     * Inequality operator.
     *
     * @param other The generator to which to compare.
     *
     * @author Wenzel Jakob, https://github.com/wjakob/pcg32.
     */
    bool operator!=(const pcg32& other) const
    {
        return m_state != other.m_state || m_inc != other.m_inc;
    }

protected:
    double draw_double() override
    {
        return next_double();
    }

    void draw_list(double* data, size_t n) override
    {
        for (size_t i = 0; i < n; ++i) {
            data[i] = next_double();
        }
    }

    void draw_list_uint32(uint32_t* data, uint32_t bound, size_t n) override
    {
        for (size_t i = 0; i < n; ++i) {
            data[i] = next_uint32(bound);
        }
    }

private:
    void seed(uint64_t initstate, uint64_t initseq)
    {
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
 * @brief Generator of a random cumulative sum of which a chunk is kept in memory.
 *
 *      -   The chunk is stored externally.
 *      -   The random number generated (pcg32) is stored externally.
 *      -   See prrng::pcg32_cumsum for wrapper with storage.
 */
class pcg32_cumsum_external {
private:
    pcg32* m_gen; ///< Random number generator
    ptrdiff_t m_gen_index; ///< Index of the generator
    ptrdiff_t m_start; ///< Index of the start of the chunk
    double m_first; ///< Value of the first entry of the chunk: optional, used for restoring
    bool m_apply_first; ///< Signal to modify the first value of the new chunk, see restore()
    double* m_chunk; ///< Pointer to the storage of the chunk
    size_t m_size; ///< Size of the chunk
    bool m_delta; ///< Signal if distribution is a delta, and no random numbers are generated.

public:
    pcg32_cumsum_external() = default;

    /**
     * @copydoc prrng::pcg32_cumsum_external::init.
     */
    pcg32_cumsum_external(double* data, size_t n, pcg32* generator, ptrdiff_t generator_index)
    {
        this->init(data, n, generator, generator_index);
    }

protected:
    /**
     * @brief Constructor.
     *
     * @param data Pointer to chunk storage.
     * @param n Size of the chunk.
     * @param generator Pointer to the generator.
     * @param generator_index Current index in the chunk that the generator corresponds to.
     */
    void init(double* data, size_t n, pcg32* generator, ptrdiff_t generator_index)
    {
        m_delta = false;
        m_gen = generator;
        m_gen_index = generator_index;
        m_start = generator_index;
        m_apply_first = false;
        m_chunk = data;
        m_size = n;
    }

private:
    /**
     * @brief Jump the random number generator somewhere.
     * @param delta Distance to jump.
     */
    void jump(ptrdiff_t delta)
    {
        if (m_delta) {
            return;
        }
        m_gen->advance(delta);
        m_gen_index += delta;
    }

    /**
     * @brief Update generator index.
     * @param n Number of drawn numbers.
     */
    void drawn(ptrdiff_t n)
    {
        if (m_delta) {
            return;
        }
        m_gen_index += n;
    }

public:
    /**
     * @brief Overwrite the data.
     *
     * @param data Pointer to the data.
     * @param n Size of the data.
     */
    void set_data(double* data, size_t n)
    {
        m_chunk = data;
        m_size = n;
    }

    /**
     * @brief Index at which the generator is currently.
     * @return ptrdiff_t
     */
    ptrdiff_t generator_index() const
    {
        return m_gen_index;
    }

    /**
     * @brief Set index at which the generator is currently.
     * @param index Index.
     */
    void set_generator_index(ptrdiff_t index)
    {
        m_gen_index = index;
    }

    /**
     * @brief Index of the first entry of the chunk.
     * @return ptrdiff_t
     */
    ptrdiff_t start() const
    {
        return m_start;
    }

    /**
     * @brief Set index of the first entry of the chunk.
     * @param index Index.
     */
    void set_start(ptrdiff_t index)
    {
        m_start = index;
    }

    /**
     * @brief Get the state of the random number generator at some index.
     * @param index The index in the random sequence.
     * @return uint64_t
     */
    uint64_t state(ptrdiff_t index)
    {
        if (m_delta) {
            return m_gen->state();
        }
        uint64_t state = m_gen->state();
        m_gen->advance(index - m_gen_index);
        uint64_t ret = m_gen->state();
        m_gen->restore(state);
        return ret;
    }

    /**
     * @brief Update the state of the generator.
     *
     * @param state The state.
     * @param index The index that `state` corresponds to.
     */
    void set_state(uint64_t state, ptrdiff_t index)
    {
        m_gen->restore(state);
        m_gen_index = index;
    }

    /**
     * @brief Restore a specific state in the cumulative sum.
     * This function should be followed by draw_chunk() or on of its wrappers
     * (e.g. draw_chunk_weibull()).
     *
     * @param state The state at the beginning of the new chunk.
     * @param value The value of the first entry of the new chunk.
     * @param index The index of the first entry of the new chunk.
     */
    void restore(uint64_t state, double value, ptrdiff_t index)
    {
        m_gen->restore(state);
        m_gen_index = index;
        m_start = index;
        m_first = value;
        m_apply_first = true;
    }

    /**
     * @brief Draw a (the first) chunk of the cumulative sum.
     *
     * @param get_chunk Function to draw the random numbers, called as `get_chunk(n)`.
     */
    template <class F>
    void draw_chunk(const F& get_chunk)
    {
        using R = decltype(get_chunk(size_t{}));

        R extra = get_chunk(m_size);
        this->drawn(static_cast<ptrdiff_t>(m_size));

        if (m_apply_first) {
            extra.front() += m_first - extra.front();
            m_apply_first = false;
        }

        std::partial_sum(extra.begin(), extra.end(), m_chunk);
    }

    /**
     * Shift chunk left.
     *
     * @param get_chunk Function to draw the random numbers, called as `get_chunk(n)`.
     * @param margin Overlap to keep with the current chunk.
     */
    template <class F>
    void prev_chunk(const F& get_chunk, size_t margin = 0)
    {
        using R = decltype(get_chunk(size_t{}));

        ptrdiff_t n = static_cast<ptrdiff_t>(m_size);
        this->jump(m_start - n + margin - m_gen_index);

        double front = m_chunk[0];
        size_t m = m_size - margin + 1;
        R extra = get_chunk({m});
        this->drawn(m);
        std::partial_sum(extra.begin(), extra.end(), extra.begin());
        extra -= extra.back() - front;

        std::copy(m_chunk, m_chunk + margin, m_chunk + m_size - margin);
        std::copy(extra.begin(), extra.end() - 1, m_chunk);

        m_start -= static_cast<ptrdiff_t>(m_size - margin);
    }

    /**
     * Shift chunk right.
     *
     * @param get_chunk Function to draw the random numbers, called as `get_chunk(n)`.
     * @param margin Overlap to keep with the current chunk.
     */
    template <class F>
    void next_chunk(const F& get_chunk, size_t margin = 0)
    {
        using R = decltype(get_chunk(size_t{}));
        PRRNG_ASSERT(margin < m_size);

        this->jump(m_start + m_size - m_gen_index);

        double back = m_chunk[m_size - 1];
        size_t n = m_size - margin;
        R extra = get_chunk({n});
        this->drawn(n);
        extra.front() += back;
        std::partial_sum(extra.begin(), extra.end(), extra.begin());
        std::copy(m_chunk + m_size - margin, m_chunk + m_size, m_chunk);
        std::copy(extra.begin(), extra.end(), m_chunk + margin);
        m_start += n;
    }

    /**
     * Align the chunk to encompass a target value.
     *
     * @param get_chunk Function to draw the random numbers, called as `get_chunk(n)`.
     * @param get_cumsum Function to get the cumsum of random numbers, called: `get_cumsum(n)`.
     * @param target Target value.
     * @param margin Buffer to keep left of the target.
     *
     * @param strict
     *      If `true` the margin is respected strictly. Otherwise the real margin can be larger
     *      or equal to the specified margin if significant efficiency can be gained.
     */
    template <class F, class G>
    void align_chunk(
        const F& get_chunk,
        const G& get_cumsum,
        double target,
        size_t margin = 0,
        bool strict = false)
    {
        using R = decltype(get_chunk(size_t{}));
        PRRNG_ASSERT(margin < m_size);

        double delta = m_chunk[m_size - 1] - m_chunk[0];
        size_t n = m_size;

        if (target > m_chunk[m_size - 1]) {

            this->jump(m_start + m_size - m_gen_index);
            double back = m_chunk[m_size - 1];

            double j = (target - m_chunk[m_size - 1]) / delta - (double)(margin) / (double)(n);

            if (j > 1) {
                size_t m = static_cast<size_t>((j - 1) * static_cast<double>(n));
                back += get_cumsum(m);
                this->drawn(m);
                m_start += m + m_size;
                R extra = get_chunk({n});
                this->drawn(n);
                extra.front() += back;
                std::partial_sum(extra.begin(), extra.end(), m_chunk);
                return this->align_chunk(get_chunk, get_cumsum, target, margin, strict);
            }

            this->next_chunk(get_chunk, 1 + margin);
            return this->align_chunk(get_chunk, get_cumsum, target, margin, strict);
        }
        else if (target < m_chunk[0]) {
            this->prev_chunk(get_chunk);
            return this->align_chunk(get_chunk, get_cumsum, target, margin, strict);
        }
        else {
            size_t i = std::lower_bound(m_chunk, m_chunk + m_size, target) - m_chunk - 1;
            if (i + 1 < margin) {
                if (!strict) {
                    return;
                }
                this->prev_chunk(get_chunk);
                return this->align_chunk(get_chunk, get_cumsum, target, margin, strict);
            }

            size_t n = i + 1 - margin;
            if (n == 0) {
                return;
            }

            this->jump(m_start + m_size - m_gen_index);

            R extra = get_chunk({n});
            this->drawn(n);
            m_start += n;
            extra.front() += m_chunk[m_size - 1];
            std::partial_sum(extra.begin(), extra.end(), extra.begin());
            std::copy(m_chunk + n, m_chunk + m_size, m_chunk);
            std::copy(extra.begin(), extra.end(), m_chunk + m_size - n);
        }
    }

    /**
     * @brief Draw new chunk.
     *
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     */
    void draw_chunk_weibull(double k = 1, double scale = 1, double offset = 0)
    {
        this->draw_chunk([this, k, scale, offset](size_t n) -> xt::xtensor<double, 1> {
            return m_gen->weibull<xt::xtensor<double, 1>>({n}, k, scale) + offset;
        });
    }

    /**
     * @brief Shift left.
     *
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Overlap to keep right.
     */
    void prev_chunk_weibull(double k = 1, double scale = 1, double offset = 0, size_t margin = 0)
    {
        this->prev_chunk(
            [this, k, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->weibull<xt::xtensor<double, 1>>({n}, k, scale) + offset;
            },
            margin);
    }

    /**
     * @brief Shift right.
     *
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Overlap to keep left.
     */
    void next_chunk_weibull(double k = 1, double scale = 1, double offset = 0, size_t margin = 0)
    {
        this->next_chunk(
            [this, k, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->weibull<xt::xtensor<double, 1>>({n}, k, scale) + offset;
            },
            margin);
    }

    /**
     * @brief Align chunk with target value.
     *
     * @param target Target value.
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    void align_chunk_weibull(
        double target,
        double k = 1,
        double scale = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        this->align_chunk(
            [this, k, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->weibull<xt::xtensor<double, 1>>({n}, k, scale) + offset;
            },
            [this, k, scale, offset](size_t n) {
                return m_gen->cumsum_weibull(n, k, scale) + static_cast<double>(n) * offset;
            },
            target,
            margin,
            strict);
    }

    /**
     * @brief Draw new chunk.
     *
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     */
    void draw_chunk_gamma(double k = 1, double scale = 1, double offset = 0)
    {
        this->draw_chunk([this, k, scale, offset](size_t n) -> xt::xtensor<double, 1> {
            return m_gen->gamma<xt::xtensor<double, 1>>({n}, k, scale) + offset;
        });
    }

    /**
     * @brief Shift left.
     *
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Overlap to keep right.
     */
    void prev_chunk_gamma(double k = 1, double scale = 1, double offset = 0, size_t margin = 0)
    {
        this->prev_chunk(
            [this, k, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->gamma<xt::xtensor<double, 1>>({n}, k, scale) + offset;
            },
            margin);
    }

    /**
     * @brief Shift right.
     *
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Overlap to keep left.
     */
    void next_chunk_gamma(double k = 1, double scale = 1, double offset = 0, size_t margin = 0)
    {
        this->next_chunk(
            [this, k, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->gamma<xt::xtensor<double, 1>>({n}, k, scale) + offset;
            },
            margin);
    }

    /**
     * @brief Align chunk with target value.
     *
     * @param target Target value.
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    void align_chunk_gamma(
        double target,
        double k = 1,
        double scale = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        this->align_chunk(
            [this, k, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->gamma<xt::xtensor<double, 1>>({n}, k, scale) + offset;
            },
            [this, k, scale, offset](size_t n) {
                return m_gen->cumsum_gamma(n, k, scale) + static_cast<double>(n) * offset;
            },
            target,
            margin,
            strict);
    }

    /**
     * @brief Draw new chunk.
     *
     * @param mu Mean.
     * @param sigma Standard deviation.
     * @param offset Fixed offset.
     */
    void draw_chunk_normal(double mu = 0, double sigma = 1, double offset = 0)
    {
        this->draw_chunk([this, mu, sigma, offset](size_t n) -> xt::xtensor<double, 1> {
            return m_gen->normal<xt::xtensor<double, 1>>({n}, mu, sigma) + offset;
        });
    }

    /**
     * @brief Shift left.
     *
     * @param mu Mean.
     * @param sigma Standard deviation.
     * @param offset Fixed offset.
     * @param margin Overlap to keep right.
     */
    void prev_chunk_normal(double mu = 0, double sigma = 1, double offset = 0, size_t margin = 0)
    {
        this->prev_chunk(
            [this, mu, sigma, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->normal<xt::xtensor<double, 1>>({n}, mu, sigma) + offset;
            },
            margin);
    }

    /**
     * @brief Shift right.
     *
     * @param mu Mean.
     * @param sigma Standard deviation.
     * @param offset Fixed offset.
     * @param margin Overlap to keep left.
     */
    void next_chunk_normal(double mu = 0, double sigma = 1, double offset = 0, size_t margin = 0)
    {
        this->next_chunk(
            [this, mu, sigma, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->normal<xt::xtensor<double, 1>>({n}, mu, sigma) + offset;
            },
            margin);
    }

    /**
     * @brief Align chunk with target value.
     *
     * @param target Target value.
     * @param mu Mean.
     * @param sigma Standard deviation.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    void align_chunk_normal(
        double target,
        double mu = 0,
        double sigma = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        this->align_chunk(
            [this, mu, sigma, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->normal<xt::xtensor<double, 1>>({n}, mu, sigma) + offset;
            },
            [this, mu, sigma, offset](size_t n) {
                return m_gen->cumsum_normal(n, mu, sigma) + static_cast<double>(n) * offset;
            },
            target,
            margin,
            strict);
    }

    /**
     * @brief Draw new chunk.
     *
     * @param scale Scale factor.
     * @param offset Fixed offset.
     */
    void draw_chunk_exponential(double scale = 1, double offset = 0)
    {
        this->draw_chunk([this, scale, offset](size_t n) -> xt::xtensor<double, 1> {
            return m_gen->exponential<xt::xtensor<double, 1>>({n}, scale) + offset;
        });
    }

    /**
     * @brief Shift left.
     *
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Overlap to keep right.
     */
    void prev_chunk_exponential(double scale = 1, double offset = 0, size_t margin = 0)
    {
        this->prev_chunk(
            [this, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->exponential<xt::xtensor<double, 1>>({n}, scale) + offset;
            },
            margin);
    }

    /**
     * @brief Shift right.
     *
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Overlap to keep left.
     */
    void next_chunk_exponential(double scale = 1, double offset = 0, size_t margin = 0)
    {
        this->next_chunk(
            [this, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->exponential<xt::xtensor<double, 1>>({n}, scale) + offset;
            },
            margin);
    }

    /**
     * @brief Align chunk with target value.
     *
     * @param target Target value.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    void align_chunk_exponential(
        double target,
        double scale = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        this->align_chunk(
            [this, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->exponential<xt::xtensor<double, 1>>({n}, scale) + offset;
            },
            [this, scale, offset](size_t n) {
                return m_gen->cumsum_exponential(n, scale) + static_cast<double>(n) * offset;
            },
            target,
            margin,
            strict);
    }

    /**
     * @brief Draw new chunk.
     *
     * @param scale Scale factor.
     * @param offset Fixed offset.
     */
    void draw_chunk_delta(double scale = 1, double offset = 0)
    {
        m_delta = true;
        this->draw_chunk([this, scale, offset](size_t n) -> xt::xtensor<double, 1> {
            return m_gen->delta<xt::xtensor<double, 1>>({n}, scale) + offset;
        });
        m_delta = false;
    }

    /**
     * @brief Shift left.
     *
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Overlap to keep right.
     */
    void prev_chunk_delta(double scale = 1, double offset = 0, size_t margin = 0)
    {
        m_delta = true;
        this->prev_chunk(
            [this, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->delta<xt::xtensor<double, 1>>({n}, scale) + offset;
            },
            margin);
        m_delta = false;
    }

    /**
     * @brief Shift right.
     *
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Overlap to keep left.
     */
    void next_chunk_delta(double scale = 1, double offset = 0, size_t margin = 0)
    {
        m_delta = true;
        this->next_chunk(
            [this, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->delta<xt::xtensor<double, 1>>({n}, scale) + offset;
            },
            margin);
        m_delta = false;
    }

    /**
     * @brief Align chunk with target value.
     *
     * @param target Target value.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    void align_chunk_delta(
        double target,
        double scale = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        m_delta = true;
        this->align_chunk(
            [this, scale, offset](size_t n) -> xt::xtensor<double, 1> {
                return m_gen->exponential<xt::xtensor<double, 1>>({n}, scale) + offset;
            },
            [this, scale, offset](size_t n) {
                return m_gen->cumsum_exponential(n, scale) + static_cast<double>(n) * offset;
            },
            target,
            margin,
            strict);
        m_delta = false;
    }
};

/**
 * @brief Generator of a random cumulative sum of which a chunk is kept in memory.
 * The random number generated by the pcg32 algorithm.
 *
 * @tparam Storage of the data.
 */
template <class R>
class pcg32_cumsum : public pcg32_cumsum_external {
private:
    R m_data;
    pcg32 m_gen;
    using pcg32_cumsum_external::set_data;

public:
    /**
     * @brief Construct a new pcg32 cumsum object
     *
     * @param shape Shape of the chunk.
     * @param initstate State initiator.
     * @param initseq Sequence initiator.
     */
    template <class D, typename T = uint64_t, typename S = uint64_t>
    pcg32_cumsum(
        const D& shape,
        T initstate = PRRNG_PCG32_INITSTATE,
        S initseq = PRRNG_PCG32_INITSEQ)
    {
        m_data = xt::empty<typename R::value_type>(shape);
        m_gen = pcg32(initstate, initseq);
        this->init(&m_data.flat(0), m_data.size(), &m_gen, 0);
    }

    /**
     * @brief Access the generator.
     * @return const pcg32&
     */
    const pcg32& generator() const
    {
        return m_gen;
    }

    /**
     * @brief Shape of the chunk.
     * @return const auto&
     */
    const auto& shape() const
    {
        return m_data.shape();
    }

    /**
     * @brief Size of the chunk.
     * @return auto
     */
    auto size() const
    {
        return m_data.size();
    }

    /**
     * @brief Pointer to the chunk.
     * @return const R&
     */
    const R& chunk() const
    {
        return m_data;
    }

    /**
     * @brief Overwrite the chunk.
     * This can be used to make modification externally.
     * Please check if set_state() or set_start() should be called too.
     *
     * @param data The chunk.
     */
    void set_chunk(const R& data)
    {
        m_data = data;
        this->set_data(&m_data.flat(0), m_data.size());
    }
};

/**
 * Base class of an array of pseudorandom number generators.
 * This class provides common methods, but itself does not really do much.
 * See the description of derived classed for information.
 *
 * @tparam M Type to use storage of the shape and array vectors. E.g. `std::vector` or `std::array`
 */
template <class M>
class GeneratorBase_array {
public:
    GeneratorBase_array() = default;

    virtual ~GeneratorBase_array() = default;

    /**
     * Return the size of the array of generators.
     *
     * @return unsigned int
     */
    size_t size() const
    {
        return m_size;
    }

    /**
     * Return the shape of the array of generators.
     *
     * @return vector of unsigned ints
     */
    M shape() const
    {
        return m_shape;
    }

    /**
     * Return the shape of the array of generators along a specific axis.
     *
     * @param axis The axis.
     * @return vector of unsigned ints
     */
    template <class T>
    size_t shape(T axis) const
    {
        return m_shape[axis];
    }

    /**
     * Return a flat index based on an array index specified as a list.
     *
     * @param index Array index, e.g. as std::vector.
     * @return Flat index.
     */
    template <class T>
    size_t flat_index(const T& index) const
    {
        PRRNG_DEBUG(this->inbounds(index));
        return std::inner_product(index.cbegin(), index.cend(), m_strides.cbegin(), 0);
    }

    /**
     * Check if an index is in bounds (and of the correct rank).
     *
     * @return `false` if out-of-bounds, `true` otherwise.
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
     * Per generator, generate an nd-array of random numbers \f$ 0 \leq r \leq 1 \f$.
     *
     * @param ishape The shape of the nd-array drawn per generator.
     * @return The array of arrays of samples: [#shape, `ishape`]
     */
    template <class S>
    auto random(const S& ishape) -> typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->random_impl<R>(ishape);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::random(const S&)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R random(const S& ishape)
    {
        return this->random_impl<R>(ishape);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::random(const S&)
     */
    template <class I, std::size_t L>
    auto random(const I (&ishape)[L]) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->random_impl<R>(detail::to_array(ishape));
    }

    /**
     * @copydoc prrng::GeneratorBase_array::random(const S&)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R random(const I (&ishape)[L])
    {
        return this->random_impl<R>(detail::to_array(ishape));
    }

    /**
     * Per generator, generate an nd-array of random integers \f$ 0 \leq r \leq bound \f$.
     *
     * @param ishape The shape of the nd-array drawn per generator.
     * @param high The upper bound of the interval.
     * @return The array of arrays of samples: [#shape, `ishape`]
     */
    template <class S, typename T>
    auto randint(const S& ishape, T high) -> typename detail::composite_return_type<T, M, S>::type
    {
        using R = typename detail::composite_return_type<T, M, S>::type;
        return this->randint_impl<R>(ishape, high);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::randint(const S&, T)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S, typename T>
    R randint(const S& ishape, T high)
    {
        return this->randint_impl<R>(ishape, high);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::randint(const S&, T)
     */
    template <class I, std::size_t L, typename T>
    auto randint(const I (&ishape)[L], T high) ->
        typename detail::composite_return_type<T, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<T, M, std::array<size_t, L>>::type;
        return this->randint_impl<R>(detail::to_array(ishape), high);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::randint(const S&, T)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L, typename T>
    R randint(const I (&ishape)[L], T high)
    {
        return this->randint_impl<R>(detail::to_array(ishape), high);
    }

    /**
     * Per generator, generate an nd-array of random integers \f$ 0 \leq r \leq bound \f$.
     *
     * @param ishape The shape of the nd-array drawn per generator.
     * @param low The lower bound of the interval.
     * @param high The upper bound of the interval.
     * @return The array of arrays of samples: [#shape, `ishape`]
     */
    template <class S, typename T, typename U>
    auto randint(const S& ishape, T low, U high) ->
        typename detail::composite_return_type<T, M, S>::type
    {
        using R = typename detail::composite_return_type<T, M, S>::type;
        return this->randint_impl<R>(ishape, low, high);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::randint(const S&, T, U)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S, typename T, typename U>
    R randint(const S& ishape, T low, U high)
    {
        return this->randint_impl<R>(ishape, low, high);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::randint(const S&, T, U)
     */
    template <class I, std::size_t L, typename T, typename U>
    auto randint(const I (&ishape)[L], T low, U high) ->
        typename detail::composite_return_type<T, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<T, M, std::array<size_t, L>>::type;
        return this->randint_impl<R>(detail::to_array(ishape), low, high);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::randint(const S&, T, U)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L, typename T, typename U>
    R randint(const I (&ishape)[L], T low, U high)
    {
        return this->randint_impl<R>(detail::to_array(ishape), low, high);
    }

    /**
     * Per generator, generate an nd-array of random numbers distributed
     * according to a normal distribution.
     * Internally, the output of random() is converted using the cumulative density
     *
     * \f$ \Phi(x) = \frac{1}{2} \left[
     *     1 + \mathrm{erf}\left( \frac{x - \mu}{\sigma \sqrt{2}} \right)
     * \right]\f$
     *
     * such that the output `r` from random() leads to
     *
     * \f$ x = \mu + \sigma \sqrt{2} \mathrm{erf}^{-1} (2r - 1) \f$
     *
     * @param ishape The shape of the nd-array drawn per generator.
     * @param mu The average.
     * @param sigma The standard deviation.
     * @return The array of arrays of samples: [#shape, `ishape`]
     */
    template <class S>
    auto normal(const S& ishape, double mu = 0, double sigma = 1) ->
        typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->normal_impl<R>(ishape, mu, sigma);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::normal(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R normal(const S& ishape, double mu = 0, double sigma = 1)
    {
        return this->normal_impl<R>(ishape, mu, sigma);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::normal(const S&, double, double)
     */
    template <class I, std::size_t L>
    auto normal(const I (&ishape)[L], double mu = 0, double sigma = 1) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->normal_impl<R>(detail::to_array(ishape), mu, sigma);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::normal(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R normal(const I (&ishape)[L], double mu = 0, double sigma = 1)
    {
        return this->normal_impl<R>(detail::to_array(ishape), mu, sigma);
    }

    /**
     * Per generator, generate an nd-array of random numbers distributed
     * according to an exponential distribution.
     *
     * @param ishape The shape of the nd-array drawn per generator.
     * @param scale The scale.
     * @return The array of arrays of samples: [#shape, `ishape`]
     */
    template <class S>
    auto exponential(const S& ishape, double scale = 1) ->
        typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->exponential_impl<R>(ishape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::exponential(const S&, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R exponential(const S& ishape, double scale = 1)
    {
        return this->exponential_impl<R>(ishape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::exponential(const S&, double)
     */
    template <class I, std::size_t L>
    auto exponential(const I (&ishape)[L], double scale = 1) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->exponential_impl<R>(detail::to_array(ishape), scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::exponential(const S&, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R exponential(const I (&ishape)[L], double scale = 1)
    {
        return this->exponential_impl<R>(detail::to_array(ishape), scale);
    }

    /**
     * Per generator, generate an nd-array of random numbers distributed
     * according to a Weibull distribution.
     * Internally, the output of random() is converted using the cumulative density
     *
     * \f$ \Phi(x) = 1 - e^{-(x / \lambda)^k} \f$
     *
     * such that the output `r` from random() leads to
     *
     * \f$ x = \lambda (- \ln (1 - r))^{1 / k}) \f$
     *
     * @param ishape The shape of the nd-array drawn per generator.
     * @param k The "shape" parameter \f$ k \f$.
     * @param scale The "scale" parameter \f$ \lambda \f$.
     * @return The array of arrays of samples: [#shape, `ishape`]
     */
    template <class S>
    auto weibull(const S& ishape, double k = 1, double scale = 1) ->
        typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->weibull_impl<R>(ishape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::weibull(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R weibull(const S& ishape, double k = 1, double scale = 1)
    {
        return this->weibull_impl<R>(ishape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::weibull(const S&, double, double)
     */
    template <class I, std::size_t L>
    auto weibull(const I (&ishape)[L], double k = 1, double scale = 1) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->weibull_impl<R>(detail::to_array(ishape), k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::weibull(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R weibull(const I (&ishape)[L], double k = 1, double scale = 1)
    {
        return this->weibull_impl<R>(detail::to_array(ishape), k, scale);
    }

    /**
     * Per generator, generate an nd-array of random numbers distributed
     * according to a Gamma distribution.
     * Only available when compiled with PRRNG_USE_BOOST
     * (e.g. using the CMake target `prrng::use_boost`).
     *
     * @param ishape The shape of the nd-array drawn per generator.
     * @param k Shape parameter.
     * @param scale Scale parameter.
     * @return The array of arrays of samples: [#shape, `ishape`]
     */
    template <class S>
    auto gamma(const S& ishape, double k = 1, double scale = 1) ->
        typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->gamma_impl<R>(ishape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::gamma(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R gamma(const S& ishape, double k = 1, double scale = 1)
    {
        return this->gamma_impl<R>(ishape, k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::gamma(const S&, double, double)
     */
    template <class I, std::size_t L>
    auto gamma(const I (&ishape)[L], double k = 1, double scale = 1) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->gamma_impl<R>(detail::to_array(ishape), k, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::gamma(const S&, double, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R gamma(const I (&ishape)[L], double k = 1, double scale = 1)
    {
        return this->gamma_impl<R>(detail::to_array(ishape), k, scale);
    }

    /**
     * Per generator, generate an nd-array of numbers that are delta distribution.
     * These numbers are not random; calling this function does not change the state of the
     * generators.
     *
     * @param ishape The shape of the nd-array drawn per generator.
     * @param scale The value of the 'peak' of the delta distribution.
     * @return The array of arrays of samples: [#shape, `ishape`]
     */
    template <class S>
    auto delta(const S& ishape, double scale = 1) ->
        typename detail::composite_return_type<double, M, S>::type
    {
        using R = typename detail::composite_return_type<double, M, S>::type;
        return this->delta_impl<R>(ishape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::delta(const S&, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class S>
    R delta(const S& ishape, double scale = 1)
    {
        return this->delta_impl<R>(ishape, scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::delta(const S&, double)
     */
    template <class I, std::size_t L>
    auto delta(const I (&ishape)[L], double scale = 1) ->
        typename detail::composite_return_type<double, M, std::array<size_t, L>>::type
    {
        using R = typename detail::composite_return_type<double, M, std::array<size_t, L>>::type;
        return this->delta_impl<R>(detail::to_array(ishape), scale);
    }

    /**
     * @copydoc prrng::GeneratorBase_array::delta(const S&, double)
     * @tparam R return type, e.g. `xt::xtensor<double, 1>`
     */
    template <class R, class I, std::size_t L>
    R delta(const I (&ishape)[L], double scale = 1)
    {
        return this->delta_impl<R>(detail::to_array(ishape), scale);
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers.
     * @param n Number of steps.
     * @return Cumulative sum.
     */
    template <class T>
    auto cumsum_random(const T& n) -> typename detail::return_type<double, M>::type
    {
        using R = typename detail::return_type<double, M>::type;
        R ret = R::from_shape(m_shape);
        this->cumsum_random_impl(ret.data(), n.data());
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers.
     * @param n Number of steps.
     * @return Cumulative sum.
     */
    template <class R, class T>
    R cumsum_random(const T& n)
    {
        R ret = R::from_shape(m_shape);
        this->cumsum_random_impl(ret.data(), n.data());
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers,
     * distributed according to a normal distribution, see normal_distribution(),
     * @param n Number of steps.
     * @param mu Mean.
     * @param sigma Standard deviation.
     * @return Cumulative sum.
     */
    template <class T>
    auto cumsum_normal(const T& n, double mu = 0, double sigma = 1) ->
        typename detail::return_type<double, M>::type
    {
        using R = typename detail::return_type<double, M>::type;
        R ret = R::from_shape(m_shape);
        this->cumsum_normal_impl(ret.data(), n.data(), mu, sigma);
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers,
     * distributed according to a normal distribution, see normal_distribution(),
     * @param n Number of steps.
     * @param mu Mean.
     * @param sigma Standard deviation.
     * @return Cumulative sum.
     */
    template <class R, class T>
    R cumsum_normal(const T& n, double mu = 0, double sigma = 1)
    {
        R ret = R::from_shape(m_shape);
        this->cumsum_normal_impl(ret.data(), n.data(), mu, sigma);
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers,
     * distributed according to an exponential distribution, see exponential_distribution(),
     * @param n Number of steps.
     * @param scale Scale.
     * @return Cumulative sum.
     */
    template <class T>
    auto cumsum_exponential(const T& n, double scale = 1) ->
        typename detail::return_type<double, M>::type
    {
        using R = typename detail::return_type<double, M>::type;
        R ret = R::from_shape(m_shape);
        this->cumsum_exponential_impl(ret.data(), n.data(), scale);
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers,
     * distributed according to an exponential distribution, see exponential_distribution(),
     * @param n Number of steps.
     * @param scale Scale.
     * @return Cumulative sum.
     */
    template <class R, class T>
    R cumsum_exponential(const T& n, double scale = 1)
    {
        R ret = R::from_shape(m_shape);
        this->cumsum_exponential_impl(ret.data(), n.data(), scale);
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers,
     * distributed according to a weibull distribution, see weibull_distribution(),
     * @param n Number of steps.
     * @param k Shape.
     * @param scale Scale.
     * @return Cumulative sum.
     */
    template <class T>
    auto cumsum_weibull(const T& n, double k = 1, double scale = 1) ->
        typename detail::return_type<double, M>::type
    {
        using R = typename detail::return_type<double, M>::type;
        R ret = R::from_shape(m_shape);
        this->cumsum_weibull_impl(ret.data(), n.data(), k, scale);
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers,
     * distributed according to a weibull distribution, see weibull_distribution(),
     * @param n Number of steps.
     * @param k Shape parameter.
     * @param scale Scale parameter.
     * @return Cumulative sum.
     */
    template <class R, class T>
    R cumsum_weibull(const T& n, double k = 1, double scale = 1)
    {
        R ret = R::from_shape(m_shape);
        this->cumsum_weibull_impl(ret.data(), n.data(), k, scale);
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers,
     * distributed according to a gamma distribution, see gamma_distribution(),
     * @param n Number of steps.
     * @param k Shape parameter.
     * @param scale Scale parameter.
     * @return Cumulative sum.
     */
    template <class T>
    auto cumsum_gamma(const T& n, double k = 1, double scale = 1) ->
        typename detail::return_type<double, M>::type
    {
        using R = typename detail::return_type<double, M>::type;
        R ret = R::from_shape(m_shape);
        this->cumsum_gamma_impl(ret.data(), n.data(), k, scale);
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers,
     * distributed according to a gamma distribution, see gamma_distribution(),
     * @param n Number of steps.
     * @param k Shape parameter.
     * @param scale Scale parameter.
     * @return Cumulative sum.
     */
    template <class R, class T>
    R cumsum_gamma(const T& n, double k = 1, double scale = 1)
    {
        R ret = R::from_shape(m_shape);
        this->cumsum_gamma_impl(ret.data(), n.data(), k, scale);
        return ret;
    }

    /**
     * @brief Decide based on probability per generator.
     * This is fully equivalent to `generators.random({}) <= p`, but avoids the
     * memory allocation of `random``.
     *
     * @param p Probability per generator [0, 1).
     * @return Decision for each `p`.
     */
    template <class P>
    auto decide(const P& p) -> typename detail::return_type<bool, P>::type
    {
        PRRNG_ASSERT(xt::has_shape(p, m_shape));
        using R = typename detail::return_type<bool, P>::type;
        R ret = R::from_shape(m_shape);
        this->decide_impl(p.data(), ret.data());
        return ret;
    }

    /**
     * @copydoc prrng::GeneratorBase_array::decide(const P& p)
     */
    template <class P, class R>
    auto decide(const P& p)
    {
        PRRNG_ASSERT(xt::has_shape(p, m_shape));
        R ret = R::from_shape(m_shape);
        this->decide_impl(p.data(), ret.data());
        return ret;
    }

    /**
     * @brief Decide based on probability per generator.
     * This is fully equivalent to `generators.random({}) <= p`, but avoids the
     * memory allocation of `random``.
     *
     * @param p Probability per generator [0, 1).
     * @param ret Decision for each ``p``.
     */
    template <class P, class R>
    void decide(const P& p, R& ret)
    {
        static_assert(
            std::is_same<typename R::value_type, bool>::value, "Return value_type must be bool");

        PRRNG_ASSERT(xt::has_shape(p, m_shape));
        PRRNG_ASSERT(xt::has_shape(p, ret.shape()));
        this->decide_impl(p.data(), ret.data());
    }

    /**
     * @brief Decide based on probability per generator.
     * This is fully equivalent to `generators.random({}) <= p`,
     * but avoids the memory allocation of `random`.
     *
     * @param p Probability per generator [0, 1).
     * @param mask Where `true` the decision is `false` (no random number is drawn there).
     * @return Decision for each `p`.
     */
    template <class P, class T>
    auto decide_masked(const P& p, const T& mask) -> typename detail::return_type<bool, P>::type
    {
        PRRNG_ASSERT(xt::has_shape(p, m_shape));
        PRRNG_ASSERT(xt::has_shape(p, mask.shape()));
        using R = typename detail::return_type<bool, P>::type;
        R ret = R::from_shape(m_shape);
        this->decide_masked_impl(p.data(), mask.data(), ret.data());
        return ret;
    }

    /**
     * @copydoc prrng::GeneratorBase_array::decide_masked(const P&, const T&)
     */
    template <class P, class T, class R>
    R decide_masked(const P& p, const T& mask)
    {
        PRRNG_ASSERT(xt::has_shape(p, m_shape));
        PRRNG_ASSERT(xt::has_shape(p, mask.shape()));
        R ret = R::from_shape(m_shape);
        this->decide_masked_impl(p.data(), mask.data(), ret.data());
        return ret;
    }

    /**
     * @brief Decide based on probability per generator.
     * This is fully equivalent to `generators.random({}) <= p`,
     * but avoids the memory allocation of `random`.
     *
     * @param p Probability per generator [0, 1).
     * @param mask Where `true` the decision is `false` (no random number is drawn there).
     * @param ret Decision for each ``p``.
     */
    template <class P, class T, class R>
    void decide_masked(const P& p, const T& mask, R& ret)
    {
        static_assert(
            std::is_same<typename R::value_type, bool>::value, "Return value_type must be bool");

        PRRNG_ASSERT(xt::has_shape(p, m_shape));
        PRRNG_ASSERT(xt::has_shape(p, mask.shape()));
        PRRNG_ASSERT(xt::has_shape(p, ret.shape()));
        this->decide_masked_impl(p.data(), mask.data(), ret.data());
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

    template <class R, class S, typename T>
    R randint_impl(const S& ishape, T high)
    {
        static_assert(
            std::numeric_limits<typename R::value_type>::max() >= std::numeric_limits<T>::max(),
            "Return value_type must must be able to accommodate the bound");

        static_assert(
            std::numeric_limits<T>::max() <= std::numeric_limits<uint32_t>::max(),
            "Bound too large");

        auto n = detail::size(ishape);
        R ret = R::from_shape(detail::concatenate<M, S>::two(m_shape, ishape));
        std::vector<uint32_t> tmp(ret.size());
        this->draw_list_uint32(&tmp.front(), static_cast<uint32_t>(high), n);
        std::copy(tmp.begin(), tmp.end(), ret.begin());
        return ret;
    }

    template <class R, class S, typename T, typename U>
    R randint_impl(const S& ishape, T low, U high)
    {
        static_assert(
            std::numeric_limits<typename R::value_type>::max() >= std::numeric_limits<T>::max(),
            "Return value_type must must be able to accommodate the bound");

        static_assert(
            std::numeric_limits<typename R::value_type>::min() >= std::numeric_limits<T>::min(),
            "Return value_type must must be able to accommodate the bound");

        static_assert(
            std::numeric_limits<typename R::value_type>::max() >= std::numeric_limits<U>::max(),
            "Return value_type must must be able to accommodate the bound");

        static_assert(
            std::numeric_limits<typename R::value_type>::min() >= std::numeric_limits<U>::min(),
            "Return value_type must must be able to accommodate the bound");

        static_assert(
            static_cast<uint32_t>(std::numeric_limits<T>::max()) <
                std::numeric_limits<uint32_t>::max(),
            "Bound too large");

        auto n = detail::size(ishape);
        R ret = R::from_shape(detail::concatenate<M, S>::two(m_shape, ishape));
        std::vector<uint32_t> tmp(ret.size());
        this->draw_list_uint32(&tmp.front(), static_cast<uint32_t>(high - low), n);
        std::copy(tmp.begin(), tmp.end(), ret.begin());
        return ret + low;
    }

    template <class R, class S>
    R normal_impl(const S& ishape, double mu, double sigma)
    {
        R r = this->random_impl<R>(ishape);
        return normal_distribution(mu, sigma).quantile(r);
    }

    template <class R, class S>
    R exponential_impl(const S& ishape, double scale)
    {
        R r = this->random_impl<R>(ishape);
        return exponential_distribution(scale).quantile(r);
    }

    template <class R, class S>
    R weibull_impl(const S& ishape, double k, double scale)
    {
        R r = this->random_impl<R>(ishape);
        return weibull_distribution(k, scale).quantile(r);
    }

    template <class R, class S>
    R gamma_impl(const S& ishape, double k, double scale)
    {
        R r = this->random_impl<R>(ishape);
        return gamma_distribution(k, scale).quantile(r);
    }

    template <class R, class S>
    R delta_impl(const S& ishape, double scale)
    {
        R ret = xt::empty<typename R::value_type>(ishape);
        ret.fill(scale);
        return ret;
    }

protected:
    /**
     * @brief For each `p` take a decision.
     * @param p Array of probabilities.
     * @param ret Outcome, same shape as `p`.
     */
    virtual void decide_impl(const double* p, bool* ret)
    {
        for (size_t i = 0; i < m_size; ++i) {
            ret[i] = 0.5 < p[i];
        }
    }

    /**
     * @brief For each `p` take a decision.
     * @param p Array of probabilities.
     * @param mask Mask entries of `p`.
     * @param ret Outcome, same shape as `p`.
     */
    virtual void decide_masked_impl(const double* p, const bool* mask, bool* ret)
    {
        for (size_t i = 0; i < m_size; ++i) {
            if (mask[i]) {
                ret[i] = false;
            }
            else {
                ret[i] = 0.5 < p[i];
            }
        }
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     */
    virtual void cumsum_random_impl(double* ret, const size_t* n)
    {
        for (size_t i = 0; i < m_size; ++i) {
            ret[i] = (double)(n[i]) * 0.5;
        }
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     */
    virtual void cumsum_normal_impl(double* ret, const size_t* n, double, double)
    {
        return cumsum_random_impl(ret, n); // dummy
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     */
    virtual void cumsum_exponential_impl(double* ret, const size_t* n, double)
    {
        return cumsum_random_impl(ret, n); // dummy
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     */
    virtual void cumsum_weibull_impl(double* ret, const size_t* n, double, double)
    {
        return cumsum_random_impl(ret, n); // dummy
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     */
    virtual void cumsum_gamma_impl(double* ret, const size_t* n, double, double)
    {
        return cumsum_random_impl(ret, n); // dummy
    }

    /**
     * Draw `n` random numbers and write them to list (input as pointer `data`).
     *
     * @param data Pointer to the data (no bounds-check).
     * @param n Size of `data`.
     */
    virtual void draw_list(double* data, size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            data[i] = 0.5;
        }
    }

    /**
     * Draw `n` random numbers and write them to list (input as pointer `data`).
     *
     * @param data Pointer to the data (no bounds-check).
     * @param bound Upper bound of the random numbers.
     * @param n Size of `data`.
     */
    virtual void draw_list_uint32(uint32_t* data, uint32_t bound, size_t n)
    {
        (void)(bound);

        for (size_t i = 0; i < n; ++i) {
            data[i] = 0;
        }
    }

protected:
    size_t m_size = 0; ///< See size().
    M m_shape; ///< See shape().
    M m_strides; ///< The strides of the array of generators.
};

/**
 * Base class, see pcg32_array for description.
 */
template <class M>
class pcg32_arrayBase : public GeneratorBase_array<M> {
protected:
    /**
     * @brief Constructor alias.
     *
     * @param initstate State initiator for every item (accept default sequence initiator).
     * The shape of the argument determines the shape of the generator array.
     */
    template <class T>
    void init(const T& initstate)
    {
        std::copy(initstate.shape().cbegin(), initstate.shape().cend(), m_shape.begin());
        std::copy(initstate.strides().cbegin(), initstate.strides().cend(), m_strides.begin());
        m_size = initstate.size();
        m_gen.reserve(m_size);

        for (size_t i = 0; i < m_size; ++i) {
            m_gen.push_back(pcg32(initstate.flat(i)));
        }
    }

    /**
     * @brief Constructor alias.
     *
     * @param initstate State initiator for every item (accept default sequence initiator).
     * @param initseq Sequence initiator for every item.
     * The shape of the argument determines the shape of the generator array.
     */
    template <class T, class U>
    void init(const T& initstate, const U& initseq)
    {
        PRRNG_ASSERT(xt::same_shape(initstate.shape(), initseq.shape()));

        std::copy(initstate.shape().cbegin(), initstate.shape().cend(), m_shape.begin());
        std::copy(initstate.strides().cbegin(), initstate.strides().cend(), m_strides.begin());
        m_size = initstate.size();
        m_gen.reserve(m_size);

        for (size_t i = 0; i < m_size; ++i) {
            m_gen.push_back(pcg32(initstate.flat(i), initseq.flat(i)));
        }
    }

public:
    pcg32_arrayBase() = default;

    virtual ~pcg32_arrayBase() = default;

    /**
     * Return a reference to one generator, using an array index.
     *
     * @param args Array index (number of arguments should correspond to the rank of the array).
     * @return Reference to underlying generator.
     */
    template <class... Args>
    pcg32& operator()(Args... args)
    {
        return m_gen[this->get_item(0, 0, args...)];
    }

    /**
     * Return a constant reference to one generator, using an array index.
     *
     * @param args Array index (number of arguments should correspond to the rank of the array).
     * @return Reference to underlying generator.
     */
    template <class... Args>
    const pcg32& operator()(Args... args) const
    {
        return m_gen[this->get_item(0, 0, args...)];
    }

    /**
     * Return a reference to one generator, using a flat index.
     *
     * @param i Flat index.
     * @return Reference to underlying generator.
     */
    pcg32& operator[](size_t i)
    {
        PRRNG_DEBUG(i < m_size);
        return m_gen[i];
    }

    /**
     * Return a constant reference to one generator, using a flat index.
     *
     * @param i Flat index.
     * @return Reference to underlying generator.
     */
    const pcg32& operator[](size_t i) const
    {
        PRRNG_DEBUG(i < m_size);
        return m_gen[i];
    }

    /**
     * Return a reference to one generator, using a flat index.
     *
     * @param i Flat index.
     * @return Reference to underlying generator.
     */
    pcg32& flat(size_t i)
    {
        PRRNG_DEBUG(i < m_size);
        return m_gen[i];
    }

    /**
     * Return a constant reference to one generator, using a flat index.
     *
     * @param i Flat index.
     * @return Reference to underlying generator.
     */
    const pcg32& flat(size_t i) const
    {
        PRRNG_DEBUG(i < m_size);
        return m_gen[i];
    }

    /**
     * Return the state of all generators.
     * See pcg32::state().
     *
     * @return The state of each generator.
     */
    auto state() -> typename detail::return_type<uint64_t, M>::type
    {
        using R = typename detail::return_type<uint64_t, M>::type;
        return this->state<R>();
    }

    /**
     * @copydoc prrng::pcg32_arrayBase::state()
     *
     * @tparam R The type of the return array, e.g. `xt::array<uint64_t>` or `xt::xtensor<uint64_t,
     * N>`
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
     * Return the state initiator of all generators.
     * See pcg32::initstate().
     *
     * @return The state initiator of each generator.
     */
    auto initstate() -> typename detail::return_type<uint64_t, M>::type
    {
        using R = typename detail::return_type<uint64_t, M>::type;
        return this->initstate<R>();
    }

    /**
     * @copydoc prrng::pcg32_arrayBase::initstate()
     *
     * @return The state initiator of each generator.
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
     * Return the sequence initiator of all generators.
     * See pcg32::initseq().
     *
     * @return The sequence initiator of each generator.
     */
    auto initseq() -> typename detail::return_type<uint64_t, M>::type
    {
        using R = typename detail::return_type<uint64_t, M>::type;
        return this->initseq<R>();
    }

    /**
     * @copydoc prrng::pcg32_arrayBase::initseq()
     *
     * @tparam R The type of the return array, e.g. `xt::array<uint64_t>` or `xt::xtensor<uint64_t,
     * N>`
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
     * Distance between two states.
     * See pcg32::distance().
     *
     * @tparam T Array, e.g. `xt::array<int64_t>` or `xt::xtensor<int64_t, N>`.
     * @param arg The state to which to compare.
     */
    template <class T>
    auto distance(const T& arg) -> typename detail::return_type<int64_t, M>::type
    {
        using R = typename detail::return_type<int64_t, M>::type;
        return this->distance<R, T>(arg);
    }

    /**
     * Distance between two states.
     * See pcg32::distance().
     *
     * @tparam R Array, e.g. `xt::array<int64_t>` or `xt::xtensor<int64_t, N>`.
     * @tparam T Array, e.g. `xt::array<int64_t>` or `xt::xtensor<int64_t, N>`.
     * @param arg The state to which to compare.
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
     * Advance generators.
     * See pcg32::advance().
     *
     * @tparam T The type of the input array, e.g. `xt::array<int64_t>` or `xt::xtensor<int64_t, N>`
     *
     * @param arg The distance (positive or negative) by which to advance each generator.
     */
    template <class T>
    void advance(const T& arg)
    {
        for (size_t i = 0; i < m_size; ++i) {
            m_gen[i].advance(arg.flat(i));
        }
    }

    /**
     * Restore generators from a state.
     * See pcg32::restore().
     *
     * @tparam T The type of the input array, e.g. `xt::array<uint64_t>` or `xt::xtensor<uint64_t,
     * N>`
     *
     * @param arg The state of each generator.
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
     * @brief For each `p` take a decision.
     * @param p Array of probabilities.
     * @param ret Outcome, same shape as `p`.
     */
    void decide_impl(const double* p, bool* ret) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            ret[i] = m_gen[i].next_double() < p[i];
        }
    }

    /**
     * @brief For each `p` take a decision.
     * @param p Array of probabilities.
     * @param mask Mask entries of `p`.
     * @param ret Outcome, same shape as `p`.
     */
    void decide_masked_impl(const double* p, const bool* mask, bool* ret) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            if (mask[i]) {
                ret[i] = false;
            }
            else {
                ret[i] = m_gen[i].next_double() < p[i];
            }
        }
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     */
    void cumsum_random_impl(double* ret, const size_t* n) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            ret[i] = m_gen[i].cumsum_random(n[i]);
        }
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     * @param mu Mean.
     * @param sigma Standard deviation.
     */
    void cumsum_normal_impl(double* ret, const size_t* n, double mu, double sigma) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            ret[i] = m_gen[i].cumsum_normal(n[i], mu, sigma);
        }
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     * @param scale Scale.
     */
    void cumsum_exponential_impl(double* ret, const size_t* n, double scale) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            ret[i] = m_gen[i].cumsum_exponential(n[i], scale);
        }
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     * @param k Shape parameter.
     * @param scale Scale parameter.
     */
    void cumsum_weibull_impl(double* ret, const size_t* n, double k, double scale) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            ret[i] = m_gen[i].cumsum_weibull(n[i], k, scale);
        }
    }

    /**
     * @brief Return the result of the cumulative sum of `n` random numbers.
     * @param ret Output, per generator.
     * @param n Number to draw, per generator.
     * @param k Shape parameter.
     * @param scale Scale parameter.
     */
    void cumsum_gamma_impl(double* ret, const size_t* n, double k, double scale) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            ret[i] = m_gen[i].cumsum_gamma(n[i], k, scale);
        }
    }

    /**
     * Draw `n` random numbers per array item, and write them to the correct position in `data`
     * (assuming row-major storage!).
     *
     * @param data Pointer to the data (no bounds-check).
     * @param n The number of random numbers per generator.
     */
    void draw_list(double* data, size_t n) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            for (size_t j = 0; j < n; ++j) {
                data[i * n + j] = m_gen[i].next_double();
            }
        }
    }

    /**
     * Draw `n` random numbers per array item, and write them to the correct position in `data`
     * (assuming row-major storage!).
     *
     * @param data Pointer to the data (no bounds-check).
     * @param bound The upper bound of the random numbers.
     * @param n The number of random numbers per generator.
     */
    void draw_list_uint32(uint32_t* data, uint32_t bound, size_t n) override
    {
        for (size_t i = 0; i < m_size; ++i) {
            for (size_t j = 0; j < n; ++j) {
                data[i * n + j] = m_gen[i].next_uint32(bound);
            }
        }
    }

private:
    /**
     * implementation of `operator()`.
     * (Last call in recursion).
     */
    template <class T>
    size_t get_item(size_t sum, size_t d, T arg)
    {
        return sum + arg * m_strides[d];
    }

    /**
     * implementation of `operator()`.
     * (Called recursively).
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
 * Array of independent generators.
 * The idea is that each array-entry has its own random sequence, initiated by its own seed.
 * An array of random numbers can then be generated whose shape if composed of the #shape,
 * the shape of the array of generators, followed by the desired shape of the random sequence
 * draw per generator.
 * Let us consider an example. Suppose that we have a list of n = 5 generators,
 * and we want to generate i = 8 random numbers for each generator.
 * Then the output will be collected in a matrix of shape [n, i] = [5, 8] where each row
 * corresponds to a generator and the columns for that row are the random sequence generated
 * by that generator.
 * Since this class is general, you can also imagine an array of [m, n, o, p] generators with
 * a random sequence reshaped in a (row-major) array of shape [a, b, c, d, e].
 * The output is then collected in an array of shape [m, n, o, p, a, b, c, d, e].
 *
 * Note that a reference to each generator can be obtained using the `[]` and `()` operators,
 * e.g. `generators[flat_index]` and `generators(i, j, k, ...)`. All functions of pcg32()
 * can be used for each reference.
 * In addition, convenience functions state(), initstate(), initseq(), restore() are provided
 * here to store/restore the state of the entire array of generators.
 */
class pcg32_array : public pcg32_arrayBase<std::vector<size_t>> {
public:
    pcg32_array() = default;

    /**
     * Constructor.
     *
     * @param initstate State initiator for every item (accept default sequence initiator).
     * The shape of the argument determines the shape of the generator array.
     */
    template <class T>
    pcg32_array(const T& initstate)
    {
        m_shape.resize(initstate.dimension());
        m_strides.resize(initstate.dimension());
        this->init(initstate);
    }

    /**
     * Constructor.
     *
     * @param initstate State initiator for every item.
     * @param initseq Sequence initiator for every item.
     * The shape of these argument determines the shape of the generator array.
     */
    template <class T, class U>
    pcg32_array(const T& initstate, const U& initseq)
    {
        m_shape.resize(initstate.dimension());
        m_strides.resize(initstate.dimension());
        this->init(initstate, initseq);
    }

protected:
    using pcg32_arrayBase<std::vector<size_t>>::m_gen;
    using GeneratorBase_array<std::vector<size_t>>::m_size;
    using GeneratorBase_array<std::vector<size_t>>::m_shape;
    using GeneratorBase_array<std::vector<size_t>>::m_strides;
};

/**
 * Fixed rank version of pcg32_array
 */
template <size_t N>
class pcg32_tensor : public pcg32_arrayBase<std::array<size_t, N>> {
public:
    pcg32_tensor() = default;

    /**
     * Constructor.
     *
     * @param initstate State initiator for every item (accept default sequence initiator).
     * The shape of the argument determines the shape of the generator array.
     */
    template <class T>
    pcg32_tensor(const T& initstate)
    {
        static_assert(detail::check_fixed_rank<N, T>::value, "Ranks to not match");
        this->init(initstate);
    }

    /**
     * Constructor.
     *
     * @param initstate State initiator for every item.
     * @param initseq Sequence initiator for every item.
     * The shape of these argument determines the shape of the generator array.
     */
    template <class T, class U>
    pcg32_tensor(const T& initstate, const U& initseq)
    {
        static_assert(detail::check_fixed_rank<N, T>::value, "Ranks to not match");
        this->init(initstate, initseq);
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
 * Return a pcg32, a pcg32_array, or a pcg32_tensor based on input.
 *
 * @param initstate The sequence initiator.
 * @return The allocated generator.
 */
template <class T>
inline auto auto_pcg32(const T& initstate)
{
    return detail::auto_pcg32<T>::get(initstate);
}

/**
 * Return a pcg32, a pcg32_array, or a pcg32_tensor based on input.
 *
 * @param initstate The sequence initiator.
 * @param initseq The sequence initiator.
 * @return The allocated generator.
 */
template <class T, class S>
inline auto auto_pcg32(const T& initstate, const S& initseq)
{
    return detail::auto_pcg32<T>::get(initstate, initseq);
}

/**
 * @brief Array of generators of a random cumulative sum.
 * A chunk is kept in memory for each generator, whereby all chunks are assembled to one big array.
 * The random number generated by the pcg32 algorithm.
 *
 * @tparam D Storage of the data, e.g. xt::xarray<double>.
 * @tparam G Storage of the generator array, e.g. prrng::pcg32_array
 */
template <class D, class G>
class pcg32_arrayBase_cumsum {
private:
    D m_data;
    std::vector<pcg32_cumsum_external> m_cumsum;
    G m_gen;

protected:
    /**
     * @brief Constructor alias.
     *
     * @param shape Shape of the chunk to keep in memory.
     * @param initstate State initiator for every item.
     * @param initseq Sequence initiator for every item.
     */
    template <class S, class T, class U>
    void init(const S& shape, const T& initstate, const U& initseq)
    {
        PRRNG_ASSERT(xt::same_shape(initstate.shape(), initseq.shape()));
        using shape_type = typename S::value_type;

        std::vector<size_t> data_shape;
        data_shape.resize(initstate.dimension() + shape.size());
        std::copy(initstate.shape().begin(), initstate.shape().end(), data_shape.begin());
        std::copy(shape.begin(), shape.end(), data_shape.begin() + initstate.dimension());
        m_data = xt::empty<typename D::value_type>(data_shape);

        m_gen = G(initstate, initseq);

        m_cumsum.reserve(m_gen.size());
        size_t n = static_cast<size_t>(
            std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<shape_type>{}));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum.push_back(pcg32_cumsum_external(&m_data.flat(i * n), n, &m_gen[i], 0));
        }
    }

public:
    pcg32_arrayBase_cumsum() = default;

    /**
     * @brief Generator array.
     * @return Pointer to generator array.
     */
    const G& generators() const
    {
        return m_gen;
    }

    /**
     * @brief Chunk.
     * @return Pointer to chunk.
     */
    const D& chunk() const
    {
        return m_data;
    }

    /**
     * @brief Overwrite chunk.
     * Please consider if prrng::pcg32_arrayBase_cumsum::set_state() and
     * prrng::pcg32_arrayBase_cumsum::set_start() should be called as well.
     *
     * @param data New chunk.
     */
    void set_chunk(const D& data)
    {
#ifdef PRRNG_ENABLE_ASSERT
        std::vector<size_t> shape(m_gen.shape().size());
        std::copy(data.shape().cbegin(), data.shape().cend(), shape.begin());
        PRRNG_ASSERT(std::equal(shape.begin(), shape.end(), m_gen.shape().cbegin()));
#endif
        m_data = data;

        size_t n = static_cast<size_t>(std::accumulate(
            data.shape().cbegin() + m_gen.shape().size(),
            data.shape().cend(),
            1,
            std::multiplies<typename D::size_type>{}));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].set_data(&m_data.flat(i * n), n);
        }
    }

    /**
     * @brief Current index of the generators.
     *
     * @return Array of indices.
     */
    template <class R>
    R generator_index() const
    {
        using value_type = typename R::value_type;
        R ret = R::from_shape(m_gen.shape());

        for (size_t i = 0; i < m_gen.size(); ++i) {
            ret.flat(i) = static_cast<value_type>(m_cumsum[i].generator_index());
        }

        return ret;
    }

    /**
     * @brief Overwrite current index of the generators.
     *
     * @param index Array of indices.
     */
    template <class T>
    void set_generator_index(const T& index)
    {
        PRRNG_ASSERT(xt::same_shape(index.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].set_generator_index(index.flat(i));
        }
    }

    /**
     * @brief Start index of the chunk.
     *
     * @return Array of indices.
     */
    template <class R>
    R start() const
    {
        using value_type = typename R::value_type;
        R ret = R::from_shape(m_gen.shape());

        for (size_t i = 0; i < m_gen.size(); ++i) {
            ret.flat(i) = static_cast<value_type>(m_cumsum[i].start());
        }

        return ret;
    }

    /**
     * @brief Overwrite start index of the chunk.
     *
     * @param index Array of indices.
     */
    template <class T>
    void set_start(const T& index)
    {
        PRRNG_ASSERT(xt::same_shape(index.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].set_start(index.flat(i));
        }
    }

    /**
     * @brief State of the generators.
     *
     * @return Array of states.
     */
    template <class R, class T>
    R state(const T& index)
    {
        PRRNG_ASSERT(xt::same_shape(index.shape(), m_gen.shape()));

        using value_type = typename R::value_type;
        R ret = R::from_shape(m_gen.shape());

        for (size_t i = 0; i < m_gen.size(); ++i) {
            ret.flat(i) = static_cast<value_type>(m_cumsum[i].state(index.flat(i)));
        }

        return ret;
    }

    /**
     * @brief Overwrite state of the generators.
     *
     * @param state Array of states.
     * @param index Index that the states correspond to.
     */
    template <class S, class T>
    void set_state(const S& state, const T& index)
    {
        PRRNG_ASSERT(xt::same_shape(state.shape(), m_gen.shape()));
        PRRNG_ASSERT(xt::same_shape(index.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].set_state(state.flat(i), index.flat(i));
        }
    }

    /**
     * @brief Restore state.
     * A `draw...` function should be called next.
     *
     * @param state Array of states.
     * @param value Array of values to begin the chunk with.
     * @param index Index that the states correspond to.
     */
    template <class S, class V, class T>
    void restore(const S& state, const V& value, const T& index)
    {
        PRRNG_ASSERT(xt::same_shape(state.shape(), m_gen.shape()));
        PRRNG_ASSERT(xt::same_shape(value.shape(), m_gen.shape()));
        PRRNG_ASSERT(xt::same_shape(index.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].restore(state.flat(i), value.flat(i), index.flat(i));
        }
    }

    /**
     * @brief Draw a new chunk.
     * See prrng::pcg32_cumsum::draw_weibull().
     *
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     */
    void draw_chunk_weibull(double k = 1, double scale = 1, double offset = 0)
    {
        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].draw_chunk_weibull(k, scale, offset);
        }
    }

    /**
     * @brief Align chunks with a target value.
     * See prrng::pcg32_cumsum::align_weibull().
     *
     * @param target Target value.
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    template <class T>
    void align_chunk_weibull(
        const T& target,
        double k = 1,
        double scale = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        PRRNG_ASSERT(xt::same_shape(target.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].align_chunk_weibull(target.flat(i), k, scale, offset, margin, strict);
        }
    }

    /**
     * @brief Draw a new chunk.
     * See prrng::pcg32_cumsum::draw_gamma().
     *
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     */
    void draw_chunk_gamma(double k = 1, double scale = 1, double offset = 0)
    {
        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].draw_chunk_gamma(k, scale, offset);
        }
    }

    /**
     * @brief Align chunks with a target value.
     * See prrng::pcg32_cumsum::align_gamma().
     *
     * @param target Target value.
     * @param k Shape factor.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    template <class T>
    void align_chunk_gamma(
        const T& target,
        double k = 1,
        double scale = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        PRRNG_ASSERT(xt::same_shape(target.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].align_chunk_gamma(target.flat(i), k, scale, offset, margin, strict);
        }
    }

    /**
     * @brief Draw a new chunk.
     * See prrng::pcg32_cumsum::draw_normal().
     *
     * @param mu Mean.
     * @param sigma Standard deviation.
     * @param offset Fixed offset.
     */
    void draw_chunk_normal(double mu = 0, double sigma = 1, double offset = 0)
    {
        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].draw_chunk_normal(mu, sigma, offset);
        }
    }

    /**
     * @brief Align chunks with a target value.
     * See prrng::pcg32_cumsum::align_normal().
     *
     * @param target Target value.
     * @param mu Mean.
     * @param sigma Standard deviation.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    template <class T>
    void align_chunk_normal(
        const T& target,
        double mu = 0,
        double sigma = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        PRRNG_ASSERT(xt::same_shape(target.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].align_chunk_normal(target.flat(i), mu, sigma, offset, margin, strict);
        }
    }

    /**
     * @brief Draw a new chunk.
     * See prrng::pcg32_cumsum::draw_exponential().
     *
     * @param scale Scale factor.
     * @param offset Fixed offset.
     */
    void draw_chunk_exponential(double scale = 1, double offset = 0)
    {
        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].draw_chunk_exponential(scale, offset);
        }
    }

    /**
     * @brief Align chunks with a target value.
     * See prrng::pcg32_cumsum::align_exponential().
     *
     * @param target Target value.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    template <class T>
    void align_chunk_exponential(
        const T& target,
        double scale = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        PRRNG_ASSERT(xt::same_shape(target.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].align_chunk_exponential(target.flat(i), scale, offset, margin, strict);
        }
    }

    /**
     * @brief Draw a new chunk.
     * See prrng::pcg32_cumsum::draw_delta().
     *
     * @param scale Scale factor.
     * @param offset Fixed offset.
     */
    void draw_chunk_delta(double scale = 1, double offset = 0)
    {
        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].draw_chunk_delta(scale, offset);
        }
    }

    /**
     * @brief Align chunks with a target value.
     * See prrng::pcg32_cumsum::align_delta().
     *
     * @param target Target value.
     * @param scale Scale factor.
     * @param offset Fixed offset.
     * @param margin Margin to leave left of the target.
     * @param strict If `false` the margin is only approximately enforced to gain speed.
     */
    template <class T>
    void align_chunk_delta(
        const T& target,
        double scale = 1,
        double offset = 0,
        size_t margin = 0,
        bool strict = false)
    {
        PRRNG_ASSERT(xt::same_shape(target.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_cumsum[i].align_chunk_delta(target.flat(i), scale, offset, margin, strict);
        }
    }
};

/**
 * @brief Array of generators of a random cumulative sum.
 * A chunk is kept in memory for each generator, whereby all chunks are assembled to one big array.
 * The random number generated by the pcg32 algorithm.
 *
 * @tparam D Storage of the data, e.g. xt::xarray<double>.
 */
template <class D>
class pcg32_array_cumsum : public pcg32_arrayBase_cumsum<D, pcg32_array> {
public:
    pcg32_array_cumsum() = default;

    /**
     * @brief Constructor.
     *
     * @param shape Shape of the chunk to keep in memory.
     * @param initstate State initiator for every item.
     * @param initseq Sequence initiator for every item.
     */
    template <class S, class T, class U>
    pcg32_array_cumsum(const S& shape, const T& initstate, const U& initseq)
    {
        this->init(shape, initstate, initseq);
    }
};

/**
 * @brief Array of generators of a random cumulative sum.
 * A chunk is kept in memory for each generator, whereby all chunks are assembled to one big array.
 * The random number generated by the pcg32 algorithm.
 *
 * @tparam D Storage of the data, e.g. xt::tensor<double, N + n>.
 * @tparam N Rank of the array of generators, e.g. xt::xarray<double, N>.
 */
template <class D, size_t N>
class pcg32_tensor_cumsum : public pcg32_arrayBase_cumsum<D, pcg32_tensor<N>> {
public:
    pcg32_tensor_cumsum() = default;

    /**
     * @brief Constructor.
     *
     * @param shape Shape of the chunk to keep in memory.
     * @param initstate State initiator for every item.
     * @param initseq Sequence initiator for every item.
     */
    template <class S, class T, class U>
    pcg32_tensor_cumsum(const S& shape, const T& initstate, const U& initseq)
    {
        this->init(shape, initstate, initseq);
    }
};

} // namespace prrng

#endif
