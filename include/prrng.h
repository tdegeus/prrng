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
#include <xtensor/xnoalias.hpp>
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

/**
 * @brief Distribution identifier.
 */
enum distribution {
    random, ///< flat
    weibull, ///< weibull
    gamma, ///< gamma
    normal, ///< normal
    exponential, ///< exponential
    delta, ///< delta
    custom ///< unknown
};

namespace detail {

/**
 * @brief Default parameters
 * @param distribution Distribution.
 * @param parameters Parameters.
 * @return Parameters with defaults filled.
 */
std::vector<double>
default_parameters_cumsum(enum distribution distribution, const std::vector<double>& parameters)
{
    std::vector<double> ret;
    switch (distribution) {
    case distribution::random:
        ret = std::vector<double>{1, 0};
        break;
    case distribution::exponential:
        ret = std::vector<double>{1, 0};
        break;
    case distribution::delta:
        ret = std::vector<double>{1, 0};
        break;
    case distribution::weibull:
        ret = std::vector<double>{1, 1, 0};
        break;
    case distribution::gamma:
        ret = std::vector<double>{1, 1, 0};
        break;
    case distribution::normal:
        ret = std::vector<double>{1, 0, 0};
        break;
    case distribution::custom:
        std::vector<double>{};
        break;
    }

    PRRNG_ASSERT(parameters.size() <= ret.size());
    std::copy(parameters.begin(), parameters.end(), ret.begin());
    return ret;
}

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

/**
 * @brief Replace a character (or a sequence of characters) in a string.
 * @param str The string.
 * @param from Search string.
 * @param to Replacement string.
 * @return `str` with `from` replaced by `to`.
 */
inline std::string replace(std::string str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
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
 * Return versions of this library and of all of its dependencies.
 * The output is a list of strings, e.g.:
 *
 *     "prrng=1.7.0",
 *     "xtensor=0.20.1"
 *     ...
 *
 * @return List of strings.
 */
inline std::vector<std::string> version_dependencies()
{
    std::vector<std::string> ret;

    ret.push_back("prrng=" + version());

    ret.push_back(
        "xtensor=" + detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));

#ifdef XSIMD_VERSION_MAJOR
    ret.push_back(
        "xsimd=" + detail::unquote(std::string(QUOTE(XSIMD_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XSIMD_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XSIMD_VERSION_PATCH))));
#endif

#ifdef XTL_VERSION_MAJOR
    ret.push_back(
        "xtl=" + detail::unquote(std::string(QUOTE(XTL_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTL_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTL_VERSION_PATCH))));
#endif

#if defined(XTENSOR_PYTHON_VERSION_MAJOR)
    ret.push_back(
        "xtensor-python=" + detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MAJOR))) +
        "." + detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_PATCH))));
#endif

#ifdef BOOST_VERSION
    ret.push_back(
        "boost=" + detail::unquote(std::to_string(BOOST_VERSION / 100000)) + "." +
        detail::unquote(std::to_string((BOOST_VERSION / 100) % 1000)) + "." +
        detail::unquote(std::to_string(BOOST_VERSION % 100)));
#endif

    std::sort(ret.begin(), ret.end(), std::greater<std::string>());

    return ret;
}

/**
 * Return information on the compiler, the platform, the C++ standard, and the compilation data.
 * @return List of strings.
 */
inline std::vector<std::string> version_compiler()
{
    std::vector<std::string> ret;

#ifdef __DATE__
    std::string date = detail::unquote(std::string(QUOTE(__DATE__)));
    ret.push_back("date=" + detail::replace(detail::replace(date, " ", "-"), "--", "-"));
#endif

#ifdef __APPLE__
    ret.push_back("platform=apple");
#endif

#ifdef __MINGW32__
    ret.push_back("platform=mingw");
#endif

#ifdef __linux__
    ret.push_back("platform=linux");
#endif

#ifdef _WIN32
    ret.push_back("platform=windows");
#else
#ifdef WIN32
    ret.push_back("platform=windows");
#endif
#endif

#ifdef __clang_version__
    ret.push_back(
        "clang=" + detail::unquote(std::string(QUOTE(__clang_major__))) + "." +
        detail::unquote(std::string(QUOTE(__clang_minor__))) + "." +
        detail::unquote(std::string(QUOTE(__clang_patchlevel__))));
#endif

#ifdef __GNUC__
    ret.push_back(
        "gcc=" + detail::unquote(std::string(QUOTE(__GNUC__))) + "." +
        detail::unquote(std::string(QUOTE(__GNUC_MINOR__))) + "." +
        detail::unquote(std::string(QUOTE(__GNUC_PATCHLEVEL__))));
#endif

#ifdef _MSC_VER
    ret.push_back("msvc=" + std::to_string(_MSC_VER));
#endif

    // c++ version

#ifdef __cplusplus
    ret.push_back("c++=" + detail::unquote(std::string(QUOTE(__cplusplus))));
#endif

    std::sort(ret.begin(), ret.end(), std::greater<std::string>());

    return ret;
}

namespace iterator {

/**
 * Return index of the first element in the range [first, last) such that `element < value` is
 * `false` (i.e. greater or equal to), or last if no such element is found.
 *
 * Compared to the default function, this function allows for a guess of the index and a proximity
 * search around. This could be efficient for finding items in large arrays.
 *
 * @param first Iterator defining the beginning of the range to examine (e.g. `a.begin()`).
 * @param last Iterator defining the end of the range to examine (e.g. `a.end()`)
 * @param value Value to find.
 * @param guess Guess of the index where to find the value.
 * @param proximity Size of the proximity search around `guess` (use `0` to disable proximity
 * search).
 * @return The index of `value` (i.e. `a[index] < value <= a[index + 1]`).
 */
template <class It, class T, class R = size_t>
inline R lower_bound(const It first, const It last, const T& value, R guess = 0, R proximity = 10)
{
    if (proximity == 0) {
        if (value <= *(first)) {
            return 0;
        }
        return std::lower_bound(first, last, value) - first - 1;
    }

    if (*(first + guess) < value && value <= *(first + guess + 1)) {
        return guess;
    }

    R l = guess > proximity ? guess - proximity : 0;
    R r = std::min(guess + proximity, static_cast<R>(last - first - 1));

    if (*(first + l) < value && *(first + r) >= value) {
        return std::lower_bound(first + l, first + r, value) - first - 1;
    }
    else if (value <= *(first)) {
        return 0;
    }
    else {
        return std::lower_bound(first, last, value) - first - 1;
    }
}

} // namespace iterator

namespace inplace {

/**
 * Similar to `lower_bound` but on the last axis of an nd-array (e.g. per row of a rank 2 matrix).
 *
 * @param matrix The matrix defining a range per row.
 * @param value The value to find (per row).
 * @param index Initial guess on `index` (updated).
 * @param proximity Size of the proximity search around `guess` (use `0` to disable proximity
 * search).
 */
template <class T, class V, class R>
inline void lower_bound(const T& matrix, const V& value, R& index, size_t proximity = 10)
{
    PRRNG_ASSERT(value.dimension() == matrix.dimension() - 1);
    PRRNG_ASSERT(value.dimension() == index.dimension());

    auto nd = value.dimension();
    auto stride = matrix.shape(nd);
    auto n = value.size();

#ifdef PRRNG_ENABLE_ASSERT
    for (decltype(nd) i = 0; i < nd; ++i) {
        PRRNG_ASSERT(matrix.shape(i) == value.shape(i));
        PRRNG_ASSERT(matrix.shape(i) == index.shape(i));
    }
#endif

    for (decltype(n) i = 0; i < n; ++i) {
        index.flat(i) = iterator::lower_bound(
            &matrix.flat(i * stride),
            &matrix.flat(i * stride) + stride,
            value.flat(i),
            index.flat(i),
            proximity);
    }
}

/**
 * Update the chunk of a cumsum computed and stored in chunks.
 *
 * @param cumsum The current chunk of the cumsum (updated).
 * @param delta The 'diff's of the next chunk in the cumsum.
 * @param shift The shift per row.
 */
template <class V, class I>
inline void cumsum_chunk(V& cumsum, const V& delta, const I& shift)
{
    PRRNG_ASSERT(cumsum.dimension() >= 1);
    PRRNG_ASSERT(cumsum.dimension() == delta.dimension());

    if (delta.size() == 0) {
        return;
    }

    size_t dim = cumsum.dimension();
    size_t n = cumsum.shape(dim - 1);
    size_t ndelta = delta.shape(dim - 1);

    for (size_t i = 0; i < shift.size(); ++i) {

        if (shift.flat(i) == 0) {
            continue;
        }

        auto* d = &delta.flat(i * ndelta);
        auto* c = &cumsum.flat(i * n);

        if (shift.flat(i) > 0) {
            PRRNG_ASSERT(shift.flat(i) <= static_cast<typename I::value_type>(n));
            PRRNG_ASSERT(static_cast<typename I::value_type>(ndelta) >= shift.flat(i));
            size_t nadd = static_cast<size_t>(shift.flat(i));
            size_t nkeep = n - nadd;
            auto offset = *(c + n - 1);
            std::copy(c + n - nkeep, c + n, c);
            std::copy(d, d + nadd, c + nkeep);
            *(c + nkeep) += offset;
            std::partial_sum(c + nkeep, c + n, c + nkeep);
        }
        else {
            PRRNG_ASSERT(-shift.flat(i) < static_cast<typename I::value_type>(n));
            PRRNG_ASSERT(static_cast<typename I::value_type>(ndelta) > -shift.flat(i));
            size_t nadd = static_cast<size_t>(-shift.flat(i));
            size_t nkeep = n - nadd;
            auto offset = *(c);
            std::copy(c, c + nkeep, c + nadd);
            std::copy(d, d + nadd + 1, c);
            std::partial_sum(c, c + nadd + 1, c);
            offset -= *(c + nadd);
            std::transform(c, c + nadd + 1, c, [&](auto& v) { return v + offset; });
        }
    }
}

} // namespace inplace

/**
 * Iterating on the last axis of an nd-array (e.g. per row of a rank 2 matrix):
 * Return index of the first element in the range [first, last) such that `element < value` is
 * `false` (i.e. greater or equal to), or last if no such element is found.
 *
 * This function allows for a guess of the index and a proximity search around.
 * This could be efficient for finding items in large arrays.
 *
 * @param matrix The matrix defining a range per row.
 * @param value The value to find (per row).
 * @param index Initial guess on `index`.
 * @param proximity Size of the proximity search around `guess` (use `0` to disable proximity
 * search).
 * @return Same shape as `index`.
 */
template <class T, class V, class R>
inline R lower_bound(const T& matrix, const V& value, const R& index, size_t proximity = 10)
{
    R ret = index;
    inplace::lower_bound(matrix, value, ret, proximity);
    return ret;
}

/**
 * Iterating on the last axis of an nd-array (e.g. per row of a rank 2 matrix):
 * Return index of the first element in the range [first, last) such that `element < value` is
 * `false` (i.e. greater or equal to), or last if no such element is found.
 *
 * @param matrix The matrix defining a range per row.
 * @param value The value to find (per row).
 * @return Same shape as `value`.
 */
template <class T, class V, class R>
inline R lower_bound(const T& matrix, const V& value)
{
    R ret = xt::zeros<typename R::value_type>(value.shape());
    inplace::lower_bound(matrix, value, ret, 0);
    return ret;
}

/**
 * @brief Update the chunk of a cumsum computed and stored in chunks.
 *
 * \section example Example
 *
 * Consider a full array:
 *
 *     da = np.random.random(N)
 *     a = np.cumsum(a)
 *
 * With chunk settings:
 *
 *     n = ...  # size of each new chunk
 *     nbuffer = ... # number of items to buffer
 *
 * The the first chunk:
 *
 *     chunk = np.copy(a[:n + nbuffer])
 *     nchunk = n + nbuffer
 *     istart = np.array(0)
 *
 * Then, moving right:
 *
 *     prrng.cumsum_chunk_inplace(chunk, da[istart + nchunk : istart + nchunk + n], n)
 *     istart += n
 *
 * Or, moving left:
 *
 *     prrng.cumsum_chunk_inplace(chunk, da[istart - n : istart + 1], -n)
 *     istart -= n
 *
 * @param cumsum The current chunk of the cumsum.
 * @param delta The 'diff's of the next chunk in the cumsum.
 * @param shift The shift per row.
 * @return Same shape as `cumsum`.
 */
template <class V, class I>
inline V cumsum_chunk(const V& cumsum, const V& delta, const I& shift)
{
    V ret = cumsum;
    inplace::cumsum_chunk(ret, delta, shift);
    return ret;
}

namespace detail {

/**
 * Shift chunk left.
 *
 * @param generator Generator, see prrng::pcg32_index() (modified).
 * @param get_chunk Function to draw the random numbers, called as `get_chunk(n)`.
 * @param margin Overlap to keep with the current chunk.
 * @param data Pointer to the chunk (modified).
 * @param size Size of the chunk.
 * @param start Start index of the chunk (modified).
 */
template <class G, class D>
void prev(
    G& generator,
    const D& get_chunk,
    ptrdiff_t margin,
    double* data,
    ptrdiff_t size,
    ptrdiff_t* start)
{
    using R = decltype(get_chunk(size_t{}));

    generator.jump_to(*start - size + margin);

    double front = data[0];
    ptrdiff_t m = size - margin + 1;
    R extra = get_chunk(static_cast<size_t>(m));
    generator.drawn(m);
    std::partial_sum(extra.begin(), extra.end(), extra.begin());
    extra -= extra.back() - front;

    std::copy(data, data + margin, data + size - margin);
    std::copy(extra.begin(), extra.end() - 1, data);

    *start -= size - margin;
}

/**
 * Shift chunk right.
 *
 * @param generator Generator, see prrng::pcg32_index() (modified).
 * @param get_chunk Function to draw the random numbers, called as `get_chunk(n)`.
 * @param margin Overlap to keep with the current chunk.
 * @param data Pointer to the chunk (modified).
 * @param size Size of the chunk.
 * @param start Start index of the chunk (modified).
 */
template <class G, class D>
void next(
    G& generator,
    const D& get_chunk,
    ptrdiff_t margin,
    double* data,
    ptrdiff_t size,
    ptrdiff_t* start)
{
    using R = decltype(get_chunk(size_t{}));
    PRRNG_ASSERT(margin < size);

    generator.jump_to(*start + size);

    double back = data[size - 1];
    ptrdiff_t n = size - margin;
    R extra = get_chunk(static_cast<size_t>(n));
    generator.drawn(n);
    extra.front() += back;
    std::partial_sum(extra.begin(), extra.end(), extra.begin());
    std::copy(data + size - margin, data + size, data);
    std::copy(extra.begin(), extra.end(), data + margin);
    *start += n;
}

/**
 * Align the chunk to encompass a target value.
 *
 * @param generator Generator, see prrng::pcg32_index() (modified).
 * @param get_chunk Function to draw the random numbers, called as `get_chunk(n)`.
 * @param get_sum Function to get the cumsum of random numbers, called: `get_sum(n)`.
 * @param param Alignment parameters, see prrng::alignment().
 * @param data Pointer to the chunk (modified).
 * @param size Size of the chunk.
 * @param start Start index of the chunk (modified).
 * @param i Last index of `target` relative to the start of the chunk (modified).
 * @param target Target value.
 * @param recursive Used internally to distinguish between external and internal calls.
 */
template <class G, class D, class S, class P>
void align(
    G& generator,
    const D& get_chunk,
    const S& get_sum,
    const P& param,
    double* data,
    ptrdiff_t size,
    ptrdiff_t* start,
    ptrdiff_t* i,
    double target,
    bool recursive = false)
{
    using R = decltype(get_chunk(size_t{}));

    if (target > data[size - 1]) {

        double delta = data[size - 1] - data[0];
        ptrdiff_t n = size;
        generator.jump_to(*start + size);
        double back = data[size - 1];

        double j = (target - data[size - 1]) / delta - (double)(param.margin) / (double)(n);

        if (j > 1) {
            ptrdiff_t m = static_cast<ptrdiff_t>((j - 1) * static_cast<double>(n));
            back += get_sum(static_cast<size_t>(m));
            generator.drawn(m);
            *start += m + size;
            R extra = get_chunk(static_cast<size_t>(n));
            generator.drawn(n);
            extra.front() += back;
            std::partial_sum(extra.begin(), extra.end(), data);
            return align(generator, get_chunk, get_sum, param, data, size, start, i, target, true);
        }

        next(generator, get_chunk, 1 + param.margin, data, size, start);
        return align(generator, get_chunk, get_sum, param, data, size, start, i, target, true);
    }
    else if (target < data[0]) {
        prev(generator, get_chunk, 0, data, size, start);
        return align(generator, get_chunk, get_sum, param, data, size, start, i, target, true);
    }
    else {
        if (recursive || *i >= size) {
            *i = std::lower_bound(data, data + size, target) - data - 1;
        }
        else {
            *i = iterator::lower_bound(data, data + size, target, *i);
        }
        if (*i == param.margin) {
            return;
        }
        if (!recursive && param.buffer > 0 && *i >= param.buffer && *i + param.buffer < size) {
            return;
        }
        if (*i < param.margin) {
            if (!param.strict && *i >= param.min_margin) {
                return;
            }
            prev(generator, get_chunk, 0, data, size, start);
            return align(generator, get_chunk, get_sum, param, data, size, start, i, target, true);
        }

        generator.jump_to(*start + size);
        ptrdiff_t n = *i - param.margin;
        R extra = get_chunk(static_cast<size_t>(n));
        generator.drawn(n);
        *start += n;
        *i -= n;
        extra.front() += data[size - 1];
        std::partial_sum(extra.begin(), extra.end(), extra.begin());
        std::copy(data + n, data + size, data);
        std::copy(extra.begin(), extra.end(), data + size - n);
    }
}

} // namespace detail

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
     * \f$
     *      \Phi(x) = \frac{1}{2} \left[
     *          1 + \mathrm{erf}\left( \frac{x - \mu}{\sigma \sqrt{2}} \right)
     *      \right]
     * \f$
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
     * For a given probability \f$ p \f$ the output is
     *
     * \f$ x = \mu + \sigma \sqrt{2} \mathrm{erf}^{-1} (2p - 1) \f$
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
 * Only available when compiled with PRRNG_USE_BOOST.
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
     * @brief Result of the cumulative sum of `n` 'random' numbers, distributed according to a
     * delta distribution,
     * @param n Number of steps.
     * @param scale Scale.
     * @return Cumulative sum.
     */
    double cumsum_delta(size_t n, double scale = 1)
    {
        return static_cast<double>(n) * scale;
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
     * Only available when compiled with PRRNG_USE_BOOST.
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

protected:
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

    uint64_t m_initstate; ///< State initiator
    uint64_t m_initseq; ///< Sequence initiator
    uint64_t m_state; ///< RNG state. All values are possible.
    uint64_t m_inc; ///< Controls which RNG sequence (stream) is selected. Must *always* be odd.
};

/**
 * @brief Overload of prrng::pcg32() that keeps track of the current index of the generator
 * in the sequence.
 * Warning: the user is responsible for updating the index.
 * The purpose of this class is therefore mostly internal, to support prrng::pcg32_cumsum().
 */
class pcg32_index : public pcg32 {
private:
    ptrdiff_t m_index; ///< Index of the generator
    bool m_delta; ///< Signal if uniquely a delta distribution will be drawn

public:
    /**
     * @param initstate State initiator.
     * @param initseq Sequence initiator.
     * @param delta `true` if uniquely a delta distribution will be drawn.
     */
    template <typename T = uint64_t, typename S = uint64_t>
    pcg32_index(
        T initstate = PRRNG_PCG32_INITSTATE,
        S initseq = PRRNG_PCG32_INITSEQ,
        bool delta = false)
    {
        static_assert(sizeof(uint64_t) >= sizeof(T), "Down-casting not allowed.");
        static_assert(sizeof(uint64_t) >= sizeof(S), "Down-casting not allowed.");
        this->seed(static_cast<uint64_t>(initstate), static_cast<uint64_t>(initseq));
        m_index = 0;
        m_delta = delta;
    }

    /**
     * @brief State at a specific index of the sequence.
     * Internally the generator is moved to the index, the state is stored,
     * and the generator is restored to its original state.
     *
     * @param index Index at which to get the state.
     * @return uint64_t
     */
    uint64_t state_at(ptrdiff_t index)
    {
        if (m_delta) {
            return m_state;
        }
        uint64_t state = m_state;
        this->advance(index - m_index);
        uint64_t ret = m_state;
        this->restore(state);
        return ret;
    }

    /**
     * @brief Move to a certain index.
     * @param index Index of the generator.
     */
    void jump_to(ptrdiff_t index)
    {
        if (m_delta) {
            return;
        }
        this->advance(index - m_index);
        m_index = index;
    }

    /**
     * @brief Update the generator index with the number of items you have drawn.
     * @param n Number of drawn numbers.
     */
    void drawn(ptrdiff_t n)
    {
        if (m_delta) {
            return;
        }
        m_index += n;
    }

    /**
     * @brief Signal if the generator is uniquely used to draw a delta distribution.
     * @param delta `true` if uniquely a delta distribution will be drawn.
     */
    void set_delta(bool delta)
    {
        m_delta = delta;
    }

    /**
     * @brief Get the generator index.
     * @return Index of the generator.
     */
    ptrdiff_t index() const
    {
        return m_index;
    }

    /**
     * @brief Overwrite the generator index.
     * @param index Index of the generator.
     */
    void set_index(ptrdiff_t index)
    {
        m_index = index;
    }
};

/**
 * @brief Structure to assemble the alignment parameters, see prrng::alignment().
 */
struct alignment {
    /**
     * @param buffer
     *      If positive, only change the chunk if target is in `chunk[:buffer]` or `chunk[-buffer:]`
     *
     * @param margin
     *      Index of the chunk to place the target.
     *
     * @param min_margin
     *      Minimal index to accept if `strict = false`.
     *
     * @param strict
     *      If `true`, `margin` is respected strictly: `argmin(target > chunk) == margin`.
     *      If `false` `min_margin <= argmin(target > chunk) <= margin`, whereby
     *      `argmin(target > chunk) < margin` if moving backwards is required to respect `margin`.
     *
     * @return List with parameters.
     */
    alignment(
        ptrdiff_t buffer = 0,
        ptrdiff_t margin = 0,
        ptrdiff_t min_margin = 0,
        bool strict = false)
    {
        this->buffer = buffer;
        this->margin = margin;
        this->min_margin = min_margin;
        this->strict = strict;
    }

    ptrdiff_t buffer = 0; ///< Buffer zone in which to check for alignment.
    ptrdiff_t margin = 0; ///< Position of the target.
    ptrdiff_t min_margin = 0; ///< Minimum position of the target if `strict = false`.
    bool strict = false; ///< Enforce `margin` strictly.
};

/**
 * @brief Generator of a random cumulative sum of which a chunk is kept in memory.
 * The random number generated by the pcg32 algorithm.
 *
 * @tparam Storage of the data.
 */
template <class Data>
class pcg32_cumsum {
private:
    Data m_data; ///< The chunk.
    pcg32_index m_gen; ///< The generator.
    std::function<Data(size_t)> m_draw; ///< Function to draw the random numbers.
    std::function<double(size_t)> m_sum; ///< Function to get the cumsum of random numbers.
    bool m_extendible; ///< Signal if the drawing functions are specified.
    alignment m_align; ///< alignment settings, see prrng::alignment().
    distribution m_distro; ///< Distribution name, see prrng::distribution().
    std::array<double, 3> m_param; ///< Distribution parameters.
    ptrdiff_t m_start; ///< Start index of the chunk.
    ptrdiff_t m_i; ///< Last know index of `target` in align.

    /**
     * @brief Set draw function.
     */
    void auto_functions()
    {
        m_extendible = true;

        switch (m_distro) {
        case random:
            m_draw = [this](size_t n) -> Data {
                return m_gen.random<Data>({n}) * m_param[0] + m_param[1];
            };
            m_sum = [this](size_t n) -> double {
                return m_gen.cumsum_random(n) * m_param[0] + static_cast<double>(n) * m_param[1];
            };
            return;
        case exponential:
            m_draw = [this](size_t n) -> Data {
                return m_gen.exponential<Data>({n}, m_param[0]) + m_param[1];
            };
            m_sum = [this](size_t n) -> double {
                return m_gen.cumsum_exponential(n, m_param[0]) +
                       static_cast<double>(n) * m_param[1];
            };
            return;
        case delta:
            m_draw = [this](size_t n) -> Data {
                return m_gen.delta<Data>({n}, m_param[0]) + m_param[1];
            };
            m_sum = [this](size_t n) -> double {
                return m_gen.cumsum_delta(n, m_param[0]) + static_cast<double>(n) * m_param[1];
            };
            return;
        case weibull:
            m_draw = [this](size_t n) -> Data {
                return m_gen.weibull<Data>({n}, m_param[0], m_param[1]) + m_param[2];
            };
            m_sum = [this](size_t n) -> double {
                return m_gen.cumsum_weibull(n, m_param[0], m_param[1]) +
                       static_cast<double>(n) * m_param[2];
            };
            return;
        case gamma:
            m_draw = [this](size_t n) -> Data {
                return m_gen.gamma<Data>({n}, m_param[0], m_param[1]) + m_param[2];
            };
            m_sum = [this](size_t n) -> double {
                return m_gen.cumsum_gamma(n, m_param[0], m_param[1]) +
                       static_cast<double>(n) * m_param[2];
            };
            return;
        case normal:
            m_draw = [this](size_t n) -> Data {
                return m_gen.normal<Data>({n}, m_param[0], m_param[1]) + m_param[2];
            };
            m_sum = [this](size_t n) -> double {
                return m_gen.cumsum_normal(n, m_param[0], m_param[1]) +
                       static_cast<double>(n) * m_param[2];
            };
            return;
        case custom:
            m_extendible = false;
            return;
        }
    }

    /**
     * @brief Copy constructor.
     * This function resets all internal pointers.
     *
     * @param other Object to copy.
     */
    void copy_from(const pcg32_cumsum& other)
    {
        m_data = other.m_data;
        m_gen = other.m_gen;
        m_align = other.m_align;
        m_distro = other.m_distro;
        m_param = other.m_param;
        m_start = other.m_start;
        m_i = other.m_i;
        this->auto_functions();
    }

public:
    /**
     * @param shape Shape of the chunk.
     * @param initstate State initiator.
     * @param initseq Sequence initiator.
     * @param distribution Distribution type, see prrng::distribution().
     *
     * @param parameters
     *      Parameters for the distribution. The following is default (and the expected order):
     *
     *      -   prrng::distribution::random: {scale = 1, offset = 0}
     *      -   prrng::distribution::normal: {mu = 1, sigma = , offset = 0}
     *      -   prrng::distribution::weibull: {k = 1, scale = 1, offset = 0}
     *      -   prrng::distribution::gamma: {k = 1, scale = 1, offset = 0}
     *      -   prrng::distribution::exponential: {scale = 1, offset = 0}
     *      -   prrng::distribution::delta: {scale = 1, offset = 0}
     *      -   prrng::distribution::custom: {}
     *
     *      Warning: if you want to use a custom distribution, you have to call
     *      prrng::pcg32_cumsum::set_functions().
     *
     * @param align alignment parameters, see prrng::alignment().
     */
    template <class R, typename T = uint64_t, typename S = uint64_t>
    pcg32_cumsum(
        const R& shape,
        T initstate = PRRNG_PCG32_INITSTATE,
        S initseq = PRRNG_PCG32_INITSEQ,
        enum distribution distribution = distribution::custom,
        const std::vector<double>& parameters = std::vector<double>{},
        const alignment& align = alignment())
    {
        m_data = xt::empty<typename Data::value_type>(shape);
        m_gen = pcg32_index(initstate, initseq, distribution == distribution::delta);
        m_start = m_gen.index();
        m_i = static_cast<ptrdiff_t>(m_data.size());
        m_align = align;
        m_distro = distribution;
        std::copy(parameters.begin(), parameters.end(), m_param.begin());
        this->auto_functions();

        if (!m_extendible) {
            return;
        }

        using E = decltype(m_draw(size_t{}));
        E extra = m_draw(m_data.size());
        m_gen.drawn(m_data.size());
        std::partial_sum(extra.begin(), extra.end(), m_data.begin());
    }

    /**
     * @brief Use external functions to draw the random numbers.
     * @param get_chunk Function to draw the random numbers, called as `get_chunk(n)`.
     * @param get_cumsum Function to get the cumsum of random numbers, called: `get_cumsum(n)`.
     * @param uses_generator Set `true` is the random generator is used by the functions.
     */
    void set_functions(
        std::function<Data(size_t)> get_chunk,
        std::function<double(size_t)> get_cumsum,
        bool uses_generator = true)
    {
        m_extendible = true;
        m_draw = get_chunk;
        m_sum = get_cumsum;
        m_gen.set_delta(!uses_generator);

        using E = decltype(m_draw(size_t{}));
        E extra = m_draw(m_data.size());
        m_gen.drawn(m_data.size());
        std::partial_sum(extra.begin(), extra.end(), m_data.begin());
    }

    /**
     * @copydoc prrng::pcg32_cumsum::copy_from(const prrng::pcg32_cumsum&)
     */
    pcg32_cumsum(const pcg32_cumsum& other)
    {
        this->copy_from(other);
    }

    /**
     * @copydoc prrng::pcg32_cumsum::copy_from(const prrng::pcg32_cumsum&)
     */
    void operator=(const pcg32_cumsum& other)
    {
        this->copy_from(other);
    }

    /**
     * @brief `true` if the chunk is extendible.
     * @return bool
     */
    bool is_extendible() const
    {
        return m_extendible;
    }

    /**
     * @brief Pointer to the generator.
     * @return const pcg32&
     */
    const pcg32_index& generator() const
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
     * @brief Add value(s) to the chunk.
     * @param value Value(s) to add.
     */
    template <class T>
    pcg32_cumsum& operator+=(const T& value)
    {
        xt::noalias(m_data) += value;
        return *this;
    }

    /**
     * @brief Subtract value(s) from the chunk.
     * @param value Value(s) to subtract.
     */
    template <class T>
    pcg32_cumsum& operator-=(const T& value)
    {
        xt::noalias(m_data) -= value;
        return *this;
    }

    /**
     * @brief Chunk.
     * @return Pointer to the chunk.
     */
    const Data& data() const
    {
        return m_data;
    }

    /**
     * @brief Overwrite the chunk.
     * Please check if set_state() or set_start() should be called too.
     *
     * @param data The chunk.
     */
    void set_data(const Data& data)
    {
        m_data = data;
    }

    /**
     * @brief Global index of the first element in the chunk.
     * @return Global index.
     */
    ptrdiff_t start() const
    {
        return m_start;
    }

    /**
     * @brief Set global index of the first element in the chunk.
     * @param index Global index.
     */
    void set_start(ptrdiff_t index)
    {
        m_start = index;
    }

    /**
     * @brief Global index of `target` (the last time prrng::pcg32_cumsum::align() was called).
     * @return Global index.
     */
    ptrdiff_t index() const
    {
        return m_start + m_i;
    }

    /**
     * @brief Index of `target` relative to the beginning of the chunk
     * (the last time prrng::pcg32_cumsum::align() was called).
     *
     * @return Local index.
     */
    ptrdiff_t chunk_index() const
    {
        return m_i;
    }

    /**
     * @copydoc prrng::pcg32_index::state()
     */
    uint64_t state_at(ptrdiff_t index)
    {
        return m_gen.state_at(index);
    }

    /**
     * @brief Restore a specific state in the cumulative sum.
     *
     * @param state The state at the beginning of the new chunk.
     * @param value The value of the first entry of the new chunk.
     * @param index The index of the first entry of the new chunk.
     */
    void restore(uint64_t state, double value, ptrdiff_t index)
    {
        m_gen.set_index(index);
        m_gen.restore(state);

        using E = decltype(m_draw(size_t{}));
        E extra = m_draw(m_data.size());
        m_gen.drawn(m_data.size());
        extra.front() += value - extra.front();
        std::partial_sum(extra.begin(), extra.end(), m_data.begin());
    }

    /**
     * @brief Check if the chunk contains a target.
     * @param target The target.
     * @return `true` if the chunk contains the target.
     */
    bool contains(double target) const
    {
        return target >= m_data.front() && target <= m_data.back();
    }

    /**
     * @brief Shift chunk left.
     * @param margin Overlap to keep with the current chunk.
     */
    void prev(size_t margin = 0)
    {
        PRRNG_ASSERT(m_extendible);
        m_i = static_cast<ptrdiff_t>(m_data.size());
        detail::prev(m_gen, m_draw, margin, m_data.data(), m_data.size(), &m_start);
    }

    /**
     * @brief Shift chunk right.
     * @param margin Overlap to keep with the current chunk.
     */
    void next(size_t margin = 0)
    {
        PRRNG_ASSERT(m_extendible);
        m_i = static_cast<ptrdiff_t>(m_data.size());
        detail::next(m_gen, m_draw, margin, m_data.data(), m_data.size(), &m_start);
    }

    /**
     * @brief Align the chunk to encompass a target value.
     * @param target Target value.
     */
    void align(double target)
    {
        if (!m_extendible) {
            PRRNG_ASSERT(target >= m_data.front() && target <= m_data.back());
            m_i = iterator::lower_bound(m_data.begin(), m_data.end(), target, m_i);
            return;
        }

        detail::align(
            m_gen, m_draw, m_sum, m_align, m_data.data(), m_data.size(), &m_start, &m_i, target);
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
     * distributed according to a delta distribution,
     * @param n Number of steps.
     * @param scale Scale.
     * @return Cumulative sum.
     */
    template <class T>
    auto cumsum_delta(const T& n, double scale = 1) -> typename detail::return_type<double, M>::type
    {
        using R = typename detail::return_type<double, M>::type;
        R ret = R::from_shape(m_shape);
        for (size_t i = 0; i < ret.size(); ++i) {
            ret.flat(i) = static_cast<double>(n.flat(i)) * scale;
        }
        return ret;
    }

    /**
     * @brief Per generator, return the result of the cumulative sum of `n` random numbers,
     * distributed according to a delta distribution,
     * @param n Number of steps.
     * @param scale Scale.
     * @return Cumulative sum.
     */
    template <class R, class T>
    R cumsum_delta(const T& n, double scale = 1)
    {
        R ret = R::from_shape(m_shape);
        for (size_t i = 0; i < ret.size(); ++i) {
            ret.flat(i) = static_cast<double>(n.flat(i)) * scale;
        }
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
        R ret = R::from_shape(detail::concatenate<M, S>::two(m_shape, ishape));
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
template <class Generator, class Shape>
class pcg32_arrayBase : public GeneratorBase_array<Shape> {
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
            m_gen.push_back(Generator(initstate.flat(i)));
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
            m_gen.push_back(Generator(initstate.flat(i), initseq.flat(i)));
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
    Generator& operator()(Args... args)
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
    const Generator& operator()(Args... args) const
    {
        return m_gen[this->get_item(0, 0, args...)];
    }

    /**
     * Return a reference to one generator, using a flat index.
     *
     * @param i Flat index.
     * @return Reference to underlying generator.
     */
    Generator& operator[](size_t i)
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
    const Generator& operator[](size_t i) const
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
    Generator& flat(size_t i)
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
    const Generator& flat(size_t i) const
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
    auto state() -> typename detail::return_type<uint64_t, Shape>::type
    {
        using R = typename detail::return_type<uint64_t, Shape>::type;
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
    auto initstate() -> typename detail::return_type<uint64_t, Shape>::type
    {
        using R = typename detail::return_type<uint64_t, Shape>::type;
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
    auto initseq() -> typename detail::return_type<uint64_t, Shape>::type
    {
        using R = typename detail::return_type<uint64_t, Shape>::type;
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
    auto distance(const T& arg) -> typename detail::return_type<int64_t, Shape>::type
    {
        using R = typename detail::return_type<int64_t, Shape>::type;
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
    std::vector<Generator> m_gen; ///< Underlying storage: one generator per array item
    using GeneratorBase_array<Shape>::m_size;
    using GeneratorBase_array<Shape>::m_shape;
    using GeneratorBase_array<Shape>::m_strides;
};

// todo: remove
// template <class Generator, class Shape>
// class pcg32_index_arrayBase : public pcg32_arrayBase<Generator, Shape>
// {
// protected:
//     using pcg32_arrayBase<Generator, Shape>::m_gen;

// public:
//     template <class R, class I>
//     R state(const I& index)
//     {
//         using value_type = typename R::value_type;
//         R ret = R::from_shape(index.shape());

//         for (size_t i = 0; i < index.size(); ++i) {
//             ret.flat(i) = static_cast<value_type>(m_gen[i].state(index.flat(i)));
//         }

//         return ret;
//     }
// };

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
class pcg32_array : public pcg32_arrayBase<pcg32, std::vector<size_t>> {
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
    using pcg32_arrayBase<pcg32, std::vector<size_t>>::m_gen;
    using GeneratorBase_array<std::vector<size_t>>::m_size;
    using GeneratorBase_array<std::vector<size_t>>::m_shape;
    using GeneratorBase_array<std::vector<size_t>>::m_strides;
};

/**
 * Fixed rank version of pcg32_array
 */
template <size_t N>
class pcg32_tensor : public pcg32_arrayBase<pcg32, std::array<size_t, N>> {
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
    using pcg32_arrayBase<pcg32, std::array<size_t, N>>::m_gen;
    using GeneratorBase_array<std::array<size_t, N>>::m_size;
    using GeneratorBase_array<std::array<size_t, N>>::m_shape;
    using GeneratorBase_array<std::array<size_t, N>>::m_strides;
};

/**
 * @brief Array of prrng::pcg32_index().
 */
class pcg32_index_array : public pcg32_arrayBase<pcg32_index, std::vector<size_t>> {
public:
    pcg32_index_array() = default;

    /**
     * Constructor.
     *
     * @param initstate State initiator for every item.
     * @param initseq Sequence initiator for every item.
     * The shape of these argument determines the shape of the generator array.
     */
    template <class T, class U>
    pcg32_index_array(const T& initstate, const U& initseq)
    {
        m_shape.resize(initstate.dimension());
        m_strides.resize(initstate.dimension());
        this->init(initstate, initseq);
    }

protected:
    using pcg32_arrayBase<pcg32_index, std::vector<size_t>>::m_gen;
    using GeneratorBase_array<std::vector<size_t>>::m_size;
    using GeneratorBase_array<std::vector<size_t>>::m_shape;
    using GeneratorBase_array<std::vector<size_t>>::m_strides;
};

/**
 * Fixed rank version of pcg32_index_array
 */
template <size_t N>
class pcg32_index_tensor : public pcg32_arrayBase<pcg32_index, std::array<size_t, N>> {
public:
    pcg32_index_tensor() = default;

    /**
     * Constructor.
     *
     * @param initstate State initiator for every item.
     * @param initseq Sequence initiator for every item.
     * The shape of these argument determines the shape of the generator array.
     */
    template <class T, class U>
    pcg32_index_tensor(const T& initstate, const U& initseq)
    {
        static_assert(detail::check_fixed_rank<N, T>::value, "Ranks to not match");
        this->init(initstate, initseq);
    }

protected:
    using pcg32_arrayBase<pcg32_index, std::array<size_t, N>>::m_gen;
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
 *
 *      -   All chunks are assembled to one big array.
 *      -   The random number generated by the pcg32 algorithm.
 *
 * @tparam D Storage of the data, e.g. xt::xarray<double>.
 * @tparam G Storage of the generator array, e.g. prrng::pcg32_array
 */
template <class Generator, class Data, class Index>
class pcg32_arrayBase_cumsum {
protected:
    Generator m_gen; ///< Array of generators
    Data m_data; ///< Data container
    std::vector<std::function<xt::xtensor<double, 1>(size_t)>> m_draw; ///< Draw functions
    std::vector<std::function<double(size_t)>> m_sum; ///< Result of cumsum fuctions
    bool m_extendible; ///< Signal if the drawing functions are specified.
    alignment m_align; ///< alignment settings, see prrng::alignment().
    distribution m_distro; ///< Distribution name, see prrng::distribution().
    std::array<double, 3> m_param; ///< Distribution parameters.
    Index m_start; ///< Start index of the chunk.
    Index m_i; ///< Last know index of `target` in align.
    size_t m_n; ///< Size of the chunk.

protected:
    /**
     * @brief Constructor.
     *
     * @param shape Shape of the chunk to keep in memory.
     * @param initstate State initiator for every item.
     * @param initseq Sequence initiator for every item.
     * @param distribution Distribution type, see prrng::distribution().
     *
     * @param parameters
     *      Parameters for the distribution. The following is default (and the expected order):
     *
     *      -   prrng::distribution::random: {scale = 1, offset = 0}
     *      -   prrng::distribution::normal: {mu = 1, sigma = , offset = 0}
     *      -   prrng::distribution::weibull: {k = 1, scale = 1, offset = 0}
     *      -   prrng::distribution::gamma: {k = 1, scale = 1, offset = 0}
     *      -   prrng::distribution::exponential: {scale = 1, offset = 0}
     *      -   prrng::distribution::delta: {scale = 1, offset = 0}
     *      -   prrng::distribution::custom: {}
     *
     * @param align alignment parameters, see prrng::alignment().
     */
    template <class S, class T, class U>
    void init(
        const S& shape,
        const T& initstate,
        const U& initseq,
        enum distribution distribution,
        const std::vector<double>& parameters,
        const alignment& align = alignment())
    {
        PRRNG_ASSERT(xt::same_shape(initstate.shape(), initseq.shape()));
        using shape_type = typename S::value_type;

        m_align = align;
        m_distro = distribution;
        m_gen = Generator(initstate, initseq);

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_gen[i].set_delta(distribution == distribution::delta);
        }

        std::vector<size_t> data_shape;
        data_shape.resize(initstate.dimension() + shape.size());
        std::copy(initstate.shape().begin(), initstate.shape().end(), data_shape.begin());
        std::copy(shape.begin(), shape.end(), data_shape.begin() + initstate.dimension());
        m_data = xt::empty<typename Data::value_type>(data_shape);
        m_n = static_cast<size_t>(
            std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<shape_type>{}));

        m_start = xt::zeros<typename Index::value_type>(m_gen.shape());
        m_i = m_n * xt::ones<typename Index::value_type>(m_gen.shape());

        auto par = detail::default_parameters_cumsum(distribution, parameters);
        std::copy(par.begin(), par.end(), m_param.begin());

        this->auto_functions();

        using E = decltype(m_draw[size_t{}](size_t{}));

        if (m_extendible) {
            for (size_t i = 0; i < m_gen.size(); ++i) {
                E extra = m_draw[i](m_n);
                m_gen[i].drawn(m_n);
                std::partial_sum(extra.begin(), extra.end(), &m_data.flat(i * m_n));
            }
        }
    }

    /**
     * @brief Set draw function.
     */
    void auto_functions()
    {
        using R = decltype(m_draw[size_t{}](size_t{}));
        m_extendible = true;

        if (m_distro != distribution::custom) {
            m_draw.resize(m_gen.size());
            m_sum.resize(m_gen.size());
        }

        switch (m_distro) {
        case random:
            for (size_t i = 0; i < m_gen.size(); ++i) {
                m_draw[i] = [this, i](size_t n) -> R {
                    return m_gen[i].template random<R>({n}) * m_param[0] + m_param[1];
                };
                m_sum[i] = [this, i](size_t n) -> double {
                    return m_gen[i].cumsum_random(n) * m_param[0] +
                           static_cast<double>(n) * m_param[1];
                };
            }
            return;
        case exponential:
            for (size_t i = 0; i < m_gen.size(); ++i) {
                m_draw[i] = [this, i](size_t n) -> R {
                    return m_gen[i].template exponential<R>({n}, m_param[0]) + m_param[1];
                };
                m_sum[i] = [this, i](size_t n) -> double {
                    return m_gen[i].cumsum_exponential(n, m_param[0]) +
                           static_cast<double>(n) * m_param[1];
                };
            }
            return;
        case delta:
            for (size_t i = 0; i < m_gen.size(); ++i) {
                m_draw[i] = [this, i](size_t n) -> R {
                    return m_gen[i].template delta<R>({n}, m_param[0]) + m_param[1];
                };
                m_sum[i] = [this, i](size_t n) -> double {
                    return m_gen[i].cumsum_delta(n, m_param[0]) +
                           static_cast<double>(n) * m_param[1];
                };
            }
            return;
        case weibull:
            for (size_t i = 0; i < m_gen.size(); ++i) {
                m_draw[i] = [this, i](size_t n) -> R {
                    return m_gen[i].template weibull<R>({n}, m_param[0], m_param[1]) + m_param[2];
                };
                m_sum[i] = [this, i](size_t n) -> double {
                    return m_gen[i].cumsum_weibull(n, m_param[0], m_param[1]) +
                           static_cast<double>(n) * m_param[2];
                };
            }
            return;
        case gamma:
            for (size_t i = 0; i < m_gen.size(); ++i) {
                m_draw[i] = [this, i](size_t n) -> R {
                    return m_gen[i].template gamma<R>({n}, m_param[0], m_param[1]) + m_param[2];
                };
                m_sum[i] = [this, i](size_t n) -> double {
                    return m_gen[i].cumsum_gamma(n, m_param[0], m_param[1]) +
                           static_cast<double>(n) * m_param[2];
                };
            }
            return;
        case normal:
            for (size_t i = 0; i < m_gen.size(); ++i) {
                m_draw[i] = [this, i](size_t n) -> R {
                    return m_gen[i].template normal<R>({n}, m_param[0], m_param[1]) + m_param[2];
                };
                m_sum[i] = [this, i](size_t n) -> double {
                    return m_gen[i].cumsum_normal(n, m_param[0], m_param[1]) +
                           static_cast<double>(n) * m_param[2];
                };
            }
            return;
        case custom:
            m_extendible = false;
            return;
        }
    }

    /**
     * @brief Copy constructor.
     * This function resets all internal pointers.
     *
     * @param other Object to copy.
     */
    void copy_from(const pcg32_arrayBase_cumsum& other)
    {
        m_gen = other.m_gen;
        m_data = other.m_data;
        m_align = other.m_align;
        m_distro = other.m_distro;
        m_param = other.m_param;
        m_start = other.m_start;
        m_i = other.m_i;
        m_n = other.m_n;
        this->auto_functions();
    }

public:
    pcg32_arrayBase_cumsum() = default;

    virtual ~pcg32_arrayBase_cumsum() = default;

    /**
     * @copydoc prrng::pcg32_arrayBase_cumsum::copy_from(const prrng::pcg32_arrayBase_cumsum&)
     */
    pcg32_arrayBase_cumsum(const pcg32_arrayBase_cumsum& other)
    {
        this->copy_from(other);
    }

    /**
     * @copydoc prrng::pcg32_arrayBase_cumsum::copy_from(const prrng::pcg32_arrayBase_cumsum&)
     */
    void operator=(const pcg32_arrayBase_cumsum& other)
    {
        this->copy_from(other);
    }

    /**
     * @brief Add values to each chunk.
     * @param values Values to add.
     */
    template <class T>
    pcg32_arrayBase_cumsum& operator+=(const T& values)
    {
        xt::noalias(m_data) += values;
        return *this;
    }

    /**
     * @brief Subtract values from each chunk.
     * @param values Values to subtract.
     */
    template <class T>
    pcg32_arrayBase_cumsum& operator-=(const T& values)
    {
        xt::noalias(m_data) -= values;
        return *this;
    }

    /**
     * @brief `true` if the chunk is extendible.
     * @return bool
     */
    bool is_extendible() const
    {
        return m_extendible;
    }

    /**
     * @copydoc prrng::pcg32_cumsum::align(double)
     */
    template <class T>
    void align(const T& target)
    {
        PRRNG_ASSERT(m_extendible);
        PRRNG_ASSERT(xt::same_shape(target.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            detail::align(
                m_gen[i],
                m_draw[i],
                m_sum[i],
                m_align,
                &m_data.flat(i * m_n),
                m_n,
                &m_start.flat(i),
                &m_i.flat(i),
                target.flat(i));
        }
    }

    /**
     * @copydoc prrng::pcg32_cumsum::align(double)
     * @param i Flat index of the item to align.
     */
    void align(size_t i, double target)
    {
        PRRNG_ASSERT(m_extendible);

        detail::align(
            m_gen[i],
            m_draw[i],
            m_sum[i],
            m_align,
            &m_data.flat(i * m_n),
            m_n,
            &m_start.flat(i),
            &m_i.flat(i),
            target);
    }

    // todo: overload with array index

    /**
     * @brief Generator array.
     * @return Pointer to generator array.
     */
    const Generator& generators() const
    {
        return m_gen;
    }

    /**
     * @brief Return a reference to one value in the chunk.
     * @param args Index in the chunk.
     * @return Reference to one value.
     */
    template <class... Args>
    typename Data::value_type& operator()(Args... args)
    {
        return m_data(args...);
    }

    /**
     * @brief Return a reference to one value in the chunk.
     * @param args Index in the chunk.
     * @return Reference to one value.
     */
    template <class... Args>
    const typename Data::value_type& operator()(Args... args) const
    {
        return m_data(args...);
    }

    /**
     * @copydoc prrng::pcg32_cumsum::data()
     */
    const Data& data() const
    {
        return m_data;
    }

    /**
     * @copydoc prrng::pcg32_cumsum::set_data(const Data&)
     */
    void set_data(const Data& data)
    {
        static_assert(
            std::is_same<typename Data::value_type, double>::value, "Data must be double");
        PRRNG_ASSERT(xt::has_shape(data, m_data.shape()));
        std::copy(data.cbegin(), data.cend(), m_data.begin());
    }

    /**
     * @copydoc prrng::pcg32_cumsum::start()
     */
    const Index& start() const
    {
        return m_start;
    }

    /**
     * @copydoc prrng::pcg32_cumsum::set_start(ptrdiff_t)
     */
    void set_start(const Index& index)
    {
        PRRNG_ASSERT(xt::same_shape(index.shape(), m_gen.shape()));
        m_start = index;
    }

    /**
     * @copydoc prrng::pcg32_cumsum::index()
     */
    Index index() const
    {
        return m_start + m_i;
    }

    /**
     * @copydoc prrng::pcg32_cumsum::chunk_index()
     */
    const Index& chunk_index() const
    {
        return m_i;
    }

    /**
     * @copydoc prrng::pcg32_cumsum::state_at(ptrdiff_t)
     */
    template <class R, class T>
    R state_at(const T& index)
    {
        PRRNG_ASSERT(xt::same_shape(index.shape(), m_gen.shape()));

        using value_type = typename R::value_type;
        R ret = R::from_shape(m_gen.shape());

        for (size_t i = 0; i < m_gen.size(); ++i) {
            ret.flat(i) = static_cast<value_type>(m_gen[i].state_at(index.flat(i)));
        }

        return ret;
    }

    /**
     * @copydoc prrng::pcg32_cumsum::restore(uint64_t, double, ptrdiff_t)
     */
    template <class S, class V, class T>
    void restore(const S& state, const V& value, const T& index)
    {
        PRRNG_ASSERT(xt::same_shape(state.shape(), m_gen.shape()));
        PRRNG_ASSERT(xt::same_shape(value.shape(), m_gen.shape()));
        PRRNG_ASSERT(xt::same_shape(index.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            m_gen[i].set_index(index.flat(i));
            m_gen[i].restore(state.flat(i));

            using E = decltype(m_draw[i](size_t{}));
            E extra = m_draw[i](m_n);
            m_gen[i].drawn(m_n);
            extra.front() += value.flat(i) - extra.front();
            std::partial_sum(extra.begin(), extra.end(), &m_data.flat(i * m_n));
        }
    }

    /**
     * @copydoc prrng::pcg32_cumsum::contains(double)
     */
    template <class T>
    bool contains(const T& target) const
    {
        PRRNG_ASSERT(xt::same_shape(target.shape(), m_gen.shape()));

        for (size_t i = 0; i < m_gen.size(); ++i) {
            if (target.flat(i) < m_data.flat(i * m_n) ||
                target.flat(i) > m_data.flat((i + 1) * m_n - 1)) {
                return false;
            }
        }

        return true;
    }
};

/**
 * @brief Array of generators of a random cumulative sum.
 * A chunk is kept in memory for each generator, whereby all chunks are assembled to one big array.
 * The random number generated by the pcg32 algorithm.
 *
 * @tparam D Storage of the data, e.g. xt::xarray<double>.
 */
template <class Data, class Index>
class pcg32_array_cumsum : public pcg32_arrayBase_cumsum<pcg32_index_array, Data, Index> {
public:
    pcg32_array_cumsum() = default;

    /**
     * @copydoc prrng::pcg32_arrayBase_cumsum::init(const S&, const T&, const U&,
     *      enum distribution, const std::vector<double>&, const alignment&)
     */
    template <class S, class T, class U>
    pcg32_array_cumsum(
        const S& shape,
        const T& initstate,
        const U& initseq,
        enum distribution distribution,
        const std::vector<double>& parameters,
        const alignment& align = alignment())
    {
        this->init(shape, initstate, initseq, distribution, parameters, align);
    }
};

/**
 * @brief Array of generators of a random cumulative sum.
 * A chunk is kept in memory for each generator, whereby all chunks are assembled to one big array.
 * The random number generated by the pcg32 algorithm.
 *
 * @tparam Data Storage of the data, e.g. xt::tensor<double, N + n>.
 * @tparam N Rank of the array of generators, e.g. xt::tensor<double, N>.
 */
template <class Data, class Index, size_t N>
class pcg32_tensor_cumsum : public pcg32_arrayBase_cumsum<pcg32_index_tensor<N>, Data, Index> {
public:
    pcg32_tensor_cumsum() = default;

    /**
     * @copydoc prrng::pcg32_arrayBase_cumsum::init(const S&, const T&, const U&,
     *      enum distribution, const std::vector<double>&, const alignment&)
     */
    template <class S, class T, class U>
    pcg32_tensor_cumsum(
        const S& shape,
        const T& initstate,
        const U& initseq,
        enum distribution distribution,
        const std::vector<double>& parameters,
        const alignment& align = alignment())
    {
        this->init(shape, initstate, initseq, distribution, parameters, align);
    }
};

} // namespace prrng

#endif
