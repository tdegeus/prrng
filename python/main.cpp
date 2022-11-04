#include <prrng.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;

/**
 * Overrides the `__name__` of a module.
 * Classes defined by pybind11 use the `__name__` of the module as of the time they are defined,
 * which affects the `__repr__` of the class type objects.
 */
class ScopedModuleNameOverride {
public:
    explicit ScopedModuleNameOverride(py::module m, std::string name) : module_(std::move(m))
    {
        original_name_ = module_.attr("__name__");
        module_.attr("__name__") = name;
    }
    ~ScopedModuleNameOverride()
    {
        module_.attr("__name__") = original_name_;
    }

private:
    py::module module_;
    py::object original_name_;
};

PYBIND11_MODULE(_prrng, m)
{
    // Ensure members to display as `prrng.X` rather than `prrng._prrng.X`
    ScopedModuleNameOverride name_override(m, "prrng");

    xt::import_numpy();

    m.doc() = "Portable Reconstructible Random Number Generator";

    m.def(
        "version",
        &prrng::version,
        "Return version string. "
        "See :cpp:func:`prrng::version`.");

    py::class_<prrng::normal_distribution>(m, "normal_distribution")

        .def(
            py::init<double, double>(),
            "Normal distribution. "
            "See :cpp:class:`prrng::normal_distribution`.",
            py::arg("mu") = 0.0,
            py::arg("sigma") = 1.0)

        .def(
            "pdf",
            &prrng::normal_distribution::pdf<xt::pytensor<double, 1>>,
            "Probability density distribution. "
            "See :cpp:func:`prrng::normal_distribution::pdf`.",
            py::arg("x"))

        .def(
            "cdf",
            &prrng::normal_distribution::cdf<xt::pytensor<double, 1>>,
            "Cumulative density distribution. "
            "See :cpp:func:`prrng::normal_distribution::cdf`.",
            py::arg("x"))

        .def(
            "quantile",
            &prrng::normal_distribution::quantile<xt::pyarray<double>>,
            "Quantile (inverse of cumulative density distribution). "
            "See :cpp:func:`prrng::normal_distribution::quantile`.",
            py::arg("r"))

        .def("__repr__", [](const prrng::normal_distribution&) {
            return "<prrng.normal_distribution>";
        });

    py::class_<prrng::exponential_distribution>(m, "exponential_distribution")

        .def(
            py::init<double>(),
            "exponential distribution. "
            "See :cpp:class:`prrng::exponential_distribution`.",
            py::arg("scale") = 1.0)

        .def(
            "pdf",
            &prrng::exponential_distribution::pdf<xt::pytensor<double, 1>>,
            "Probability density distribution. "
            "See :cpp:func:`prrng::exponential_distribution::pdf`.",
            py::arg("x"))

        .def(
            "cdf",
            &prrng::exponential_distribution::cdf<xt::pytensor<double, 1>>,
            "Cumulative density distribution. "
            "See :cpp:func:`prrng::exponential_distribution::cdf`.",
            py::arg("x"))

        .def(
            "quantile",
            &prrng::exponential_distribution::quantile<xt::pyarray<double>>,
            "Quantile (inverse of cumulative density distribution). "
            "See :cpp:func:`prrng::exponential_distribution::quantile`.",
            py::arg("r"))

        .def("__repr__", [](const prrng::exponential_distribution&) {
            return "<prrng.exponential_distribution>";
        });

    py::class_<prrng::weibull_distribution>(m, "weibull_distribution")

        .def(
            py::init<double, double>(),
            "Weibull distribution. "
            "See :cpp:class:`prrng::weibull_distribution`.",
            py::arg("k") = 1.0,
            py::arg("scale") = 1.0)

        .def(
            "pdf",
            &prrng::weibull_distribution::pdf<xt::pytensor<double, 1>>,
            "Probability density distribution. "
            "See :cpp:func:`prrng::weibull_distribution::pdf`.",
            py::arg("x"))

        .def(
            "cdf",
            &prrng::weibull_distribution::cdf<xt::pytensor<double, 1>>,
            "Cumulative density distribution. "
            "See :cpp:func:`prrng::weibull_distribution::cdf`.",
            py::arg("x"))

        .def(
            "quantile",
            &prrng::weibull_distribution::quantile<xt::pyarray<double>>,
            "Quantile (inverse of cumulative density distribution). "
            "See :cpp:func:`prrng::weibull_distribution::quantile`.",
            py::arg("r"))

        .def("__repr__", [](const prrng::weibull_distribution&) {
            return "<prrng.weibull_distribution>";
        });

    py::class_<prrng::gamma_distribution>(m, "gamma_distribution")

        .def(
            py::init<double, double>(),
            "Weibull distribution. "
            "See :cpp:class:`prrng::gamma_distribution`.",
            py::arg("k") = 1.0,
            py::arg("theta") = 1.0)

        .def(
            "pdf",
            &prrng::gamma_distribution::pdf<xt::pytensor<double, 1>>,
            "Probability density distribution. "
            "See :cpp:func:`prrng::gamma_distribution::pdf`.",
            py::arg("x"))

        .def(
            "cdf",
            &prrng::gamma_distribution::cdf<xt::pytensor<double, 1>>,
            "Cumulative density distribution. "
            "See :cpp:func:`prrng::gamma_distribution::cdf`.",
            py::arg("x"))

        .def(
            "quantile",
            &prrng::gamma_distribution::quantile<xt::pyarray<double>>,
            "Quantile (inverse of cumulative density distribution). "
            "See :cpp:func:`prrng::gamma_distribution::quantile`.",
            py::arg("r"))

        .def("__repr__", [](const prrng::gamma_distribution&) {
            return "<prrng.gamma_distribution>";
        });

    py::class_<prrng::GeneratorBase>(m, "GeneratorBase")

        .def(
            py::init<>(),
            "Random number generator base class. "
            "See :cpp:class:`prrng::GeneratorBase`.")

        .def(
            "cumsum_random",
            &prrng::GeneratorBase::cumsum_random,
            "The result of the cumsum of `n` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::cumsum_random`.",
            py::arg("n"))

        .def(
            "cumsum_normal",
            &prrng::GeneratorBase::cumsum_normal,
            "The result of the cumsum of `n` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::cumsum_normal`.",
            py::arg("n"),
            py::arg("mu") = 0.0,
            py::arg("sigma") = 1.0)

        .def(
            "cumsum_exponential",
            &prrng::GeneratorBase::cumsum_exponential,
            "The result of the cumsum of `n` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::cumsum_exponential`.",
            py::arg("n"),
            py::arg("scale") = 1.0)

        .def(
            "cumsum_weibull",
            &prrng::GeneratorBase::cumsum_weibull,
            "The result of the cumsum of `n` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::cumsum_weibull`.",
            py::arg("n"),
            py::arg("k") = 1.0,
            py::arg("scale") = 1.0)

        .def(
            "cumsum_gamma",
            &prrng::GeneratorBase::cumsum_gamma,
            "The result of the cumsum of `n` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::cumsum_gamma`.",
            py::arg("n"),
            py::arg("k") = 1.0,
            py::arg("theta") = 1.0)

        .def(
            "decide",
            py::overload_cast<const xt::pyarray<double>&>(
                &prrng::GeneratorBase::decide<xt::pyarray<double>, xt::pyarray<bool>>),
            "ndarray of decision. "
            "See :cpp:func:`prrng::GeneratorBase::decide`.",
            py::arg("p"))

        .def(
            "decide",
            py::overload_cast<const xt::pyarray<double>&, xt::pyarray<bool>&>(
                &prrng::GeneratorBase::decide<xt::pyarray<double>, xt::pyarray<bool>>),
            "ndarray of decision. "
            "See :cpp:func:`prrng::GeneratorBase::decide`.",
            py::arg("p"),
            py::arg("ret"))

        .def(
            "decide_masked",
            py::overload_cast<const xt::pyarray<double>&, const xt::pyarray<bool>&>(
                &prrng::GeneratorBase::
                    decide_masked<xt::pyarray<double>, xt::pyarray<bool>, xt::pyarray<bool>>),
            "ndarray of decision. "
            "See :cpp:func:`prrng::GeneratorBase::decide_masked`.",
            py::arg("p"),
            py::arg("mask"))

        .def(
            "decide_masked",
            py::overload_cast<
                const xt::pyarray<double>&,
                const xt::pyarray<bool>&,
                xt::pyarray<bool>&>(
                &prrng::GeneratorBase::
                    decide_masked<xt::pyarray<double>, xt::pyarray<bool>, xt::pyarray<bool>>),
            "ndarray of decision. "
            "See :cpp:func:`prrng::GeneratorBase::decide_masked`.",
            py::arg("p"),
            py::arg("mak"),
            py::arg("ret"))

        .def(
            "random",
            py::overload_cast<const std::vector<size_t>&>(
                &prrng::GeneratorBase::random<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::random`.",
            py::arg("shape"))

        .def(
            "randint",
            py::overload_cast<const std::vector<size_t>&, uint32_t>(
                &prrng::GeneratorBase::
                    randint<xt::pyarray<uint32_t>, std::vector<size_t>, uint32_t>),
            "ndarray of random integers. "
            "See :cpp:func:`prrng::GeneratorBase::randint`.",
            py::arg("shape"),
            py::arg("high"))

        .def(
            "randint",
            py::overload_cast<const std::vector<size_t>&, int32_t, int32_t>(
                &prrng::GeneratorBase::
                    randint<xt::pyarray<int32_t>, std::vector<size_t>, int32_t, int32_t>),
            "ndarray of random integers. "
            "See :cpp:func:`prrng::GeneratorBase::randint`.",
            py::arg("shape"),
            py::arg("low"),
            py::arg("high"))

        .def(
            "normal",
            py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase::normal<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a normal distribution. "
            "See :cpp:func:`prrng::GeneratorBase::normal`.",
            py::arg("shape"),
            py::arg("mu") = 0.0,
            py::arg("sigma") = 1.0)

        .def(
            "exponential",
            py::overload_cast<const std::vector<size_t>&, double>(
                &prrng::GeneratorBase::exponential<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a exponential distribution. "
            "See :cpp:func:`prrng::GeneratorBase::exponential`.",
            py::arg("shape"),
            py::arg("scale") = 1.0)

        .def(
            "weibull",
            py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase::weibull<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a weibull distribution. "
            "See :cpp:func:`prrng::GeneratorBase::weibull`.",
            py::arg("shape"),
            py::arg("k") = 1.0,
            py::arg("scale") = 1.0)

        .def(
            "gamma",
            py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase::gamma<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a gamma distribution. "
            "See :cpp:func:`prrng::GeneratorBase::gamma`.",
            py::arg("shape"),
            py::arg("k") = 1.0,
            py::arg("theta") = 1.0)

        .def(
            "delta",
            py::overload_cast<const std::vector<size_t>&, double>(
                &prrng::GeneratorBase::delta<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray equal to mean. This is not a random distribution!."
            "See :cpp:func:`prrng::GeneratorBase::delta`.",
            py::arg("shape"),
            py::arg("mean") = 1.0)

        .def("__repr__", [](const prrng::GeneratorBase&) { return "<prrng.GeneratorBase>"; });

    py::class_<prrng::pcg32, prrng::GeneratorBase>(m, "pcg32")

        .def(
            py::init<uint64_t, uint64_t>(),
            "Random number generator. "
            "See :cpp:class:`prrng::pcg32`.",
            py::arg("initstate") = PRRNG_PCG32_INITSTATE,
            py::arg("initseq") = PRRNG_PCG32_INITSEQ)

        .def(py::self - py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)

        .def(
            "state",
            &prrng::pcg32::state<uint64_t>,
            "Current state. "
            "See :cpp:func:`prrng::pcg32::state`.")

        .def(
            "initstate",
            &prrng::pcg32::initstate<uint64_t>,
            "``initstate`` used in constructor. "
            "See :cpp:func:`prrng::pcg32::initstate`.")

        .def(
            "initseq",
            &prrng::pcg32::initseq<uint64_t>,
            "``initseq`` used in constructor. "
            "See :cpp:func:`prrng::pcg32::initseq`.")

        .def(
            "restore",
            &prrng::pcg32::restore<uint64_t>,
            "Restore state. "
            "See :cpp:func:`prrng::pcg32::restore`.",
            py::arg("state"))

        .def(
            "distance",
            py::overload_cast<uint64_t>(&prrng::pcg32::distance<int64_t, uint64_t>, py::const_),
            "Distance to another state. "
            "See :cpp:func:`prrng::pcg32::distance`.",
            py::arg("state"))

        .def(
            "distance",
            py::overload_cast<const prrng::pcg32&>(&prrng::pcg32::distance<int64_t>, py::const_),
            "Distance to another state. "
            "See :cpp:func:`prrng::pcg32::distance`.",
            py::arg("generator"))

        .def(
            "advance",
            &prrng::pcg32::advance<int64_t>,
            "Advance by a distance. "
            "See :cpp:func:`prrng::pcg32::advance`.",
            py::arg("distance"))

        .def(
            "random",
            py::overload_cast<const std::vector<size_t>&>(
                &prrng::pcg32::random<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers. "
            "See :cpp:func:`prrng::pcg32::random`.",
            py::arg("shape"))

        .def(
            "randint",
            py::overload_cast<const std::vector<size_t>&, uint32_t>(
                &prrng::pcg32::randint<xt::pyarray<uint32_t>, std::vector<size_t>, uint32_t>),
            "ndarray of random integers. "
            "See :cpp:func:`prrng::pcg32::randint`.",
            py::arg("shape"),
            py::arg("high"))

        .def(
            "randint",
            py::overload_cast<const std::vector<size_t>&, int32_t, int32_t>(
                &prrng::pcg32::
                    randint<xt::pyarray<int32_t>, std::vector<size_t>, int32_t, int32_t>),
            "ndarray of random integers. "
            "See :cpp:func:`prrng::pcg32::randint`.",
            py::arg("shape"),
            py::arg("low"),
            py::arg("high"))

        .def("__repr__", [](const prrng::pcg32&) { return "<prrng.pcg32>"; });

    py::class_<prrng::GeneratorBase_array<std::vector<size_t>>>(m, "GeneratorBase_array")

        .def(
            py::init<>(),
            "Random number generator base class. "
            "See :cpp:class:`prrng::GeneratorBase_array`.")

        .def(
            "shape",
            [](const prrng::GeneratorBase_array<std::vector<size_t>>& s) { return s.shape(); },
            "Shape of the array of generators. "
            "See :cpp:func:`prrng::GeneratorBase_array::shape`.")

        .def(
            "shape",
            py::overload_cast<size_t>(
                &prrng::GeneratorBase_array<std::vector<size_t>>::shape<size_t>, py::const_),
            "Shape of the array of generators, along a certain axis. "
            "See :cpp:func:`prrng::GeneratorBase_array::shape`.",
            py::arg("axis"))

        .def(
            "size",
            &prrng::GeneratorBase_array<std::vector<size_t>>::size,
            "Size of the array of generators. "
            "See :cpp:func:`prrng::GeneratorBase_array::size`.")

        .def(
            "decide",
            py::overload_cast<const xt::pyarray<double>&>(
                &prrng::GeneratorBase_array<
                    std::vector<size_t>>::decide<xt::pyarray<double>, xt::pyarray<bool>>),
            "ndarray of decision. "
            "See :cpp:func:`prrng::GeneratorBase_array::decide`.",
            py::arg("p"))

        .def(
            "decide",
            py::overload_cast<const xt::pyarray<double>&, xt::pyarray<bool>&>(
                &prrng::GeneratorBase_array<
                    std::vector<size_t>>::decide<xt::pyarray<double>, xt::pyarray<bool>>),
            "ndarray of decision. "
            "See :cpp:func:`prrng::GeneratorBase_array::decide`.",
            py::arg("p"),
            py::arg("ret"))

        .def(
            "decide_masked",
            py::overload_cast<const xt::pyarray<double>&, const xt::pyarray<bool>&>(
                &prrng::GeneratorBase_array<std::vector<size_t>>::
                    decide_masked<xt::pyarray<double>, xt::pyarray<bool>, xt::pyarray<bool>>),
            "ndarray of decision. "
            "See :cpp:func:`prrng::GeneratorBase_array::decide_masked`.",
            py::arg("p"),
            py::arg("mask"))

        .def(
            "decide_masked",
            py::overload_cast<
                const xt::pyarray<double>&,
                const xt::pyarray<bool>&,
                xt::pyarray<bool>&>(
                &prrng::GeneratorBase_array<std::vector<size_t>>::
                    decide_masked<xt::pyarray<double>, xt::pyarray<bool>, xt::pyarray<bool>>),
            "ndarray of decision. "
            "See :cpp:func:`prrng::GeneratorBase_array::decide_masked`.",
            py::arg("p"),
            py::arg("mask"),
            py::arg("ret"))

        .def(
            "random",
            py::overload_cast<const std::vector<size_t>&>(
                &prrng::GeneratorBase_array<
                    std::vector<size_t>>::random<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers. "
            "See :cpp:func:`prrng::GeneratorBase_array::random`.",
            py::arg("ishape"))

        .def(
            "randint",
            py::overload_cast<const std::vector<size_t>&, uint32_t>(
                &prrng::GeneratorBase_array<std::vector<size_t>>::
                    randint<xt::pyarray<uint32_t>, std::vector<size_t>, uint32_t>),
            "ndarray of random integers. "
            "See :cpp:func:`prrng::GeneratorBase_array::randint`.",
            py::arg("ishape"),
            py::arg("high"))

        .def(
            "randint",
            py::overload_cast<const std::vector<size_t>&, int32_t, int32_t>(
                &prrng::GeneratorBase_array<std::vector<size_t>>::
                    randint<xt::pyarray<int32_t>, std::vector<size_t>, int32_t, int32_t>),
            "ndarray of random integers. "
            "See :cpp:func:`prrng::GeneratorBase_array::randint`.",
            py::arg("ishape"),
            py::arg("low"),
            py::arg("high"))

        .def(
            "normal",
            py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase_array<
                    std::vector<size_t>>::normal<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a normal distribution. "
            "See :cpp:func:`prrng::GeneratorBase_array::normal`.",
            py::arg("ishape"),
            py::arg("mu") = 0.0,
            py::arg("sigma") = 1.0)

        .def(
            "exponential",
            py::overload_cast<const std::vector<size_t>&, double>(
                &prrng::GeneratorBase_array<
                    std::vector<size_t>>::exponential<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a exponential distribution. "
            "See :cpp:func:`prrng::GeneratorBase_array::exponential`.",
            py::arg("ishape"),
            py::arg("scale") = 1.0)

        .def(
            "weibull",
            py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase_array<
                    std::vector<size_t>>::weibull<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a weibull distribution. "
            "See :cpp:func:`prrng::GeneratorBase_array::weibull`.",
            py::arg("ishape"),
            py::arg("k") = 1.0,
            py::arg("scale") = 1.0)

        .def(
            "gamma",
            py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase_array<
                    std::vector<size_t>>::gamma<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a gamma distribution. "
            "See :cpp:func:`prrng::GeneratorBase_array::gamma`.",
            py::arg("ishape"),
            py::arg("k") = 1.0,
            py::arg("theta") = 1.0)

        .def(
            "delta",
            py::overload_cast<const std::vector<size_t>&, double>(
                &prrng::GeneratorBase_array<
                    std::vector<size_t>>::delta<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray equal to mean. This is not a random distribution!. "
            "See :cpp:func:`prrng::GeneratorBase_array::delta`.",
            py::arg("ishape"),
            py::arg("mean") = 1.0)

        .def(
            "cumsum_random",
            &prrng::GeneratorBase_array<
                std::vector<size_t>>::cumsum_random<xt::pyarray<double>, xt::pyarray<size_t>>,
            "Cumsum of ``n`` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase_array::cumsum_random`.",
            py::arg("n"))

        .def(
            "cumsum_normal",
            &prrng::GeneratorBase_array<
                std::vector<size_t>>::cumsum_normal<xt::pyarray<double>, xt::pyarray<size_t>>,
            "Cumsum of ``n`` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase_array::cumsum_normal`.",
            py::arg("n"),
            py::arg("mu") = 0.0,
            py::arg("sigma") = 1.0)

        .def(
            "cumsum_exponential",
            &prrng::GeneratorBase_array<
                std::vector<size_t>>::cumsum_exponential<xt::pyarray<double>, xt::pyarray<size_t>>,
            "Cumsum of ``n`` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase_array::cumsum_exponential`.",
            py::arg("n"),
            py::arg("scale") = 1.0)

        .def(
            "cumsum_weibull",
            &prrng::GeneratorBase_array<
                std::vector<size_t>>::cumsum_weibull<xt::pyarray<double>, xt::pyarray<size_t>>,
            "Cumsum of ``n`` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase_array::cumsum_weibull`.",
            py::arg("n"),
            py::arg("k") = 1.0,
            py::arg("scale") = 1.0)

        .def(
            "cumsum_gamma",
            &prrng::GeneratorBase_array<
                std::vector<size_t>>::cumsum_gamma<xt::pyarray<double>, xt::pyarray<size_t>>,
            "Cumsum of ``n`` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase_array::cumsum_gamma`.",
            py::arg("n"),
            py::arg("k") = 1.0,
            py::arg("theta") = 1.0)

        .def("__repr__", [](const prrng::GeneratorBase_array<std::vector<size_t>>&) {
            return "<prrng.GeneratorBase_array>";
        });

    py::class_<
        prrng::pcg32_arrayBase<std::vector<size_t>>,
        prrng::GeneratorBase_array<std::vector<size_t>>>(m, "pcg32_arrayBase")

        .def(
            py::init<>(),
            "Random number generator base class. "
            "See :cpp:class:`prrng::pcg32_arrayBase`.")

        // https://github.com/pybind/pybind11/blob/master/tests/test_sequences_and_iterators.cpp

        .def(
            "__getitem__",
            [](prrng::pcg32_arrayBase<std::vector<size_t>>& s, size_t i) {
                if (i >= s.size())
                    throw py::index_error();
                return &s[i];
            },
            py::return_value_policy::reference_internal)

        .def(
            "__getitem__",
            [](prrng::pcg32_arrayBase<std::vector<size_t>>& s, std::vector<size_t> index) {
                if (!s.inbounds(index))
                    throw py::index_error();
                return &s[s.flat_index(index)];
            },
            py::return_value_policy::reference_internal)

        .def(
            "state",
            &prrng::pcg32_arrayBase<std::vector<size_t>>::state<xt::pyarray<uint64_t>>,
            "Get current state. "
            "See :cpp:func:`prrng::pcg32_arrayBase::state`.")

        .def(
            "initstate",
            &prrng::pcg32_arrayBase<std::vector<size_t>>::initstate<xt::pyarray<uint64_t>>,
            "``initstate`` used in constructor. "
            "See :cpp:func:`prrng::pcg32_arrayBase::initstate`.")

        .def(
            "initseq",
            &prrng::pcg32_arrayBase<std::vector<size_t>>::initseq<xt::pyarray<uint64_t>>,
            "``initseq`` used in constructor. "
            "See :cpp:func:`prrng::pcg32_arrayBase::initseq`.")

        .def(
            "distance",
            py::overload_cast<const xt::pyarray<uint64_t>&>(
                &prrng::pcg32_arrayBase<std::vector<size_t>>::distance<xt::pyarray<uint64_t>>),
            "Distance to a state. "
            "See :cpp:func:`prrng::pcg32_arrayBase::distance`.",
            py::arg("arg"))

        .def(
            "distance",
            py::overload_cast<const prrng::pcg32_arrayBase<std::vector<size_t>>&>(
                &prrng::pcg32_arrayBase<std::vector<size_t>>::distance<
                    prrng::pcg32_arrayBase<std::vector<size_t>>>),
            "Distance to a state. "
            "See :cpp:func:`prrng::pcg32_arrayBase::distance`.",
            py::arg("arg"))

        .def(
            "advance",
            &prrng::pcg32_arrayBase<std::vector<size_t>>::advance<xt::pyarray<uint64_t>>,
            "Advance generators. "
            "See :cpp:func:`prrng::pcg32_arrayBase::advance`.",
            py::arg("distance"))

        .def(
            "restore",
            &prrng::pcg32_arrayBase<std::vector<size_t>>::restore<xt::pyarray<uint64_t>>,
            "Restore state. "
            "See :cpp:func:`prrng::pcg32_arrayBase::restore`.",
            py::arg("state"))

        .def("__repr__", [](const prrng::pcg32_arrayBase<std::vector<size_t>>&) {
            return "<prrng.pcg32_arrayBase>";
        });

    py::class_<prrng::pcg32_array, prrng::pcg32_arrayBase<std::vector<size_t>>>(m, "pcg32_array")

        .def(
            py::init<xt::pyarray<uint64_t>>(),
            "Random number generator. "
            "See :cpp:class:`prrng::pcg32_array`.",
            py::arg("initstate"))

        .def(
            py::init<xt::pyarray<uint64_t>, xt::pyarray<uint64_t>>(),
            "Random number generator. "
            "See :cpp:class:`prrng::pcg32_array`.",
            py::arg("initstate"),
            py::arg("initseq"))

        .def("__repr__", [](const prrng::pcg32_array&) { return "<prrng.pcg32_array>"; });

} // PYBIND11_MODULE
