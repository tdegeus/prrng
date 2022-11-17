#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/xtensor_python_config.hpp> // todo: remove for xtensor-python >0.26.1

#include <prrng.h>

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

template <class C, class Parent>
void init_GeneratorBase_array(C& cls)
{
    cls.def(
        "shape",
        [](const Parent& s) { return s.shape(); },
        "Shape of the array of generators. "
        "See :cpp:func:`prrng::GeneratorBase_array::shape`.");

    cls.def(
        "shape",
        py::overload_cast<size_t>(&Parent::template shape<size_t>, py::const_),
        "Shape of the array of generators, along a certain axis. "
        "See :cpp:func:`prrng::GeneratorBase_array::shape`.",
        py::arg("axis"));

    cls.def(
        "size",
        &Parent::size,
        "Size of the array of generators. "
        "See :cpp:func:`prrng::GeneratorBase_array::size`.");

    cls.def(
        "decide",
        py::overload_cast<const xt::pyarray<double>&>(
            &Parent::template decide<xt::pyarray<double>, xt::pyarray<bool>>),
        "ndarray of decision. "
        "See :cpp:func:`prrng::GeneratorBase_array::decide`.",
        py::arg("p"));

    cls.def(
        "decide",
        py::overload_cast<const xt::pyarray<double>&, xt::pyarray<bool>&>(
            &Parent::template decide<xt::pyarray<double>, xt::pyarray<bool>>),
        "ndarray of decision. "
        "See :cpp:func:`prrng::GeneratorBase_array::decide`.",
        py::arg("p"),
        py::arg("ret"));

    cls.def(
        "decide_masked",
        py::overload_cast<const xt::pyarray<double>&, const xt::pyarray<bool>&>(
            &Parent::
                template decide_masked<xt::pyarray<double>, xt::pyarray<bool>, xt::pyarray<bool>>),
        "ndarray of decision. "
        "See :cpp:func:`prrng::GeneratorBase_array::decide_masked`.",
        py::arg("p"),
        py::arg("mask"));

    cls.def(
        "decide_masked",
        py::overload_cast<const xt::pyarray<double>&, const xt::pyarray<bool>&, xt::pyarray<bool>&>(
            &Parent::
                template decide_masked<xt::pyarray<double>, xt::pyarray<bool>, xt::pyarray<bool>>),
        "ndarray of decision. "
        "See :cpp:func:`prrng::GeneratorBase_array::decide_masked`.",
        py::arg("p"),
        py::arg("mask"),
        py::arg("ret"));

    cls.def(
        "random",
        py::overload_cast<const std::vector<size_t>&>(
            &Parent::template random<xt::pyarray<double>, std::vector<size_t>>),
        "ndarray of random numbers. "
        "See :cpp:func:`prrng::GeneratorBase_array::random`.",
        py::arg("ishape"));

    cls.def(
        "randint",
        py::overload_cast<const std::vector<size_t>&, uint32_t>(
            &Parent::template randint<xt::pyarray<uint32_t>, std::vector<size_t>, uint32_t>),
        "ndarray of random integers. "
        "See :cpp:func:`prrng::GeneratorBase_array::randint`.",
        py::arg("ishape"),
        py::arg("high"));

    cls.def(
        "randint",
        py::overload_cast<const std::vector<size_t>&, int32_t, int32_t>(
            &Parent::template randint<xt::pyarray<int32_t>, std::vector<size_t>, int32_t, int32_t>),
        "ndarray of random integers. "
        "See :cpp:func:`prrng::GeneratorBase_array::randint`.",
        py::arg("ishape"),
        py::arg("low"),
        py::arg("high"));

    cls.def(
        "normal",
        py::overload_cast<const std::vector<size_t>&, double, double>(
            &Parent::template normal<xt::pyarray<double>, std::vector<size_t>>),
        "ndarray of random numbers, distributed according to a normal distribution. "
        "See :cpp:func:`prrng::GeneratorBase_array::normal`.",
        py::arg("ishape"),
        py::arg("mu") = 0,
        py::arg("sigma") = 1);

    cls.def(
        "exponential",
        py::overload_cast<const std::vector<size_t>&, double>(
            &Parent::template exponential<xt::pyarray<double>, std::vector<size_t>>),
        "ndarray of random numbers, distributed according to a exponential distribution. "
        "See :cpp:func:`prrng::GeneratorBase_array::exponential`.",
        py::arg("ishape"),
        py::arg("scale") = 1);

    cls.def(
        "weibull",
        py::overload_cast<const std::vector<size_t>&, double, double>(
            &Parent::template weibull<xt::pyarray<double>, std::vector<size_t>>),
        "ndarray of random numbers, distributed according to a weibull distribution. "
        "See :cpp:func:`prrng::GeneratorBase_array::weibull`.",
        py::arg("ishape"),
        py::arg("k") = 1,
        py::arg("scale") = 1);

    cls.def(
        "gamma",
        py::overload_cast<const std::vector<size_t>&, double, double>(
            &Parent::template gamma<xt::pyarray<double>, std::vector<size_t>>),
        "ndarray of random numbers, distributed according to a gamma distribution. "
        "See :cpp:func:`prrng::GeneratorBase_array::gamma`.",
        py::arg("ishape"),
        py::arg("k") = 1,
        py::arg("scale") = 1);

    cls.def(
        "delta",
        py::overload_cast<const std::vector<size_t>&, double>(
            &Parent::template delta<xt::pyarray<double>, std::vector<size_t>>),
        "ndarray equal to mean. This is not a random distribution!. "
        "See :cpp:func:`prrng::GeneratorBase_array::delta`.",
        py::arg("ishape"),
        py::arg("mean") = 1.0);

    cls.def(
        "cumsum_random",
        &Parent::template cumsum_random<xt::pyarray<double>, xt::pyarray<size_t>>,
        "Cumsum of ``n`` random numbers. "
        "See :cpp:func:`prrng::GeneratorBase_array::cumsum_random`.",
        py::arg("n"));

    cls.def(
        "cumsum_normal",
        &Parent::template cumsum_normal<xt::pyarray<double>, xt::pyarray<size_t>>,
        "Cumsum of ``n`` random numbers. "
        "See :cpp:func:`prrng::GeneratorBase_array::cumsum_normal`.",
        py::arg("n"),
        py::arg("mu") = 0,
        py::arg("sigma") = 1);

    cls.def(
        "cumsum_exponential",
        &Parent::template cumsum_exponential<xt::pyarray<double>, xt::pyarray<size_t>>,
        "Cumsum of ``n`` random numbers. "
        "See :cpp:func:`prrng::GeneratorBase_array::cumsum_exponential`.",
        py::arg("n"),
        py::arg("scale") = 1);

    cls.def(
        "cumsum_delta",
        &Parent::template cumsum_delta<xt::pyarray<double>, xt::pyarray<size_t>>,
        "Cumsum of ``n`` random numbers. "
        "See :cpp:func:`prrng::GeneratorBase_array::cumsum_delta`.",
        py::arg("n"),
        py::arg("scale") = 1);

    cls.def(
        "cumsum_weibull",
        &Parent::template cumsum_weibull<xt::pyarray<double>, xt::pyarray<size_t>>,
        "Cumsum of ``n`` random numbers. "
        "See :cpp:func:`prrng::GeneratorBase_array::cumsum_weibull`.",
        py::arg("n"),
        py::arg("k") = 1,
        py::arg("scale") = 1);

    cls.def(
        "cumsum_gamma",
        &Parent::template cumsum_gamma<xt::pyarray<double>, xt::pyarray<size_t>>,
        "Cumsum of ``n`` random numbers. "
        "See :cpp:func:`prrng::GeneratorBase_array::cumsum_gamma`.",
        py::arg("n"),
        py::arg("k") = 1,
        py::arg("scale") = 1);
}

template <class C, class Parent>
void init_pcg32_arrayBase(C& cls)
{
    cls.def(
        "__getitem__",
        [](Parent& s, size_t i) {
            if (i >= s.size())
                throw py::index_error();
            return &s[i];
        },
        py::return_value_policy::reference_internal);

    cls.def(
        "__getitem__",
        [](Parent& s, std::vector<size_t> index) {
            if (!s.inbounds(index))
                throw py::index_error();
            return &s[s.flat_index(index)];
        },
        py::return_value_policy::reference_internal);

    cls.def(
        "state",
        &Parent::template state<xt::pyarray<uint64_t>>,
        "Get current state. "
        "See :cpp:func:`prrng::pcg32_arrayBase::state`.");

    cls.def(
        "initstate",
        &Parent::template initstate<xt::pyarray<uint64_t>>,
        "``initstate`` used in constructor. "
        "See :cpp:func:`prrng::pcg32_arrayBase::initstate`.");

    cls.def(
        "initseq",
        &Parent::template initseq<xt::pyarray<uint64_t>>,
        "``initseq`` used in constructor. "
        "See :cpp:func:`prrng::pcg32_arrayBase::initseq`.");

    cls.def(
        "distance",
        py::overload_cast<const xt::pyarray<uint64_t>&>(
            &Parent::template distance<xt::pyarray<uint64_t>>),
        "Distance to a state. "
        "See :cpp:func:`prrng::pcg32_arrayBase::distance`.",
        py::arg("arg"));

    cls.def(
        "distance",
        py::overload_cast<const Parent&>(&Parent::template distance<Parent>),
        "Distance to a state. "
        "See :cpp:func:`prrng::pcg32_arrayBase::distance`.",
        py::arg("arg"));

    cls.def(
        "advance",
        &Parent::template advance<xt::pyarray<uint64_t>>,
        "Advance generators. "
        "See :cpp:func:`prrng::pcg32_arrayBase::advance`.",
        py::arg("distance"));

    cls.def(
        "restore",
        &Parent::template restore<xt::pyarray<uint64_t>>,
        "Restore state. "
        "See :cpp:func:`prrng::pcg32_arrayBase::restore`.",
        py::arg("state"));
}

template <class C, class Parent, class Data, class State, class Value, class Index>
void init_pcg32_arrayBase_cumsum(C& cls)
{
    cls.def(py::self += Data());
    cls.def(py::self -= Data());
    cls.def(py::self += double());
    cls.def(py::self -= double());

    cls.def_property_readonly("generators", &Parent::generators);

    cls.def_property_readonly("is_extendible", &Parent::is_extendible);

    cls.def_property("data", &Parent::data, &Parent::set_data);

    // cls.def_property(
    //     "generator_index",
    //     &Parent::template generator_index<Index>,
    //     &Parent::template set_generator_index<Index>);

    cls.def_property("start", &Parent::start, &Parent::set_start);

    cls.def_property_readonly("index", &Parent::index);
    cls.def_property_readonly("chunk_index", &Parent::chunk_index);

    cls.def(
        "state_at",
        &Parent::template state_at<State, Index>,
        "Get current state at an index. "
        "See :cpp:func:`prrng::pcg32_arrayBase::state_at`.",
        py::arg("index"));

    // cls.def(
    //     "state",
    //     &Parent::template state<State, Index>,
    //     "Get state at some index.",
    //     py::arg("index"));

    // cls.def(
    //     "set_state",
    //     &Parent::template set_state<State, Index>,
    //     "Set state at some index.",
    //     py::arg("state"),
    //     py::arg("index"));

    cls.def(
        "restore",
        &Parent::template restore<State, Value, Index>,
        "Restore state.",
        py::arg("state"),
        py::arg("value"),
        py::arg("index"));

    cls.def("align", &Parent::template align<Value>, "Align chunk with target.", py::arg("target"));

    cls.def(
        "contains",
        &Parent::template contains<Value>,
        "Check is target is contained in the chunk.",
        py::arg("target"));
}

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

    m.def(
        "version_dependencies",
        &prrng::version_dependencies,
        "Return version string of all dependencies. "
        "See :cpp:func:`prrng::version_dependencies`.");

    m.def(
        "version_compiler",
        &prrng::version_compiler,
        "Return version information of the used compiler. "
        "See :cpp:func:`prrng::version_compiler`.");

    m.def(
        "lower_bound",
        py::overload_cast<const xt::pyarray<double>&, const xt::pyarray<double>&>(
            &prrng::lower_bound<xt::pyarray<double>, xt::pyarray<double>, xt::pyarray<size_t>>),
        "Find column for each row.",
        py::arg("matrix"),
        py::arg("value"));

    m.def(
        "lower_bound",
        py::overload_cast<
            const xt::pyarray<double>&,
            const xt::pyarray<double>&,
            const xt::pyarray<size_t>&,
            size_t>(
            &prrng::lower_bound<xt::pyarray<double>, xt::pyarray<double>, xt::pyarray<size_t>>),
        "Find column for each row.",
        py::arg("matrix"),
        py::arg("value"),
        py::arg("index"),
        py::arg("proximity") = 10);

    m.def(
        "cumsum_chunk",
        &prrng::cumsum_chunk<xt::pyarray<double>, xt::pyarray<ptrdiff_t>>,
        "Compute chunk of chunked cumsum",
        py::arg("cumsum"),
        py::arg("delta"),
        py::arg("shift"));

    py::module minplace = m.def_submodule("inplace", "In-place operations");

    minplace.def(
        "lower_bound",
        &prrng::inplace::lower_bound<xt::pyarray<double>, xt::pyarray<double>, xt::pyarray<size_t>>,
        "Find column for each row.",
        py::arg("matrix"),
        py::arg("value"),
        py::arg("index"),
        py::arg("proximity") = 10);

    minplace.def(
        "cumsum_chunk",
        &prrng::inplace::cumsum_chunk<xt::pyarray<double>, xt::pyarray<ptrdiff_t>>,
        "Compute chunk of chunked cumsum",
        py::arg("cumsum"),
        py::arg("delta"),
        py::arg("shift"));

    m.def(
        "alignment",
        &prrng::alignment,
        "Alignment options. "
        "See :cpp:func:`prrng::alignment`.",
        py::arg("buffer") = 0,
        py::arg("margin") = 0,
        py::arg("min_margin") = 0,
        py::arg("strict") = false);

    py::enum_<prrng::distribution>(m, "distribution")
        .value("random", prrng::distribution::random)
        .value("weibull", prrng::distribution::weibull)
        .value("gamma", prrng::distribution::gamma)
        .value("normal", prrng::distribution::normal)
        .value("exponential", prrng::distribution::exponential)
        .value("delta", prrng::distribution::delta)
        .value("custom", prrng::distribution::custom)
        .export_values();

    py::class_<prrng::normal_distribution>(m, "normal_distribution")

        .def(
            py::init<double, double>(),
            "Normal distribution. "
            "See :cpp:class:`prrng::normal_distribution`.",
            py::arg("mu") = 0,
            py::arg("sigma") = 1)

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
            py::arg("scale") = 1)

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
            py::arg("k") = 1,
            py::arg("scale") = 1)

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
            py::arg("k") = 1,
            py::arg("scale") = 1)

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
            py::arg("mu") = 0,
            py::arg("sigma") = 1)

        .def(
            "cumsum_exponential",
            &prrng::GeneratorBase::cumsum_exponential,
            "The result of the cumsum of `n` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::cumsum_exponential`.",
            py::arg("n"),
            py::arg("scale") = 1)

        .def(
            "cumsum_delta",
            &prrng::GeneratorBase::cumsum_delta,
            "The result of the cumsum of `n` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::cumsum_delta`.",
            py::arg("n"),
            py::arg("scale") = 1)

        .def(
            "cumsum_weibull",
            &prrng::GeneratorBase::cumsum_weibull,
            "The result of the cumsum of `n` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::cumsum_weibull`.",
            py::arg("n"),
            py::arg("k") = 1,
            py::arg("scale") = 1)

        .def(
            "cumsum_gamma",
            &prrng::GeneratorBase::cumsum_gamma,
            "The result of the cumsum of `n` random numbers. "
            "See :cpp:func:`prrng::GeneratorBase::cumsum_gamma`.",
            py::arg("n"),
            py::arg("k") = 1,
            py::arg("scale") = 1)

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
            py::arg("mu") = 0,
            py::arg("sigma") = 1)

        .def(
            "exponential",
            py::overload_cast<const std::vector<size_t>&, double>(
                &prrng::GeneratorBase::exponential<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a exponential distribution. "
            "See :cpp:func:`prrng::GeneratorBase::exponential`.",
            py::arg("shape"),
            py::arg("scale") = 1)

        .def(
            "weibull",
            py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase::weibull<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a weibull distribution. "
            "See :cpp:func:`prrng::GeneratorBase::weibull`.",
            py::arg("shape"),
            py::arg("k") = 1,
            py::arg("scale") = 1)

        .def(
            "gamma",
            py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase::gamma<xt::pyarray<double>, std::vector<size_t>>),
            "ndarray of random numbers, distributed according to a gamma distribution. "
            "See :cpp:func:`prrng::GeneratorBase::gamma`.",
            py::arg("shape"),
            py::arg("k") = 1,
            py::arg("scale") = 1)

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

    py::class_<prrng::pcg32_index, prrng::pcg32>(m, "pcg32_index")

        .def(
            py::init<uint64_t, uint64_t, bool>(),
            "Random number generator. "
            "See :cpp:class:`prrng::pcg32`.",
            py::arg("initstate") = PRRNG_PCG32_INITSTATE,
            py::arg("initseq") = PRRNG_PCG32_INITSEQ,
            py::arg("delta") = false)

        .def("__repr__", [](const prrng::pcg32_index&) { return "<prrng.pcg32_index>"; });

    py::class_<prrng::pcg32_cumsum<xt::pyarray<double>>>(m, "pcg32_cumsum")

        .def(
            py::init<
                const std::vector<size_t>&,
                uint64_t,
                uint64_t,
                enum prrng::distribution,
                const std::vector<double>&,
                const std::vector<size_t>&>(),
            "Generator of cumulative sum of random numbers. "
            "See :cpp:class:`prrng::pcg32_cumsum`.",
            py::arg("shape"),
            py::arg("initstate") = PRRNG_PCG32_INITSTATE,
            py::arg("initseq") = PRRNG_PCG32_INITSEQ,
            py::arg("distribution") = prrng::distribution::custom,
            py::arg("parameters") = std::vector<double>{},
            py::arg("align") = prrng::alignment())

        .def(
            "set_functions",
            &prrng::pcg32_cumsum<xt::pyarray<double>>::set_functions,
            py::arg("get_chunk"),
            py::arg("get_cumsum"),
            py::arg("uses_generator") = true,
            "Set draw function and draw the first chunk.")

        .def(py::self += xt::pyarray<double>())
        .def(py::self -= xt::pyarray<double>())
        .def(py::self += double())
        .def(py::self -= double())

        .def_property_readonly(
            "shape", &prrng::pcg32_cumsum<xt::pyarray<double>>::shape, "Shape of the chunk.")

        .def_property_readonly(
            "size", &prrng::pcg32_cumsum<xt::pyarray<double>>::size, "Size of the chunk.")

        .def_property_readonly(
            "is_extendible", &prrng::pcg32_cumsum<xt::pyarray<double>>::is_extendible)

        .def_property_readonly(
            "generator",
            &prrng::pcg32_cumsum<xt::pyarray<double>>::generator,
            "Underlying generator.")

        .def_property(
            "data",
            &prrng::pcg32_cumsum<xt::pyarray<double>>::data,
            &prrng::pcg32_cumsum<xt::pyarray<double>>::set_data,
            "Current chunk.")

        .def_property(
            "start",
            &prrng::pcg32_cumsum<xt::pyarray<double>>::start,
            &prrng::pcg32_cumsum<xt::pyarray<double>>::set_start,
            "Index of the first entry of the chunk.")

        .def_property_readonly(
            "index", &prrng::pcg32_cumsum<xt::pyarray<double>>::index, "Index or target.")

        .def_property_readonly(
            "chunk_index",
            &prrng::pcg32_cumsum<xt::pyarray<double>>::chunk_index,
            "Index or target in the current chunk.")

        .def(
            "state_at",
            &prrng::pcg32_cumsum<xt::pyarray<double>>::state_at,
            py::arg("index"),
            "State of the generator at any index.")

        .def(
            "restore",
            &prrng::pcg32_cumsum<xt::pyarray<double>>::restore,
            py::arg("state"),
            py::arg("value"),
            py::arg("index"))

        .def("prev", &prrng::pcg32_cumsum<xt::pyarray<double>>::prev, py::arg("margin") = 0)
        .def("next", &prrng::pcg32_cumsum<xt::pyarray<double>>::next, py::arg("margin") = 0)
        .def("align", &prrng::pcg32_cumsum<xt::pyarray<double>>::align, py::arg("target"))
        .def("contains", &prrng::pcg32_cumsum<xt::pyarray<double>>::contains, py::arg("target"))

        .def("__repr__", [](const prrng::pcg32_cumsum<xt::pyarray<double>>&) {
            return "<prrng.pcg32_cumsum>";
        });

    {
        using Parent = prrng::pcg32_array;
        using Class = py::class_<Parent>;

        Class cls(m, "pcg32_array");

        init_GeneratorBase_array<Class, Parent>(cls);
        init_pcg32_arrayBase<Class, Parent>(cls);

        cls.def(
            py::init<const xt::pyarray<uint64_t>&>(),
            "Random number generator. "
            "See :cpp:class:`prrng::pcg32_array`.",
            py::arg("initstate"));

        cls.def(
            py::init<const xt::pyarray<uint64_t>&, const xt::pyarray<uint64_t>&>(),
            "Random number generator. "
            "See :cpp:class:`prrng::pcg32_array`.",
            py::arg("initstate"),
            py::arg("initseq"));

        cls.def("__repr__", [](const Parent&) { return "<prrng.pcg32_array>"; });
    }

    {
        using Parent = prrng::pcg32_index_array;
        using Class = py::class_<Parent>;
        using Index = xt::pyarray<ptrdiff_t>;

        Class cls(m, "pcg32_index_array");

        init_GeneratorBase_array<Class, Parent>(cls);
        init_pcg32_arrayBase<Class, Parent>(cls);

        cls.def(
            py::init<const xt::pyarray<uint64_t>&, const xt::pyarray<uint64_t>&>(),
            "Random number generator. "
            "See :cpp:class:`prrng::pcg32_index_array`.",
            py::arg("initstate"),
            py::arg("initseq"));

        cls.def("__repr__", [](const Parent&) { return "<prrng.pcg32_index_array>"; });
    }

    // py::class_<
    //     prrng::pcg32_index_tensor<1>,
    //     prrng::pcg32_arrayBase<prrng::pcg32_index, std::array<size_t, 1>>>(m,
    //     "pcg32_index_tensor1")

    //     .def(
    //         py::init<const xt::pytensor<uint64_t, 1>&, const xt::pytensor<uint64_t, 1>&>(),
    //         "Random number generator. "
    //         "See :cpp:class:`prrng::pcg32_index_tensor`.",
    //         py::arg("initstate"),
    //         py::arg("initseq"))

    //     .def("__repr__", [](const prrng::pcg32_index_tensor<1>&) {
    //         return "<prrng.pcg32_index_tensor1>";
    //     });

    {
        using Data = xt::pyarray<double>;
        using Index = xt::pyarray<ptrdiff_t>;
        using Parent = prrng::pcg32_array_cumsum<Data, Index>;
        using State = xt::pyarray<uint64_t>;
        using Value = xt::pyarray<double>;
        using Class = py::class_<Parent>;

        Class cls(m, "pcg32_array_cumsum");

        cls.def(
            py::init<
                const std::vector<size_t>&,
                const State&,
                const State&,
                prrng::distribution,
                const std::vector<double>&,
                const std::vector<size_t>&>(),
            "Random number generator. "
            "See :cpp:class:`prrng::pcg32_array`.",
            py::arg("shape"),
            py::arg("initstate"),
            py::arg("initseq"),
            py::arg("distribution"),
            py::arg("parameters"),
            py::arg("align") = prrng::alignment());

        init_pcg32_arrayBase_cumsum<Class, Parent, Data, State, Value, Index>(cls);

        cls.def("__repr__", [](const Parent&) { return "<prrng.pcg32_array_cumsum>"; });
    }

    {
        using Data = xt::pytensor<double, 2>;
        using Index = xt::pytensor<ptrdiff_t, 1>;
        using Parent = prrng::pcg32_tensor_cumsum<Data, Index, 1>;
        using State = xt::pytensor<uint64_t, 1>;
        using Value = xt::pytensor<double, 1>;
        using Class = py::class_<Parent>;

        Class cls(m, "pcg32_tensor_cumsum_1_1");

        cls.def(
            py::init<
                const std::vector<size_t>&,
                const State&,
                const State&,
                prrng::distribution,
                const std::vector<double>&,
                const std::vector<size_t>&>(),
            "Random number generator. "
            "See :cpp:class:`prrng::pcg32_array`.",
            py::arg("shape"),
            py::arg("initstate"),
            py::arg("initseq"),
            py::arg("distribution"),
            py::arg("parameters"),
            py::arg("align") = prrng::alignment());

        init_pcg32_arrayBase_cumsum<Class, Parent, Data, State, Value, Index>(cls);

        cls.def("__repr__", [](const Parent&) { return "<prrng.pcg32_tensor_cumsum_1_1>"; });
    }

} // PYBIND11_MODULE
