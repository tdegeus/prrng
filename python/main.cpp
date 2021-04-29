#include <pyxtensor/pyxtensor.hpp>
#include <pybind11/operators.h>
#include <prrng.h>

PYBIND11_MODULE(prrng, m)
{

    m.doc() = "Portable Reconstructible Random Number Generator";

    m.def("version",
          &prrng::version,
          "Return version string."
          "See :cpp:class:`prrng::version`.");

    py::class_<prrng::Generator>(m, "Generator")

        .def(py::init<>(),
             "Random number generator base class."
             "See :cpp:class:`prrng::Generator`.")

        .def("random",
             py::overload_cast<const std::vector<size_t>&>(
                &prrng::Generator::random<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers."
             "See :cpp:func:`prrng::Generator::random`.",
             py::arg("shape"))

        .def("weibull",
             py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::Generator::weibull<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers, distributed according to a weibull distribution."
             "See :cpp:func:`prrng::Generator::weibull`.",
             py::arg("shape"),
             py::arg("k") = 1.0,
             py::arg("l") = 1.0)

        .def("__repr__",
            [](const prrng::Generator&) { return "<prrng.Generator>"; });

    py::class_<prrng::pcg32, prrng::Generator>(m, "pcg32")

        .def(py::init<uint64_t, uint64_t>(),
             "Random number generator."
             "See :cpp:class:`prrng::pcg32`.",
             py::arg("initstate") = PRRNG_PCG32_INITSTATE,
             py::arg("initseq") = PRRNG_PCG32_INITSEQ)

        .def(py::self - py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)

        .def("state",
             &prrng::pcg32::state<uint64_t>,
             "current state."
             "See :cpp:func:`prrng::pcg32::state`.")

        .def("initstate",
             &prrng::pcg32::initstate<uint64_t>,
             "used initstate."
             "See :cpp:func:`prrng::pcg32::initstate`.")

        .def("initseq",
             &prrng::pcg32::initseq<uint64_t>,
             "used initseq."
             "See :cpp:func:`prrng::pcg32::initseq`.")

        .def("restore",
             &prrng::pcg32::restore<uint64_t>,
             "restore state."
             "See :cpp:func:`prrng::pcg32::restore`.",
             py::arg("state"))

        .def("distance",
            py::overload_cast<uint64_t>(
                &prrng::pcg32::distance<int64_t, uint64_t>),
             "distance."
             "See :cpp:func:`prrng::pcg32::distance`.",
             py::arg("state"))

        .def("distance",
            py::overload_cast<const prrng::pcg32&>(
                &prrng::pcg32::distance<int64_t>),
             "distance."
             "See :cpp:func:`prrng::pcg32::distance`.",
             py::arg("generator"))

        .def("advance",
             &prrng::pcg32::advance<int64_t>,
             "advance by a distance."
             "See :cpp:func:`prrng::pcg32::advance`.",
             py::arg("distance"))

        .def("random",
             py::overload_cast<const std::vector<size_t>&>(
                &prrng::pcg32::random<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers."
             "See :cpp:func:`prrng::pcg32::random`.",
             py::arg("shape"))

        .def("__repr__",
            [](const prrng::pcg32&) { return "<prrng.pcg32>"; });

    py::class_<prrng::nd_Generator>(m, "nd_Generator")

        .def(py::init<>(),
             "Random number generator base class."
             "See :cpp:class:`prrng::nd_Generator`.")

        .def("size",
             &prrng::nd_Generator::size,
             "Size of the array of generators."
             "See :cpp:func:`prrng::nd_Generator::size`.")

        .def("random",
             py::overload_cast<const std::vector<size_t>&>(
                &prrng::nd_Generator::random<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers."
             "See :cpp:func:`prrng::nd_Generator::random`.",
             py::arg("shape"))

        .def("weibull",
             py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::nd_Generator::weibull<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers, distributed according to a weibull distribution."
             "See :cpp:func:`prrng::nd_Generator::weibull`.",
             py::arg("shape"),
             py::arg("k") = 1.0,
             py::arg("l") = 1.0)

        .def("__repr__",
            [](const prrng::nd_Generator&) { return "<prrng.nd_Generator>"; });

    py::class_<prrng::nd_pcg32, prrng::nd_Generator>(m, "nd_pcg32")

        .def(py::init<xt::xarray<uint64_t>>(),
             "Random number generator."
             "See :cpp:class:`prrng::nd_pcg32`.",
             py::arg("initstate"))

        .def(py::init<xt::xarray<uint64_t>, xt::xarray<uint64_t>>(),
             "Random number generator."
             "See :cpp:class:`prrng::nd_pcg32`.",
             py::arg("initstate"),
             py::arg("initseq"))

        // https://github.com/pybind/pybind11/blob/master/tests/test_sequences_and_iterators.cpp

        .def("__getitem__", [](prrng::nd_pcg32& s, size_t i) {
            if (i >= s.size()) throw py::index_error();
            return &s[i];
        }, py::return_value_policy::reference_internal)

        .def("__getitem__", [](prrng::nd_pcg32& s, std::vector<size_t> index) {
            if (!s.inbounds(index)) throw py::index_error();
            return &s.get(index);
        }, py::return_value_policy::reference_internal)
        // .def(py::self - py::self)
        // .def(py::self == py::self)
        // .def(py::self != py::self)

        .def("state",
             &prrng::nd_pcg32::state<xt::xarray<uint64_t>>,
             "current state."
             "See :cpp:func:`prrng::nd_pcg32::state`.")

        .def("initstate",
             &prrng::nd_pcg32::initstate<xt::xarray<uint64_t>>,
             "used initstate."
             "See :cpp:func:`prrng::nd_pcg32::initstate`.")

        .def("initseq",
             &prrng::nd_pcg32::initseq<xt::xarray<uint64_t>>,
             "used initseq."
             "See :cpp:func:`prrng::nd_pcg32::initseq`.")

        .def("restore",
             &prrng::nd_pcg32::restore<xt::xarray<uint64_t>>,
             "restore state."
             "See :cpp:func:`prrng::nd_pcg32::restore`.",
             py::arg("state"))

        .def("__repr__",
            [](const prrng::nd_pcg32&) { return "<prrng.nd_pcg32>"; });

} // PYBIND11_MODULE
