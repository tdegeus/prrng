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
             "See :cpp:class:`prrng::Generator`.",
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

} // PYBIND11_MODULE
