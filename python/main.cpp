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


    py::class_<prrng::Generator_array>(m, "Generator_array")

        .def(py::init<>(),
             "Random number generator base class."
             "See :cpp:class:`prrng::Generator_array`.")

        .def("shape",
             [](const prrng::Generator_array& s) { return s.shape(); },
             "Shape of the array of generators."
             "See :cpp:func:`prrng::Generator_array::shape`.")

        .def("shape",
             py::overload_cast<size_t>(&prrng::Generator_array::shape<size_t>, py::const_),
             "Shape of the array of generators, along a certain axis."
             "See :cpp:func:`prrng::Generator_array::shape`.",
             py::arg("axis"))

        .def("size",
             &prrng::Generator_array::size,
             "Size of the array of generators."
             "See :cpp:func:`prrng::Generator_array::size`.")

        .def("random",
             py::overload_cast<const std::vector<size_t>&>(
                &prrng::Generator_array::random<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers."
             "See :cpp:func:`prrng::Generator_array::random`.",
             py::arg("ishape"))

        .def("weibull",
             py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::Generator_array::weibull<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers, distributed according to a weibull distribution."
             "See :cpp:func:`prrng::Generator_array::weibull`.",
             py::arg("ishape"),
             py::arg("k") = 1.0,
             py::arg("l") = 1.0)

        .def("__repr__",
            [](const prrng::Generator_array&) { return "<prrng.Generator_array>"; });


    py::class_<prrng::pcg32_array, prrng::Generator_array>(m, "pcg32_array")

        .def(py::init<xt::xarray<uint64_t>>(),
             "Random number generator."
             "See :cpp:class:`prrng::pcg32_array`.",
             py::arg("initstate"))

        .def(py::init<xt::xarray<uint64_t>, xt::xarray<uint64_t>>(),
             "Random number generator."
             "See :cpp:class:`prrng::pcg32_array`.",
             py::arg("initstate"),
             py::arg("initseq"))

        // https://github.com/pybind/pybind11/blob/master/tests/test_sequences_and_iterators.cpp

        .def("__getitem__", [](prrng::pcg32_array& s, size_t i) {
            if (i >= s.size()) throw py::index_error();
            return &s[i];
        }, py::return_value_policy::reference_internal)

        .def("__getitem__", [](prrng::pcg32_array& s, std::vector<size_t> index) {
            if (!s.inbounds(index)) throw py::index_error();
            return &s[s.flat_index(index)];
        }, py::return_value_policy::reference_internal)

        .def("state",
             &prrng::pcg32_array::state<xt::xarray<uint64_t>>,
             "current state."
             "See :cpp:func:`prrng::pcg32_array::state`.")

        .def("initstate",
             &prrng::pcg32_array::initstate<xt::xarray<uint64_t>>,
             "used initstate."
             "See :cpp:func:`prrng::pcg32_array::initstate`.")

        .def("initseq",
             &prrng::pcg32_array::initseq<xt::xarray<uint64_t>>,
             "used initseq."
             "See :cpp:func:`prrng::pcg32_array::initseq`.")

        .def("restore",
             &prrng::pcg32_array::restore<xt::xarray<uint64_t>>,
             "restore state."
             "See :cpp:func:`prrng::pcg32_array::restore`.",
             py::arg("state"))

        .def("__repr__",
            [](const prrng::pcg32_array&) { return "<prrng.pcg32_array>"; });

} // PYBIND11_MODULE
