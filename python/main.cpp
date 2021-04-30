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


    py::class_<prrng::GeneratorBase>(m, "GeneratorBase")

        .def(py::init<>(),
             "Random number generator base class."
             "See :cpp:class:`prrng::GeneratorBase`.")

        .def("random",
             py::overload_cast<const std::vector<size_t>&>(
                &prrng::GeneratorBase::random<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers."
             "See :cpp:func:`prrng::GeneratorBase::random`.",
             py::arg("shape"))

        .def("weibull",
             py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase::weibull<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers, distributed according to a weibull distribution."
             "See :cpp:func:`prrng::GeneratorBase::weibull`.",
             py::arg("shape"),
             py::arg("k") = 1.0,
             py::arg("l") = 1.0)

        .def("__repr__",
            [](const prrng::GeneratorBase&) { return "<prrng.GeneratorBase>"; });


    py::class_<prrng::pcg32, prrng::GeneratorBase>(m, "pcg32")

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


    py::class_<prrng::GeneratorBase_array<std::vector<size_t>>>(m, "GeneratorBase_array")

        .def(py::init<>(),
             "Random number generator base class."
             "See :cpp:class:`prrng::GeneratorBase_array`.")

        .def("shape",
             [](const prrng::GeneratorBase_array<std::vector<size_t>>& s) { return s.shape(); },
             "Shape of the array of generators."
             "See :cpp:func:`prrng::GeneratorBase_array::shape`.")

        .def("shape",
             py::overload_cast<size_t>(&prrng::GeneratorBase_array<std::vector<size_t>>::shape<size_t>, py::const_),
             "Shape of the array of generators, along a certain axis."
             "See :cpp:func:`prrng::GeneratorBase_array::shape`.",
             py::arg("axis"))

        .def("size",
             &prrng::GeneratorBase_array<std::vector<size_t>>::size,
             "Size of the array of generators."
             "See :cpp:func:`prrng::GeneratorBase_array::size`.")

        .def("random",
             py::overload_cast<const std::vector<size_t>&>(
                &prrng::GeneratorBase_array<std::vector<size_t>>::random<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers."
             "See :cpp:func:`prrng::GeneratorBase_array::random`.",
             py::arg("ishape"))

        .def("weibull",
             py::overload_cast<const std::vector<size_t>&, double, double>(
                &prrng::GeneratorBase_array<std::vector<size_t>>::weibull<xt::xarray<double>, std::vector<size_t>>),
             "ndarray of random number numbers, distributed according to a weibull distribution."
             "See :cpp:func:`prrng::GeneratorBase_array::weibull`.",
             py::arg("ishape"),
             py::arg("k") = 1.0,
             py::arg("l") = 1.0)

        .def("__repr__",
            [](const prrng::GeneratorBase_array<std::vector<size_t>>&) { return "<prrng.GeneratorBase_array>"; });


    py::class_<prrng::pcg32_array, prrng::GeneratorBase_array<std::vector<size_t>>>(m, "pcg32_array")

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
