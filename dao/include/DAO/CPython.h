#ifndef __DAO_CPYTHON_H__
#define __DAO_CPYTHON_H__
#include <Python.h>

#include <DAO/executor.h>
#include <DAO/globals.h>

#include <pybind11/pybind11.h>

namespace DAO {
    namespace python {

    void initModule(PyObject* module) {
        auto m = py::cast<py::module>(module);
        m.def(
            "_dao_launch",
            []() {
                DAO::executor::launch();
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_dao_sync",
            []() {
                DAO::executor::sync();
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_dao_verbose",
            [](int val) {
                DAO::verbose = val;
            },
            py::call_guard<py::gil_scoped_release>(),
            py::arg("val")
        )
        .def(
            "_dao_status",
            []() {
                DAO::executor::status();
            },
            py::call_guard<py::gil_scoped_release>()
        )
        .def(
            "_dao_stop",
            []() {
                DAO::executor::stop();
                DAO::executor::sync();
            },
            py::call_guard<py::gil_scoped_release>()
        )
        .def(
            "_dao_log",
            [](const char* msg) {
                DAO::executor::log(msg);
            },
            py::call_guard<py::gil_scoped_release>(),
            py::arg("msg")
        );
    } 

    } // namespace python
} // namespace DAO

#endif


