#ifndef __DAO_CPYTHON_H__
#define __DAO_CPYTHON_H__
#include <Python.h>

#include <DAO/executor.h>
#include <DAO/globals.h>

#include <pybind11/pybind11.h>

namespace DAO {
    namespace python {
    // static PyObject* launch_wrapper(PyObject* _unused, PyObject* arg) {
    //     DAO::executor::launch();
    //     return _unused; 
    // }
    // // static PyObject* join_wrapper(PyObject* _unused, PyObject* arg) {
    // //     DAO::executor::join();
    // //     return NULL;
    // // }
    // static PyObject* sync_wrapper(PyObject* _unused, PyObject* arg) {
    //     Py_BEGIN_ALLOW_THREADS
    //     // DAO::executor::sync();
    //     Py_END_ALLOW_THREADS
    //     return _unused;
    // }

    // static PyObject* verbose_wrapper(PyObject* _unused, PyObject* arg) {
    //     int val;
    //     if (!PyArg_ParseTuple(arg, "i", &val)) {
    //         return NULL;
    //     }
    //     DAO::verbose = val; 
    //     return _unused;
    // }
    // static PyObject* status_wrapper(PyObject* _unused, PyObject* arg) {
    //     DAO::executor::status();
    //     return _unused;
    // }

    // static PyObject* stop_wrapper(PyObject* _unused, PyObject* arg) {
    //     Py_BEGIN_ALLOW_THREADS 
    //     DAO::executor::stop();
    //     DAO::executor::sync(); 
    //     Py_END_ALLOW_THREADS 
    //     return _unused;
    // }

    // static PyMethodDef methods[] = {
    //     {"_dao_launch", launch_wrapper, METH_NOARGS, nullptr},
    //     // {"join", join_wrapper, METH_NOARGS, nullptr},
    //     {"_dao_sync", sync_wrapper, METH_NOARGS, nullptr},
    //     {"_dao_verbose", verbose_wrapper, METH_VARARGS, nullptr},
    //     {"_dao_status", status_wrapper, METH_NOARGS, nullptr},
    //     {"_dao_stop", stop_wrapper, METH_NOARGS, nullptr},
    //     {nullptr, nullptr, 0, nullptr}
    // };
    
    // PyMethodDef* python_functions() {
    //     return methods;
    // }

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
        );
    } 

    } // namespace python
} // namespace DAO

#endif


