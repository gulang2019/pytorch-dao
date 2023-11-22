#ifndef __DAO_CPYTHON_H__
#define __DAO_CPYTHON_H__
#include <Python.h>

#include <DAO/executor.h>
#include <DAO/globals.h>

namespace DAO {
    static PyObject* launch_wrapper(PyObject* _unused, PyObject* arg) {
        DAO::executor::launch();
        return _unused; 
    }
    // static PyObject* join_wrapper(PyObject* _unused, PyObject* arg) {
    //     DAO::executor::join();
    //     return NULL;
    // }
    static PyObject* sync_wrapper(PyObject* _unused, PyObject* arg) {
        DAO::executor::sync();
        return _unused;
    }

    static PyObject* verbose_wrapper(PyObject* _unused, PyObject* arg) {
        int val;
        if (!PyArg_ParseTuple(arg, "i", &val)) {
            return NULL;
        }
        DAO::verbose = val; 
        return _unused;
    }
    static PyObject* status_wrapper(PyObject* _unused, PyObject* arg) {
        DAO::executor::status();
        return _unused;
    }

    static PyObject* stop_wrapper(PyObject* _unused, PyObject* arg) {
        DAO::executor::stop();
        return _unused;
    }

    static PyMethodDef methods[] = {
        {"_dao_launch", launch_wrapper, METH_NOARGS, nullptr},
        // {"join", join_wrapper, METH_NOARGS, nullptr},
        {"_dao_sync", sync_wrapper, METH_NOARGS, nullptr},
        {"_dao_verbose", verbose_wrapper, METH_VARARGS, nullptr},
        {"_dao_status", status_wrapper, METH_NOARGS, nullptr},
        {"_dao_stop", stop_wrapper, METH_NOARGS, nullptr},
        {nullptr, nullptr, 0, nullptr}
    };
    
    PyMethodDef* python_functions() {
        return methods;
    }
} // namespace DAO

#endif


