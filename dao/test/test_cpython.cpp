#include <vector> 

#include <DAO/CPython.h>
#include "printer.h"

namespace DAO {

namespace testing {

void addPyMethodDefs(
    std::vector<PyMethodDef>& vector,
    PyMethodDef* methods) {
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

static std::vector<PyMethodDef> methods;

static PyObject* print_wrapper(PyObject* _unused, PyObject* arg) {
    const char *str;
    if (!PyArg_ParseTuple(arg, "s", &str)) {
        return NULL;
    }
    print(str);
    return _unused;
}

static PyMethodDef printer_methods[] = {
    {"print", print_wrapper, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

PyMODINIT_FUNC
PyInit_dao(void)
{
    addPyMethodDefs(methods, DAO::python_functions());
    addPyMethodDefs(methods, printer_methods);
    static struct PyModuleDef daomodule = {
      PyModuleDef_HEAD_INIT, "dao", nullptr, -1, methods.data()};
    PyObject *m;
    m = PyModule_Create(&daomodule);
    if (m == NULL)
        return NULL;
    return m;
}

}
}