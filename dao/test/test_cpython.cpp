#include <vector> 

#include <DAO/CPython.h>

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

PyMODINIT_FUNC
PyInit_dao(void)
{
    addPyMethodDefs(methods, DAO::python_functions());
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