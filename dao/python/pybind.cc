#include <pybind11/pybind11.h>
#include <DAO.h>
#include <thread>

std::thread executor_thread;

void launch() {
    executor_thread = std::thread(DAO::executor_entry); 
}

void join(){
    executor_thread.join(); 
}

void sync() {
    DAO::synchronize();
}

namespace py = pybind11;

PYBIND11_MODULE(dao, m) {
    m.doc() = R"pbdoc(
        DAO
        -----------------------

        .. currentmodule:: dao 

        .. autosummary::
           :toctree: _generate

           launch
    )pbdoc";

    m.def("launch", &launch, R"pbdoc(
        Launch the executor thread
    )pbdoc");

    m.def("join", &join, R"pbdoc(
        Join the executor thread
    )pbdoc");

    m.def("sync", &sync, R"pbdoc(
        Synchronize the executor thread
    )pbdoc"); 

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
