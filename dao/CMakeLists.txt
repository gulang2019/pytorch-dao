cmake_minimum_required(VERSION 3.15) 

project(DAO) 

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

pybind11_add_module(pt_dao ${DAO_C_SRCS} ${DAO_PYTHON_SRCS})
target_link_libraries(pt_dao PRIVATE c10)

add_subdirectory(test)