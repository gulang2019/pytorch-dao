
file(GLOB CSRCS *.cc)

add_library(dao ${CSRCS}) 

target_include_directories(dao PUBLIC ${CMAKE_SOURCE_DIR}/torch/include)
target_link_libraries(dao Threads::Threads)