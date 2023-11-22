find_package(Threads REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest) 

file(GLOB DAO_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cc")
find_package(Python COMPONENTS Interpreter Development) 

message("Python_FOUND:${Python_FOUND}")
message("Python_VERSION:${Python_VERSION}")
message("Python_Development_FOUND:${Python_Development_FOUND}")
message("Python_LIBRARIES:${Python_LIBRARIES}")
message("Python_INCLUDE_DIRS:${Python_INCLUDE_DIRS}")

add_library(dao_c SHARED ${DAO_SRCS})
target_link_libraries(dao_c PUBLIC ${Python_LIBRARIES})
target_include_directories(dao_c PUBLIC ${Python_INCLUDE_DIRS})

enable_testing()
# find_package(GTest REQUIRED)
include(GoogleTest) 

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub)
add_executable(test_kernel test_kernel.cpp)

set(TEST_LINK_LIBRARIES gtest gtest_main Threads::Threads)
target_link_libraries(test_kernel ${TEST_LINK_LIBRARIES} dao_c)
# add_executable(test_lock test_lock.cpp)
# target_link_libraries(test_lock ${TEST_LINK_LIBRARIES} dao)

gtest_discover_tests(test_kernel)
# gtest_discover_tests(test_lock)

execute_process(
  COMMAND bash -c "python3-config --extension-suffix"
  OUTPUT_VARIABLE PYTHON_LIB_SURFIX
)
set(DAO_MODULE_NAME "dao${PYTHON_LIB_SURFIX}")
# set(DAO_MODULE_NAME "libdao.so")
message("DAO_MODULE_NAME:${DAO_MODULE_NAME}")

# add_library(dao_cpython SHARED test_cpython.cpp)
# # set_target_properties(dao_cpython PROPERTIES OUTPUT_NAME ${DAO_MODULE_NAME})
# target_link_libraries(dao_cpython ${Python_LIBRARIES} dao)
# target_include_directories(dao_cpython PUBLIC ${Python_INCLUDE_DIRS})

set(DAO_GENERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../src/generator.cc" "${CMAKE_CURRENT_SOURCE_DIR}/../src/globals.cc")
add_library(printer SHARED printer.cc ${DAO_GENERATOR_SRCS})

set(DAO_EXECUTOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../src/generator.cc" "${CMAKE_CURRENT_SOURCE_DIR}/../src/executor.cc")
add_library(dao SHARED test_cpython.cpp ${DAO_EXECUTOR_SRCS})
target_include_directories(dao PUBLIC ${Python_INCLUDE_DIRS})
target_link_libraries(dao printer ${Python_LIBRARIES})
# set_target_properties(dao PROPERTIES
#   OUTPUT_NAME ${DAO_MODULE_NAME}
#   PREFIX ""
# )

add_executable(test_concurrentcounter test_concurrentcounter.cpp ../src/globals.cc)
target_link_libraries(test_concurrentcounter ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_concurrentcounter)
# gtest_discover_tests(test_kernel)
# include(GoogleTest)
# gtest_discover_tests(test_kernel)