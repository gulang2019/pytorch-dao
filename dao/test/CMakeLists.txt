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

add_library(dao SHARED ${DAO_SRCS})
target_link_libraries(dao PUBLIC ${Python_LIBRARIES})
target_include_directories(dao PUBLIC ${Python_INCLUDE_DIRS})

enable_testing()
# find_package(GTest REQUIRED)
include(GoogleTest) 

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub)
add_executable(test_kernel test_kernel.cpp)
add_executable(test_lock test_lock.cpp)

set(TEST_LINK_LIBRARIES gtest gtest_main Threads::Threads)
target_link_libraries(test_kernel ${TEST_LINK_LIBRARIES} dao)
target_link_libraries(test_lock ${TEST_LINK_LIBRARIES} dao)

gtest_discover_tests(test_kernel)
gtest_discover_tests(test_lock)

execute_process(
  COMMAND bash -c "python3-config --extension-suffix"
  OUTPUT_VARIABLE PYTHON_LIB_SURFIX
)
# string(REPLACE "." "-" PYTHON_LIB_SURFIX ${PYTHON_LIB_SURFIX})
# string(REPLACE "-so" ".so" PYTHON_LIB_SURFIX ${PYTHON_LIB_SURFIX})
set(DAO_MODULE_NAME "dao${PYTHON_LIB_SURFIX}")
# set(DAO_MODULE_NAME "libdao.so")
message("DAO_MODULE_NAME:${DAO_MODULE_NAME}")

add_library(dao_cpython SHARED test_cpython.cpp)
# set_target_properties(dao_cpython PROPERTIES OUTPUT_NAME ${DAO_MODULE_NAME})
target_link_libraries(dao_cpython ${Python_LIBRARIES} dao)
target_include_directories(dao_cpython PUBLIC ${Python_INCLUDE_DIRS})


# gtest_discover_tests(test_kernel)
# include(GoogleTest)
# gtest_discover_tests(test_kernel)