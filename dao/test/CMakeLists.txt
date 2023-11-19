find_package(Threads REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest) 

enable_testing()
# find_package(GTest REQUIRED)
include(GoogleTest) 

# add_executable(test_kernel test_kernel.cpp)
add_executable(test_lock test_lock.cpp)

set(TEST_LINK_LIBRARIES gtest gtest_main Threads::Threads)
# target_link_libraries(test_kernel ${TEST_LINK_LIBRARIES})
target_link_libraries(test_lock ${TEST_LINK_LIBRARIES})


# gtest_discover_tests(test_kernel)
gtest_discover_tests(test_lock)

# gtest_discover_tests(test_kernel)
# include(GoogleTest)
# gtest_discover_tests(test_kernel)