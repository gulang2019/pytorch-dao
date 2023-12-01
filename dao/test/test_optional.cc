#include <optional>
#include <vector>
#include <iostream>
#include <thread>
#include <gtest/gtest.h>
#include <DAO/DAO.h>

TEST(TestOptional, HasValue) {
    DAO::executor::launch();
    DAO::Kernel kernel;
    int idx = kernel.set_optional_array(std::optional<std::vector<int64_t>>{std::vector<int64_t>{1, 2, 3, 4, 5}});
    kernel.set_impl([=](DAO::Kernel*kernel){
        auto array = kernel->get_optional_array(idx);
        printf("%p\n", &array);
        EXPECT_EQ(array.has_value(), true);
        EXPECT_EQ((*array)[0], 1);
        EXPECT_EQ((*array)[1], 2);
        EXPECT_EQ((*array)[2], 3);
        EXPECT_EQ((*array)[3], 4);
        EXPECT_EQ((*array)[4], 5);
    });
    DAO::push_kernel(std::move(kernel));
    DAO::executor::stop();
    DAO::executor::sync();    
}

TEST(TestOptional, NullOpt) {
    DAO::executor::launch();
    DAO::Kernel kernel;
    int idx = kernel.set_optional_array(std::optional<std::vector<int64_t>>{std::nullopt});
    kernel.set_impl([=](DAO::Kernel*kernel){
        auto array = kernel->get_optional_array(idx);
        EXPECT_EQ(array.has_value(), false);
    });
    DAO::push_kernel(std::move(kernel));
    DAO::executor::stop();
    DAO::executor::sync();
}

/**
 * NEW Arg Type at::OptionalIntArrayRef
NEW Arg Type c10::optional<int64_t>
NEW Arg Type c10::optional<at::ScalarType>
NEW Arg Type c10::optional<c10::string_view>
NEW Arg Type at::ScalarType
NEW Arg Type c10::string_view
NEW Arg Type c10::optional<bool>
NEW Arg Type c10::optional<double>
*/