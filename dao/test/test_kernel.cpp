#include <gtest/gtest.h> 
#include <DAO/DAO.h>

TEST(TestKernel, TestKernelQueue) {
    auto producer_op = []() { 
        DAO::Kernel kernel; 
        kernel.set_impl([]() { std::cout << "hello world" << std::endl; });
        DAO::push_kernel(std::move(kernel));
    };

    std::thread producer(producer_op);
    DAO_INFO("launching executor");
    DAO::executor::launch();
    DAO_INFO("joining producer");
    producer.join();
    DAO_INFO("producer joined");
    sleep(1);
    DAO_INFO("stopping");
    DAO::executor::sync();
}

int main(int argc, char** argv) {
    DAO::verbose = 1;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}