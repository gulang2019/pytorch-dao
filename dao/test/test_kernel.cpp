#include <gtest/gtest.h> 
#include <DAO/DAO.h>

// TEST(TestKernel, TestKernelQueue) {
//     auto producer_op = []() { 
//         DAO::Kernel kernel; 
//         kernel.set_impl([]() { std::cout << "hello world" << std::endl; });
//         DAO::push_kernel(std::move(kernel));
//     };

//     std::thread producer(producer_op);
//     DAO_INFO("launching executor");
//     DAO::executor::launch();
//     DAO_INFO("joining producer");
//     producer.join();
//     DAO_INFO("producer joined");
//     sleep(1);
//     DAO_INFO("stopping");
//     DAO::executor::sync();
// }

TEST(TestKernel, TestSync) {
    // auto producer_op = []() { 
        DAO::Kernel kernel; 
        kernel.set_impl([](DAO::Kernel*) { std::cout << "hello world" << std::endl; });
        DAO::push_kernel(std::move(kernel));
    // };

    // std::thread producer(producer_op);
    DAO_INFO("launching executor");
    DAO::executor::launch();
    for (int i = 0; i < 100; i++) {
        DAO::executor::sync();
    }
    DAO::executor::stop();
    // DAO_INFO("joining producer");
    // producer.join();
    // DAO_INFO("producer joined");
    // sleep(1);
    // DAO_INFO("stopping");
    // DAO::executor::sync();
}