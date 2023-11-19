#include <gtest/gtest.h> 
#include <kernel-queue.h> 
#include <executor.h> 

TEST(TestKernel, TestKernelQueue) {
    std::cout << "run test" << std::endl;
    auto producer_op = []() { 
        DAO::Kernel kernel; 
        kernel.op = []() { std::cout << "hello world" << std::endl; }; 
        DAO::kernel_queue.push(kernel);
    };

    std::thread producer(producer_op);
    std::thread consumer(DAO::executor_entry, std::ref(DAO::kernel_queue));
    producer.join();
    consumer.join(); 
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}