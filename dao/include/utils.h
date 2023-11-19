// concurrent-queue.h
#ifndef DAO_UTIL_H_
#define DAO_UTIL_H_

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <functional>
#include <vector>
#include <sstream>
#include <iomanip>


namespace DAO {
// // Define a function type
// using FunctionType = std::function<void()>;

// // Store serialized functions and their inputs
// std::vector<FunctionType> serializedFunctions;

// // Serialize a function and its inputs
// template <typename Func, typename... Args>
// void serializeFunction(Func func, Args... args) {
//     // Create a lambda function that captures the original function and its arguments
//     auto serializedFunc = [=]() {
//         func(args...);
//     };

//     // Store the lambda function
//     serializedFunctions.push_back(serializedFunc);
// }

template <typename T>
class ConcurrentQueue {
 public:
  T pop() {
    std::cout << "pop" << std::endl;
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    auto val = queue_.front();
    queue_.pop();
    mlock.unlock();
    cond_.notify_one();
    return val;
  }

  void pop(T& item) {
    std::cout << "pop" << std::endl;
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    item = queue_.front();
    queue_.pop();
    mlock.unlock();
    cond_.notify_one();
  }

  void push(const T& item) {
    std::cout << "push" << std::endl;
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.size() >= BUFFER_SIZE) {
       cond_.wait(mlock);
    }
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }

  void push(T&& item) {
    std::cout << "push" << std::endl;
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.size() >= BUFFER_SIZE) {
       cond_.wait(mlock);
    }
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }
  ConcurrentQueue()=default;
  ConcurrentQueue(const ConcurrentQueue&) = delete;            // disable copying
  ConcurrentQueue& operator=(const ConcurrentQueue&) = delete; // disable assignment

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  const static unsigned int BUFFER_SIZE = 10;
};

} // namespace DAO 
#endif // DAO_UTIL_H_
