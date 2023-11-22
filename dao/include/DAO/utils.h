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
#include <stdio.h>

#include <DAO/globals.h>

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
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    item = queue_.front();
    queue_.pop();
    mlock.unlock();
    cond_.notify_one();
  }

  void wait_until_empty() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (!queue_.empty()) {
      cond_.wait(mlock);
    }
    mlock.unlock();
    cond_.notify_one();
  }

  void push(const T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.size() >= BUFFER_SIZE) {
       cond_.wait(mlock);
    }
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }

  void push(T&& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.size() >= BUFFER_SIZE) {
       cond_.wait(mlock);
    }
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }
  
  // is this thread safe? 
  int size() const {
    return queue_.size();
  }

  ConcurrentQueue(){printf("ConcurrentQueue constructor %p\n", this);};
  ConcurrentQueue(const ConcurrentQueue&) = delete;            // disable copying
  ConcurrentQueue& operator=(const ConcurrentQueue&) = delete; // disable assignment
  ~ConcurrentQueue() {printf("ConcurrentQueue destructor %p\n", this);}
 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  const static unsigned int BUFFER_SIZE = 1000;
};

class ConcurrentCounter {
public: 
  void increment() {
    std::unique_lock<std::mutex> mlock(mutex_);
    count_++;
    mlock.unlock();
    cond_.notify_one();
  }
  void decrement() {
    std::unique_lock<std::mutex> mlock(mutex_);
    count_--;
    mlock.unlock();
    cond_.notify_one();
  }
  void wait_until_zero() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (count_ != 0) {
      cond_.wait(mlock);
    }
    mlock.unlock();
    cond_.notify_one();
  }
  int peek() const {
    return count_;
  }
private: 
  std::condition_variable cond_;
  std::mutex mutex_; 
  int count_ = 0; 
};

} // namespace DAO 
#endif // DAO_UTIL_H_
