#include <DAO/utils.h>
#include <gtest/gtest.h>
using namespace DAO;
TEST(ConcurrentCounter, Increment) {
  ConcurrentCounter counter;
  ASSERT_EQ(counter.peek(), 0);
  counter.increment();
  ASSERT_EQ(counter.peek(), 1);
}

TEST(ConcurrentCounter, Decrement) {
  ConcurrentCounter counter;
  counter.increment();
  ASSERT_EQ(counter.peek(), 1);
  counter.decrement();
  ASSERT_EQ(counter.peek(), 0);
}

TEST(ConcurrentCounter, WaitUntilZero) {
  ConcurrentCounter counter;
  counter.increment();
  std::thread t([&]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    counter.decrement();
  });
  counter.wait_until_zero();
  ASSERT_EQ(counter.peek(), 0);
  t.join();
}

TEST(ConcurrentCounter, MultipleIncrementsAndDecrements) {
  ConcurrentCounter counter;
  for (int i = 0; i < 10; ++i) {
    counter.increment();
  }
  ASSERT_EQ(counter.peek(), 10);
  for (int i = 0; i < 10; ++i) {
    counter.decrement();
  }
  ASSERT_EQ(counter.peek(), 0);
}

TEST(ConcurrentCounter, IncrementWaitAndDecrement) {
  ConcurrentCounter counter;
  ConcurrentQueue<int> queue;

  std::thread t1([&]() {
    for(int i = 0; i < 100; i++){
        counter.increment();
        printf("pushing %d\n", i);
        queue.push(i);
        counter.wait_until_zero();
    }
    // counter.wait_until_zero();
  });

  std::thread t2([&]() {
    for (int i = 0; i < 100; i++) {
      auto val = queue.pop();
      printf("val = %d\n", val);
      counter.decrement();
    }
  });

  t1.join();
  t2.join();

  ASSERT_EQ(counter.peek(), 0);
}