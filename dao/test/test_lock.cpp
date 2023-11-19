#include <thread>
#include <gtest/gtest.h> 
#include <utils.h>

namespace DAO {
namespace testing {

// class MutexBoolTest : public ::testing::Test {
// protected:
DAO::MutexBool mutex_bool;
// };

TEST(MutexBoolTest, SetAndPeek) {
  EXPECT_FALSE(mutex_bool.peek());
  mutex_bool.set();
  EXPECT_TRUE(mutex_bool.peek());
}

TEST(MutexBoolTest, UnsetAndPeek) {
  mutex_bool.set();
  EXPECT_TRUE(mutex_bool.peek());
  mutex_bool.unset();
  EXPECT_FALSE(mutex_bool.peek());
}

TEST(MutexBoolTest, WaitUntilSet) {
  std::thread setter([&]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    mutex_bool.set();
  });

  mutex_bool.wait_until(true);
  EXPECT_TRUE(mutex_bool.peek());

  setter.join();
}

TEST(MutexBoolTest, WaitUntilUnset) {
  mutex_bool.set();

  std::thread unsetter([&]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    mutex_bool.unset();
  });

  mutex_bool.wait_until(false);
  EXPECT_FALSE(mutex_bool.peek());

  unsetter.join();
}

}}  // namespace dao::testing

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}