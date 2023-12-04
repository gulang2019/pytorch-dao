#ifndef DAO_EXECUTOR_H_
#define DAO_EXECUTOR_H_

#include <DAO/generator.h>
#include <DAO/utils.h>

namespace DAO {
namespace executor {
class Executor {
public: 
    Executor(ConcurrentQueue<Kernel>& kernel_queue) : kernel_queue_(kernel_queue) {}
    void run();
private: 
    ConcurrentQueue<Kernel>& kernel_queue_;
};

void launch();
// void join();
void status();
void sync();
void stop();
void log (const char* msg);
// void stop();
} // executor 

} // DAO 

#endif // DAO_EXECUTOR_H_
