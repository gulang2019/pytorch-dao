#ifndef DAO_EXECUTOR_H_
#define DAO_EXECUTOR_H_

#include "kernel-queue.h"

namespace DAO {

class Executor {
public: 
    Executor(ConcurrentQueue<Kernel>& kernel_queue) : kernel_queue_(kernel_queue) {}
    void run();
private: 
    ConcurrentQueue<Kernel>& kernel_queue_;
};

void executor_entry(ConcurrentQueue<Kernel>& kernel_queue); 

} // DAO 

#endif // DAO_EXECUTOR_H_
