#include <stdio.h>
#include <DAO/generator.h>
#include <cstring>
#include <unistd.h>
#include <stdlib.h>
void print(const char* c) {
    printf("PRINING GROUND TRUTH %s\n", c);
    char* data = new char[strlen(c) + 10];
    strcpy(data, c);
    printf("allocated data %p\n", data);
    auto impl = [data](DAO::Kernel*) {sleep((rand()+0.0) / RAND_MAX / 2); printf("PRINTING %p %s\n", data, data); delete[] data;};
    DAO::Kernel kernel;
    kernel.set_impl(impl);
    DAO::push_kernel(std::move(kernel));
}