#ifndef TEST_H
#define TEST_H

#include <string>

bool verify_results(float* test, float* reference, int M, int N);

void run_unit_tests(std::string version);

#endif