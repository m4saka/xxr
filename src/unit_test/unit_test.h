#pragma once

#include <iostream>

bool testStatus = true;

void expect(const std::string & testName, bool cond)
{
    if (!cond)
    {
        testStatus = false;
    }
#if defined(__linux__) || defined(__APPLE__)
    std::cout << (cond ? "[ \x1b[32mPASS\x1b[39m ]" : "[  \x1b[31mNG\x1b[39m  ]");
#else
    std::cout << (cond ? "[ PASS ]" : "[  NG  ]");
#endif
    std::cout << " " << testName << std::endl;
}