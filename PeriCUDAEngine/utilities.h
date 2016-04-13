//------------------------------------------------------------------------------------------
//
//
// Created on: 1/31/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <stdint.h>
#include <iostream>

#ifndef UTILITIES_H
#define UTILITIES_H

#define PRINT_ERROR(_errStr) \
{ \
    std::cout<< "Error occured at line: " << __LINE__ << ", file: " << __FILE__ << std::endl; \
    std::cout<< "Error message: " << _errStr << std::endl; \
}

#define PRINT_AND_DIE(_errStr) \
{ \
    std::cout<< "Error occured at line: " << __LINE__ << ", file: " << __FILE__ << std::endl; \
    std::cout<< "Error message: " << _errStr << std::endl; \
    exit(EXIT_FAILURE); \
}

#define TRUE_OR_DIE(_condition, _errStr) \
{ \
    if(!(_condition)) \
    { \
        std::cout<< "Fatal error occured at line: " << __LINE__ << ", file: " << __FILE__ << std::endl; \
        std::cout<< "Error message: " << _errStr << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define PRINT_LINE std::cout << __LINE__ << ": " << __FILE__ << std::endl;

typedef unsigned char uchar;



#endif // UTILITIES_H

