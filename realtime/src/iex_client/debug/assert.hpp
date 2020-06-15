#pragma once

#include "debug/debug_break.hpp"

//#ifdef _WIN32
//#include <crtdbg.h>
//#elif __linux__
//#include <csignal>
//#endif

#include <cassert>
#include <iostream>

//#define assertf1( x )      \
//    do {                   \
//        (void)sizeof( x ); \
//    } while( 0 )

//#define assertf2( x, msg ) \
//    do {                   \
//        (void)sizeof( x ); \
//    } while( 0 )
//#else

//#ifdef _WIN32
//#define _ARC_DEBUG_BREAK() __debugbreak()
//#elif __linux__
//#define _ARC_DEBUG_BREAK() std::raise( SIGTRAP )
//#endif
//#ifdef _NDEBUG

namespace jmnel {

    inline void _assertf( const char* exp, const char* file, int line ) {
        std::cerr << "Assertion failed: " << exp << std::endl;
        std::cerr << "  File: " << file << std::endl;
        std::cerr << "  Line: " << line;
        std::cerr << std::endl;
    }

    inline void _massertf( const char* exp, const char* msg, const char* file, int line ) {
        std::cerr << "Assertion failed: (2) " << msg << std::endl;
        std::cerr << "  Exp: " << exp << std::endl;
        std::cerr << "  File: " << file << std::endl;
        std::cerr << "  Line: " << line;
        std::cerr << std::endl;
    }

#define assertf( x )                            \
    do {                                        \
        if( !( x ) ) {                          \
            _assertf( #x, __FILE__, __LINE__ ); \
            _ARC_DEBUG_BREAK();                 \
        }                                       \
    } while( 0 )

#define massertf( x, msg )                             \
    do {                                               \
        if( !( x ) ) {                                 \
            _massertf( #x, #msg, __FILE__, __LINE__ ); \
            _ARC_DEBUG_BREAK();                        \
        }                                              \
    } while( 0 )

}  // namespace jmnel

//#endif
