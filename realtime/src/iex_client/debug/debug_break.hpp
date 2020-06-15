#pragma once

#ifdef _WIN32
#include <crtdbg.h>
#elif __linux__
#include <csignal>
#endif

namespace jmnel {

#ifdef _WIN32
#define _ARC_DEBUG_BREAK() __debugbreak()
#elif __linux__
#define _ARC_DEBUG_BREAK() std::raise( SIGTRAP )
#endif

#define debug_break() _ARC_DEBUG_BREAK()

}  // namespace jmnel
