cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
add_definitions("-std=c++17")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_BUILD_TYPE Debug)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

project(realtime_trader VERSION 0.1 LANGUAGES CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

### Detect build platform ###
if( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
    set( PLATFORM_LINUX TRUE )
    message( STATUS "Linux platform detected." )
elseif( ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin" )
    set( PLATFORM_APPLE TRUE )
    message( STATUS "Apple platform detected." )
elseif( ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )
    set( PLATFORM_WIN64 TRUE )
    message( STATUS "Windows platform detected." )
else( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
    message( FATAL_ERROR "Failed to determine platform." )
endif( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )

set(WARNING_FLAGS
    "-pedantic "
    "-Wall "
    "-Werror=return-type "
    "-Wextra "
    "-Wmissing-field-initializers "
    "-Wno-predefined-identifier-outside-function "
    )

set(MISC_FLAGS
    "-Wpedantic "
    "-fcolor-diagnostics "
#    "-fsanitize=undefined,address "
    "-mlzcnt "
#    "-fsanitize-trap=all "
#    "-fno-exceptions "
#    "-O2 -fprofile-instr-generate -fcoverage-mapping -fprofile-arcs -ftest-coverage "
#    "-O2 "
    )

string(CONCAT WARNING_FLAGS_STR ${WARNING_FLAGS})
string(CONCAT MISC_FLAGS_STR ${MISC_FLAGS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MISC_FLAGS_STR}${WARNING_FLAGS_STR}")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Arcturus_SOURCE_DIR}/../targets/Debug)
set(CMAKE_MODULE_PATH ${Arcturus_SOURCE_DIR}/cmake/modules)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/client)
