if( ${PLATFORM_LINUX} )
    add_executable( iex_client main_linux.cpp 
        message.cpp 
#        iex_hist_list.cpp
        decoder.cpp
        downloader.cpp
        idownloader.cpp
#        iex_downloader.cpp )
        )
endif()

include( ${CMAKE_CURRENT_SOURCE_DIR}/debug/CMakeLists.txt )

set_property( TARGET iex_client PROPERTY CXX_STANDARD 17 )
set_property( TARGET iex_client PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON )

target_link_libraries( iex_client ${CONAN_LIBS} )
include_directories( ${Boost_INCLUDE_DIRS} )
#target_link_libraries( client sqlitecpp )
target_link_libraries( iex_client ${Boost_LIBRARIES} )

target_include_directories( iex_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )
