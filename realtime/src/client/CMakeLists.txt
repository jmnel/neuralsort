if( ${PLATFORM_LINUX} )
    add_executable( client main_linux.cpp 
        iex_message.cpp 
#        iex_hist_list.cpp
        iex_decoder.cpp
        iex_downloader.cpp )
endif()

set_property( TARGET client PROPERTY CXX_STANDARD 17 )
set_property( TARGET client PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON )

target_link_libraries( client ${CONAN_LIBS} )
include_directories( ${Boost_INCLUDE_DIRS} )
#target_link_libraries( client Boost )
