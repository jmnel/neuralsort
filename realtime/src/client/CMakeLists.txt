if( ${PLATFORM_LINUX} )
    add_executable( client main_linux.cpp iex_message.cpp iex_decoder.cpp )
endif()

set_property( TARGET client PROPERTY CXX_STANDARD 17 )
set_property( TARGET client PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON )

target_link_libraries( client ${CONAN_LIBS} )
target_link_libraries( client ${Boost_LIBRARIES} )
