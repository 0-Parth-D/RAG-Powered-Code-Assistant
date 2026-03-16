cmake_minimum_required(VERSION 3.15...3.26)
project(fast_tokenizer)

# Find the pybind11 library provided by the Python environment
find_package(pybind11 CONFIG REQUIRED)

# Compile the C++ file into a Python module
pybind11_add_module(fast_tokenizer src/tokenizer.cpp)

# Tell CMake to package the compiled extension into the root of the Python wheel
install(TARGETS fast_tokenizer DESTINATION .)

# If compiling with GCC on Windows (MinGW), force static linking of the standard libraries
if (WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_options(fast_tokenizer PRIVATE "-static")
    target_link_options(fast_tokenizer PRIVATE "-static-libgcc")
    target_link_options(fast_tokenizer PRIVATE "-static-libstdc++")
endif()