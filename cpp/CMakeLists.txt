# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Information_Optimisation)

# Set the default build type if not already set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Check if the compiler is GCC/G++
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Set optimization flags for GCC (G++)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Release optimizations
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto -march=native")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # Debug build (no optimization)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    endif()

    # Optionally, add more flags like warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find the YAML-CPP library (for YAML support)
find_package(yaml-cpp REQUIRED)

# Add the executable
add_executable(main src/main.cpp src/information_optimisation.cpp)

# Link OpenCV and yaml-cpp libraries to your project
target_link_libraries(main ${OpenCV_LIBS} yaml-cpp )

# Include OpenCV header files
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
