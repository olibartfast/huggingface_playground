cmake_minimum_required(VERSION 3.25)

project(video_classification VERSION 1.0.0 LANGUAGES CXX)

# Dependencies
find_package(fmt CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

# Compiler hardening
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)

# Subdirectories
add_subdirectory(src/core)
add_subdirectory(src/app)

# Testing
option(ENABLE_TESTING "Enable testing" ON)
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests/unit)
    add_subdirectory(tests/e2e)
endif()