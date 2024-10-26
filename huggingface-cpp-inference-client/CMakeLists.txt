
cmake_minimum_required(VERSION 3.20)
project(huggingface-cpp-inference-client)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED) 
find_package(nlohmann_json 3.2.0 REQUIRED)

# Add cpp-base64
include(FetchContent)
FetchContent_Declare(
  cpp_base64
  GIT_REPOSITORY https://github.com/ReneNyffenegger/cpp-base64.git
  GIT_TAG master  # You might want to specify a specific commit or tag for stability
)
FetchContent_MakeAvailable(cpp_base64)
message(STATUS "cpp_base64_SOURCE_DIR: ${cpp_base64_SOURCE_DIR}")

# Add cxxopts
include(FetchContent)
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.0.0  # You might want to specify a specific commit or tag for stability
)
FetchContent_MakeAvailable(cxxopts)
message(STATUS "cxxopts_SOURCE_DIR: ${cxxopts_SOURCE_DIR}")

# Include directories

# Add library
add_library(huggingface_lib
    src/curl_wrapper.cpp
    src/huggingface_task.cpp
    src/object_detection.cpp
    src/image_classification.cpp
    src/image_segmentation.cpp
    src/image_text_to_text.cpp
    src/image_processing.cpp
    ${cpp_base64_SOURCE_DIR}/base64.cpp
    # Add other task implementations here
)

# Link libraries
target_link_libraries(huggingface_lib PRIVATE CURL::libcurl nlohmann_json::nlohmann_json   ${OpenCV_LIBS})
target_include_directories(huggingface_lib PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/include        
     ${cpp_base64_SOURCE_DIR} 
     ${OpenCV_INCLUDE_DIRS}
     )

# Add main executable
add_executable(huggingface_app main.cpp)

# Include directories for the main executable
target_include_directories(huggingface_app PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${cpp_base64_SOURCE_DIR} 
    ${cxxopts_SOURCE_DIR}/include

)

# Link libraries for the main executable
target_link_libraries(huggingface_app PRIVATE 
    huggingface_lib 
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
)

# # Enable testing
# enable_testing()

# # Add Google Test
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
# )
# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)

# # Include Google Test
# include(GoogleTest)

# # Add test subdirectory
# add_subdirectory(tests)