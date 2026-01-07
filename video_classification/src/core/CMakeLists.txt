add_library(video_classification_core STATIC
    json_utils.cpp
    triton_client.cpp
    video_processor.cpp
    videomae_image_processor.cpp
    video_utils.cpp
)

target_include_directories(video_classification_core SYSTEM PUBLIC
    /home/oli/dependencies/triton_client_libs/install/include
)

target_include_directories(video_classification_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

find_library(TRITON_HTTP_CLIENT httpclient PATHS /home/oli/dependencies/triton_client_libs/install/lib NO_DEFAULT_PATH)
if(NOT TRITON_HTTP_CLIENT)
    message(FATAL_ERROR "httpclient library not found")
endif()

target_link_libraries(video_classification_core PUBLIC
    fmt::fmt
    rapidjson
    ${OpenCV_LIBS}
    ${TRITON_HTTP_CLIENT}
)

target_link_directories(video_classification_core PUBLIC
    /home/oli/dependencies/triton_client_libs/install/lib
)





# Enforce C++20
target_compile_features(video_classification_core PUBLIC cxx_std_20)

# Add warnings
target_link_libraries(video_classification_core PRIVATE project_warnings)
