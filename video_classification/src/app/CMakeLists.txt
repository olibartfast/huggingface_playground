add_executable(video_classification_app main.cpp)

target_link_libraries(video_classification_app PRIVATE
    video_classification_core
    project_warnings
)

# Enforce C++20
target_compile_features(video_classification_app PRIVATE cxx_std_20)
