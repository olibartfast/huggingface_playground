find_package(GTest CONFIG REQUIRED)

add_executable(unit_tests
    test_main.cpp
)

target_link_libraries(unit_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    video_classification_core
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
