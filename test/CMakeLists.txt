add_executable(memory_test memory_test.cpp)
add_executable(math_test math_test.cpp)

target_link_libraries(memory_test PRIVATE
	EngineCore
	GTest::gtest_main
)

target_link_libraries(math_test PRIVATE
	EngineCore
	GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(memory_test)
gtest_discover_tests(math_test)
