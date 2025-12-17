add_executable(memory_test memory_test.cpp)

target_link_libraries(memory_test PRIVATE
	EngineCore
	GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(memory_test)
