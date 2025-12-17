add_library(EngineCore STATIC
	core/memory/allocator.hpp
	core/memory/allocator.cpp
)

set_target_properties(EngineCore PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(EngineCore PUBLIC
	SDL3::SDL3
	Vulkan::Vulkan
)

target_include_directories(EngineCore PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
)
