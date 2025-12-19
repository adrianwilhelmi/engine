add_library(EngineCore STATIC
	core/memory/allocator_handle.hpp
	core/memory/allocator_utils.hpp
	core/memory/default_heap.hpp
	core/memory/linear_arena.hpp
	core/memory/page_allocator.hpp
	core/memory/pool_allocator.hpp
	core/memory/virtual_memory.hpp

	core/memory/default_heap.cpp
	core/memory/linear_arena.cpp
	core/memory/page_allocator.cpp
	core/memory/pool_allocator.cpp
	core/memory/virtual_memory.cpp
)

set_target_properties(EngineCore PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(EngineCore PUBLIC
	SDL3::SDL3
	Vulkan::Vulkan
)

target_include_directories(EngineCore PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
)
