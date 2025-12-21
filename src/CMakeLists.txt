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


	core/math/vec3.hpp
	core/math/vec3packed.hpp
	core/math/vec4.hpp
)

if(ENGINE_NO_SIMD)
	target_compile_options(EngineCore PUBLIC FORCE_NO_SIMD)
	message(STATUS "SIMD: Manually DISABLED (Fallback mode)")
else()
	if(MSVC)
		target_compile_options(EngineCore PUBLIC /arch:AVX2)
		message(STATUS "SIMD: MSVC detected, enabling /arch:AVX2")
	else()
		if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
			target_compile_options(EngineCore PUBLIC -mavx2 -mfma)
			message(STATUS "SIMD: x86_64 detected, enabling -mavx2 -mfma")
		elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
			message(STATUS "SIMD: ARM64 detected, using default NEON support")
		else()
			message(WARNING "SIMD: Unknown processor architecture. Using default compiler")
		endif()
	endif()
endif()

set_target_properties(EngineCore PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(EngineCore PUBLIC
	SDL3::SDL3
	Vulkan::Vulkan
)

target_include_directories(EngineCore PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
)
