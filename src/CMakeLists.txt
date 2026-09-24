set(SOURCES_RENDER
	render/software_renderer/renderer_software.hpp
	render/frame/frame_buffer.hpp
	render/color.hpp
	render/render_object.hpp
	render/camera_data.hpp

	render/software_renderer/renderer_software.cpp
	render/software_renderer/renderer_software_factory.cpp

	render/frame/frame_buffer.cpp
)

set(SOURCES_CORE
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


	core/math/vec2.hpp
	core/math/vec3.hpp
	core/math/vec3packed.hpp
	core/math/vec4.hpp
	core/math/mat4.hpp
)

set(SOURCES_PLATFORM
	platform/window/window.hpp
	platform/window/window_desc.hpp
	platform/window_sdl/window_sdl.hpp
	platform/input/input.hpp
	platform/input_sdl/input_sdl.hpp

	platform/window_sdl/window_sdl.cpp
	platform/window_sdl/window_sdl.cpp
	platform/input_sdl/input_sdl.cpp
)

set(SOURCES_WORLD
	world/world.hpp

	world/world.cpp
)

set(SORUCES_RESOURCE
	resource/resource_manager.hpp
	resource/mesh.hpp
)

add_library(EngineCore STATIC
	${SOURCES_RENDER}
	${SOURCES_CORE}
	${SOURCES_PLATFORM}
	${SOURCES_WORLD}
)

add_library(EngineAssetFormat INTERFACE)

target_include_directories(EngineAssetFormat INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENGINE_NO_SIMD)
	target_compile_definitions(EngineCore PUBLIC FORCE_NO_SIMD)
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

target_link_libraries(EngineCore PRIVATE
	engine_strict_flags
)

target_link_libraries(EngineCore PUBLIC
	SDL3::SDL3
)

target_include_directories(EngineCore PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
)
