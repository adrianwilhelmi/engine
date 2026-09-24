include(FetchContent)

FetchContent_Declare(
	fastgltf
	URL https://github.com/spnda/fastgltf/archive/refs/tags/v0.9.0.tar.gz
)

FetchContent_MakeAvailable(fastgltf)

add_library(AssetCompiler STATIC
	asset_compiler/asset_compiler.hpp
	asset_compiler/gltf/compiler_gltf.hpp
	asset_compiler/gltf/compiler_gltf.cpp
)

target_include_directories(AssetCompiler PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(AssetCompiler PUBLIC
	EngineAssetFormat
)
target_link_libraries(AssetCompiler PRIVATE
	fastgltf::fastgltf
	engine_strict_flags
)


add_library(PakBuilder STATIC
	pak_builder/pak_builder.hpp
	pak_builder/pak_builder.cpp
)

target_link_libraries(PakBuilder PUBLIC
	engine_strict_flags
)

add_executable(AssetPipeline
	main.cpp
)

target_link_libraries(AssetPipeline PRIVATE
	AssetCompiler
	PakBuilder
	engine_strict_flags
)
