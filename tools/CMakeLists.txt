add_executable(AssetCompiler
	asset_compiler/main.cpp
	asset_compiler/asset_compiler.hpp
)

target_link_libraries(AssetCompiler PRIVATE
	fastgltf::fastgltf
	engine_strict_flags
)

add_executable(PakBuilder
	pak_builder/main.cpp
	pak_builder/pak_builder.hpp
)

target_link_libraries(PakBuilder PRIVATE
	engine_strict_flags
)
