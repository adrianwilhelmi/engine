#include<iostream>
#include<filesystem>

#include<asset_compiler/asset_compiler.hpp>
#include<asset_compiler/gltf/compiler_gltf.hpp>
#include<pak_builder/pak_builder.hpp>


int main(){
	std::filesystem::path input = "tools/asset/input/models/example/GlassHurricaneCandleHolder/glTF/GlassHurricaneCandleHolder.gltf";
	const std::filesystem::path compiled = "tools/asset/compiled";
	const std::filesystem::path output = "data/paks/game.pak";

	engine::tools::asset::CompilerGltf compiler_gltf;

	if(!compiler_gltf.compile(input)){
		std::cout << "gltf compilation failed" << '\n';
	}

	return 0;
}
