#include<filesystem>
#include<iostream>

#include<asset_compiler/gltf/compiler_gltf.hpp>

#include<fastgltf/core.hpp>

namespace engine::tools::asset{

bool CompilerGltf::compile(std::filesystem::path& source_file){
	auto data = fastgltf::GltfDataBuffer::FromPath(source_file);

	if(data.error() != fastgltf::Error::None){
		std::cerr << "failed to read gltf: " << source_file << '\n';
		return false;
	}

	fastgltf::Parser parser;

	auto asset = parser.loadGltf(
			data.get(),
			source_file.parent_path(),
			fastgltf::Options::LoadExternalBuffers |
			fastgltf::Options::LoadGLBBuffers
	);

	if(asset.error() != fastgltf::Error::None){
		std::cerr << "failed to parse gltf: " << source_file << '\n';
		return false;
	}

	const fastgltf::Asset& gltf = asset.get();

	std::cout << "meshes:		" << gltf.meshes.size() << '\n';
	std::cout << "materials:	" << gltf.materials.size() << '\n';
	std::cout << "textures:		" << gltf.textures.size() << '\n';
	std::cout << "images:		" << gltf.images.size() << '\n';
	std::cout << "skins:		" << gltf.skins.size() << '\n';
	std::cout << "animations:	" << gltf.animations.size() << '\n';

	//todo:
	//	cook everything.. 
	//	write to compiled/temp/source_file/<asset>/<nazwa?>_<i>.<asset>

	return true;
}

} // engine::utils::asset
