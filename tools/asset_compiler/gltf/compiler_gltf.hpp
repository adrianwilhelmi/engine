#pragma once

#include<vector>
#include<string_view>

#include<resource/compiler/asset_compiler.hpp>

namespace engine::utils::asset{

class CompilerGltf final : public AssetCompiler{
public:
	bool compile(std::string_view source_path) override;
privte:
	uint64_t create_id(std::string_view source_path);
};

} //engine::utils::asset
