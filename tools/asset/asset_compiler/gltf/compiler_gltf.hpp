#pragma once

#include<vector>
#include<string_view>
#include<filesystem>

#include<asset_compiler/asset_compiler.hpp>

namespace engine::tools::asset{

class CompilerGltf final : public AssetCompiler{
public:
	bool compile(std::filesystem::path& source_path) override;
private:
	uint64_t create_id(std::filesystem::path& source_path);
};

} //engine::tools::asset
