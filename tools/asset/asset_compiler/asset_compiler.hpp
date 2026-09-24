#pragma once

#include<asset/asset.hpp>

#include<vector>
#include<filesystem>

namespace engine::tools::asset{

struct CompiledAssetHeader{
	uint32_t magic;
	uint32_t version;

	engine::asset::AssetId id;
	engine::asset::AssetType type;

	uint64_t data_size;
};

class AssetCompilerRegistry{

};

class AssetCompiler{
public:
	virtual ~AssetCompiler() = default;
	virtual bool compile(std::filesystem::path& source_path) = 0;

protected:
	uint64_t create_id(std::filesystem::path& source_path); // id = hash(source_path)
};

} // namespace engine::tools::asset
