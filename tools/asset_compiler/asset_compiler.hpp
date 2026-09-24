#pragma once

#include<resource/asset.hpp>

#include<vector>
#include<string_view>

namespace engine::utils::asset{

struct CompiledAssetHeader{
	uint32_t magic;
	uint32_t version;

	AssetId id;
	AssetType type;

	uint64_t data_size;
};

class AssetCompilerRegistry{

};

class AssetCompiler{
public:
	virtual ~AssetCompiler() = default;
	virtual bool compile(std::string_view source_path) = 0;

protected:
	uint64_t create_id(std::string_view source_path); // id = hash(source_path)
};

} // namespace engine::utils::asset
