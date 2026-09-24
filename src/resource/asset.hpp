#pragma once

namespace engine::utils::asset{

struct AssetId{
	uint64_t value;
};

enum class AssetType{
	Mesh,
	Texture,
	Material
};

struct MeshAsset{
	uint32_t asset_id;

	uint32_t vertex_count;
	uint32_t index_count;
	uint32_t submesh_count;

	uint32_t positions_offset;
	uint32_t normals_offset;
	uint32_t uvs_offset;
	uint32_t tangents_offset;
	uint32_t indices_offset;
	uint32_t submeshes_offset;
	uint32_t AABB_offset;

	// uint32_t index_type; uint32_t albo uint16_t
};

} // namespace engine::utils::assset
