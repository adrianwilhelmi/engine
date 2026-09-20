#pragma once

#include<cstdint>
#include<vector>

#include<core/math/vec2.hpp>
#include<core/math/vec3.hpp>
#include<core/math/vec3packed.hpp>

namespace engine::resource{

/*
 * probuje tera zaprojektowac mesh, tak aby:
 * -sensownie go przechowywac w .pak (zawierajacy cooked assets, obslugiwany przez resource manager)
 * -ulozenie danych i pola byly wydajne pod renderer
 * -simd???
 * -zwieral wszystkie niezbedne pola
 *
 *
 * na pewno jest std::span<stream> ... 
 * ale czy stream ma byc [x,y,z,pad] .. czy jednak [x,x,x,x] , [y,y,y,y] .. ???
 *
 * chyba najlepiej SoA .. czyli cos w stylu:
 * span<float> positions_xs
 * span<float> positions_ys
 * ...
 * span<float> normals_xs
 * span<float> normals_ys
 * i tak dalej
 *
 *
 * a pozniej AoSoA dzielac SoA na rowne chunki (chyba)
 *
 * na razie AoS ze streamami
 *
 *
 * zeby zdecydowac: trzeba bardziej sie zastanowic nad architektura renderera i w ktroych miejscach ulozenie danych Mesh w pamieci ma najwieksze znaczenie..
 * */

struct AABB{
	engine::math::Vec3Packed bound[4];
}

struct SubMesh;

struct Mesh{
	std::span<const engine::math::Vec3> positions;
	std::span<const engine::math::Vec3> normals;
	std::span<const engine::math::Vec2> uvs;
	std::span<const engine::math::Vec4> tangents;

	std::span<const uint32_t> indices; // uint16_t .. czy mesh ma mniej niz 65536 vertexow?

	std::span<const SubMesh> submeshes;

	AABB bounds;
};

} // engine::render
