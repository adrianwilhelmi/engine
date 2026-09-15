#pragma once

namespace engine::render{

struct Color{
	uint8_t a;
	uint8_t r;
	uint8_t g;
	uint8_t b;

	static constexpr Color black() {return {255,0,0,0}; }
	static constexpr Color white() {return {255,255,255,255}; }
	static constexpr Color red() {return {0,255,0,0}; }
	static constexpr Color green() {return {0,0,255,0}; }
	static constexpr Color blue() {return {0,0,0,255}; }
};

constexpr uint32_t pack_argb8888(Color c){
	return
		(static_cast<uint32_t>(c.a) << 24) |
		(static_cast<uint32_t>(c.r) << 16) |
		(static_cast<uint32_t>(c.g) << 8) |
		(static_cast<uint32_t>(c.b));
}

} //engine::render
