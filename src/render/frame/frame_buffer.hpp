#pragma once

#include<vector>
#include<cstdint>
#include<cstddef>

namespace engine::render{

class FrameBuffer{
public:
	FrameBuffer(uint32_t width, uint32_t height);
	~FrameBuffer() = default;

	void resize(uint32_t height, uint32_t width);

	uint32_t* data() noexcept;
	const uint32_t* data() const noexcept;

	uint32_t height() const noexcept;
	uint32_t width() const noexcept;
	std::size_t size() const noexcept;

	uint32_t& pixel(uint32_t x, uint32_t y) noexcept;

private:
	uint32_t height_ = 0;
	uint32_t width_ = 0;
	std::vector<uint32_t> data_;
};

} // namespace engine::render
