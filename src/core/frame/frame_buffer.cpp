#include<core/frame/frame_buffer.hpp>

namespace engine{

FrameBuffer::FrameBuffer(uint32_t height, uint32_t width)
	: height_(height), width_(width), data_(static_cast<std::size_t>(height) * width, 0) {}

void FrameBuffer::resize(uint32_t height, uint32_t width){
	this->height_ = height;
	this->width_ = width;
	this->data_.resize(static_cast<std::size_t>(height_) * width_);
}

uint32_t* FrameBuffer::data() noexcept{
	return data_.data();
}

const uint32_t* FrameBuffer::data() const noexcept{
	return data_.data();
}

uint32_t& FrameBuffer::pixel(uint32_t x, uint32_t y) noexcept{
	return this->data_[y*this->width_ + x];
}

uint32_t FrameBuffer::height() const noexcept {return this->height_;}
uint32_t FrameBuffer::width() const noexcept {return this->width_;}
std::size_t FrameBuffer::size() const noexcept {return this->height_*this->width_;}

} // namespace engine
