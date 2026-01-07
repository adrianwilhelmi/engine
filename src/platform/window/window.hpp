#pragma once

#include<string>

#include"window_desc.hpp"

namespace engine::window{

class Window{
public:
	virtual ~Window() = default;
	virtual bool init(const WindowDesc& desc) = 0;

	virtual void poll_events() = 0;

	virtual bool should_close() const = 0;

	virtual uint32_t width() const = 0;
	virtual uint32_t height() const = 0;

	virtual void* native_handle() const = 0;
	virtual void swap_buffers() = 0;
};

} // namespace engine::window

