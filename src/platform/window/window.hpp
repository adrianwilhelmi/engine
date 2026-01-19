#pragma once

#include<string>
#include<memory>
#include<vector>

#include"window_desc.hpp"
#include"platform/input/input.hpp"

#include<vulkan/vulkan.h>

namespace engine::window{

class Window{
public:
	virtual ~Window() = default;
	virtual bool init(const WindowDesc& desc) = 0;

	virtual void poll_events(std::unique_ptr<engine::input::Input>& input) = 0;

	virtual bool should_close() const = 0;

	virtual uint32_t width() const = 0;
	virtual uint32_t height() const = 0;

	virtual void* native_handle() const = 0;
	virtual void swap_buffers() = 0;

	virtual std::vector<const char*> get_vulkan_instance_extensions() const = 0;
	virtual bool create_vulkan_surface(
		VkInstance instance, 
		VkSurfaceKHR* out_surface
	) const = 0;
};

} // namespace engine::window

