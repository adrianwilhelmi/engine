#pragma once

#include<memory>
#include<vector>

#include"platform/window/window.hpp"

#include<SDL3/SDL.h>
#ifdef ENGINE_ENABLE_VULKAN
	#include<vulkan/vulkan.h>
#endif // ENGINE_ENABLE_VULKAN

namespace engine::window{

class SDLWindow final : public Window{
public:
	SDLWindow() = default;
	virtual ~SDLWindow() override;

	bool init(const WindowDesc& desc);

	void poll_events(std::shared_ptr<engine::input::Input>& input) override;
	void swap_buffers() override;

	uint32_t width() const override {return width_; }
	uint32_t height() const override {return height_; }
	bool should_close() const override;

	void* native_handle() const override;

#ifdef ENGINE_ENABLE_VULKAN
	std::vector<const char*> get_vulkan_instance_extensions() const override;
	bool create_vulkan_surface(
		VkInstance instance, 
		VkSurfaceKHR* out_surface
	) const override;
#endif // ENGINE_ENABLE_VULKAN

	void present_pixels(const uint32_t* data);

private:
	SDL_Window* window_ = nullptr;
	uint32_t width_ = 0;
	uint32_t height_ = 0;
	bool should_close_ = false;

};

} // namespace engine::window
