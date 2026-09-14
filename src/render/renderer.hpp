#pragma once

#include<cstdint>
#include<memory>

#include<platform/window/window.hpp>

namespace render{

struct RenderInitInfo{
	std::shared_ptr<engine::window::Window> window_handle;
	int width, height;
};

class Renderer{
public:
	virtual ~Renderer() = default;
	virtual bool init(const RenderInitInfo& info) = 0;
	//virtual bool resize(int w, int h) = 0;
	virtual void render_frame() = 0;
	virtual void shutdown() = 0;
};

std::unique_ptr<Renderer> create_vulkan_renderer();
std::unique_ptr<Renderer> create_software_renderer();

} // namespace render
