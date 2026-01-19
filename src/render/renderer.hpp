#pragma once

#include<cstdint>
#include<memory>

namespace render{

struct RenderInitInfo{
	void* window_handle;
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

} // namespace render
