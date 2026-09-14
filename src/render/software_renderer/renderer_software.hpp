#pragma once

#include<memory>

#include"platform/window/window.hpp"
#include"render/renderer.hpp"

namespace render{

class SoftwareRenderer : public Renderer{
public:
	SoftwareRenderer() = default;
	~SoftwareRenderer() override = default;

	bool init(const RenderInitInfo& info) override;
	void render_frame() override;
	void shutdown() override;

private:
	int width_;
	int height_;
	std::shared_ptr<engine::window::Window> window_ptr_;

};

} // namespace render
