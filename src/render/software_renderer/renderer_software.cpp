#pragma once

namespace render{

class SoftwareRenderer : public Renderer{
	SoftwareRenderer();
	~SoftwareRenderer() override;

	bool init(const RenderInitInfo& info) override;
	void render_frame() override;
	void shutdown() override;

private:
	int width;
	int height;
	std::shared_ptr<Window> window_handle;

};


} // namespace render
