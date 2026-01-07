#pragma once

#include<memory>

#include"platform/window/window.hpp"

#include<SDL3/SDL.h>

namespace engine::window{

class SDLWindow final : public Window{
public:
	SDLWindow() = default;
	virtual ~SDLWindow() override;

	bool init(const WindowDesc& desc);

	void poll_events(std::unique_ptr<engine::input::Input>& input) override;
	void swap_buffers() override;

	uint32_t width() const override {return width_; }
	uint32_t height() const override {return height_; }
	bool should_close() const override;

	void* native_handle() const override;

private:
	SDL_Window* window_ = nullptr;
	uint32_t width_ = 0;
	uint32_t height_ = 0;
	bool should_close_ = false;

};

} // namespace engine::window
