#pragma once

#include<cstdint>
#include<string>

namespace engine::window{

struct WindowDesc{
	std::string title = "Engine";
	uint32_t width = 1280;
	uint32_t height = 720;
	bool resizable = true;
	bool fullscreen = false;
	bool high_dpi = true;
};

} // namespace engine::window
