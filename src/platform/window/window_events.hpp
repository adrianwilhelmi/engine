#pragma once

#include<cstdint>

namespace engine::window{

enum class WindowEventType{
	Close,
	Resize,
	FocusGained,
	FocusLost
};

struct WindowResizeEvent{
	uint32_t width;
	uint32_t height;
};

} // engine::window
