#pragma once

#include<cstdint>

#include<platform/input/key_codes.hpp>

namespace engine::event{

enum class EventType : uint8_t {
	None = 0,
	KeyDown,
	KeyUp,
	MouseMove,
	MouseWheel,
	WindowClose,
	WindowResize,
	TextInput
};

struct EventKey{
	engine::input::Key key;
	bool repeat;
};

struct EventMouseButton{
	uint8_t button;
};

struct EventMouseMove{
	float x, y;
	float dx, dy;
};

struct EventWindowResize{
	uint32_t width, height;
};

struct EventText{
	char text[16];
};

struct Event{
	EventType type = EventType::None;

	union{
		EventKey key;
		EventMouseButton mb;
		EventMouseMove mm;
		EventWindowResize wr;
		EventText txt;
	} payload;
};

static inline Event make_keydown(engine::input::Key k, bool repeat=false){
	Event e{};
	e.type = EventType::KeyDown;
	e.payload.key = {k, repeat};
	return e;
}

static inline Event make_keyup(engine::input::Key k){
	Event e{};
	e.type = EventType::KeyUp;
	e.payload.key = {k, false};
	return e;
}

} // namespace engine::event
