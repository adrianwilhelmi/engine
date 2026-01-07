#pragma once

#include"key_codes.hpp"
#include"platform/event/event.hpp"

namespace engine::input{

class Input{
public:
	virtual ~Input() = default;

	virtual void new_frame() = 0;

	virtual void process_event(const engine::event::Event& ev) = 0;
	virtual void process_events() = 0;

	virtual bool key_down(Key key) const = 0;
	virtual bool key_pressed(Key key) const = 0;
	virtual bool key_released(Key key) const = 0;

	virtual float mouse_x() const = 0;
	virtual float mouse_y() const = 0;
	virtual float mouse_dx() const = 0;
	virtual float mouse_dy() const = 0;
};

} // engine::input
