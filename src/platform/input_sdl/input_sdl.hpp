#pragma once

#include<array>

#include"platform/input/input.hpp"
#include"platform/event/event.hpp"
#include"platform/event/event_queue.hpp"

#include<SDL3/SDL.h>

namespace engine::input{

class SDLInput final : public Input{
public:
	SDLInput();

	void new_frame() override;
	void process_event(const engine::event::Event& e) override;
	void process_events() override;

	bool key_down(Key key) const override;
	bool key_pressed(Key key) const override;
	bool key_released(Key key) const override;

	float mouse_x() const override { return mouse_x_; }
	float mouse_y() const override { return mouse_y_; }
	float mouse_dx() const override { return mouse_dx_; }
	float mouse_dy() const override { return mouse_dy_; }

private:
	static constexpr std::size_t QCAP = 1 << 10;
	engine::event::EventQueue<QCAP> queue_;

    std::array<bool, static_cast<size_t>(Key::Count)> current_{};
    std::array<bool, static_cast<size_t>(Key::Count)> previous_{};
    std::array<bool, static_cast<size_t>(Key::Count)> pressed_{};
    std::array<bool, static_cast<size_t>(Key::Count)> released_{};

	float mouse_x_ = 0.0f;
	float mouse_y_ = 0.0f;
	float mouse_dx_ = 0.0f;
	float mouse_dy_ = 0.0f;
};

} // namespace engine::input
