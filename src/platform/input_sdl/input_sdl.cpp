#include<algorithm>
#include<cstring>

#include"input_sdl.hpp"

namespace engine::input{

SDLInput::SDLInput(){
	std::fill(std::begin(current_), std::end(current_), false);
	std::fill(std::begin(previous_), std::end(previous_), false);
}

void SDLInput::new_frame(){
	previous_ = current_;

	std::fill(pressed_.begin(), pressed_.end(), false);
	std::fill(released_.begin(), released_.end(), false);

	mouse_dx_ = 0.0f;
	mouse_dy_ = 0.0f;
}

void SDLInput::process_event(const engine::event::Event& e){
	queue_.push(e);
}

void SDLInput::process_events(){
	engine::event::Event ev;
	while(queue_.pop(ev)){
		switch(ev.type){
			case engine::event::EventType::KeyDown:{
				auto k = static_cast<int>(ev.payload.key.key);
				current_[k] = true;
				break;
			}

			case engine::event::EventType::KeyUp:{
				auto k = static_cast<int>(ev.payload.key.key);
				current_[k] = false;
				break;
			}

			case engine::event::EventType::MouseMove:{
				mouse_x_ = static_cast<float>(ev.payload.mm.x);
				mouse_y_ = static_cast<float>(ev.payload.mm.y);
				mouse_dx_ = static_cast<float>(ev.payload.mm.dx);
				mouse_dy_ = static_cast<float>(ev.payload.mm.dy);
				break;
			}

			case engine::event::EventType::MouseWheel:{
				mouse_wheel_x_ += static_cast<float>(ev.payload.mm.dx);
				mouse_wheel_y_ += static_cast<float>(ev.payload.mm.dy);
				break;
			}

			case engine::event::EventType::WindowResize:{
				break;
			}

			case engine::event::EventType::TextInput:{
				break;
			}

			default:{
				break;
			}

		}
	} //while queue pop

	for(std::size_t i = 0; i < pressed_.size(); ++i){
		pressed_[i] = current_[i] && !previous_[i];
		released_[i] = !current_[i] && previous_[i];
	}
}

bool SDLInput::key_down(Key key) const{
	return current_[static_cast<int>(key)];
}

bool SDLInput::key_pressed(Key key) const{
	return current_[static_cast<int>(key)] 
		&& !previous_[static_cast<int>(key)];
}

bool SDLInput::key_released(Key key) const{
	return !current_[static_cast<int>(key)] 
		&& previous_[static_cast<int>(key)];
}

} // engine::input
