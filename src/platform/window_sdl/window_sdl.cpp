#include<iostream>
#include<memory>

#include"platform/window/window.hpp"
#include"platform/window/window_desc.hpp"
#include"platform/window_sdl/window_sdl.hpp"
#include"platform/input_sdl/input_sdl.hpp"

#include<SDL3/SDL.h>

namespace engine::window{

static inline engine::input::Key sdl_to_internal(SDL_Scancode code){
	switch(code){
		case SDL_SCANCODE_W:      return engine::input::Key::W;
		case SDL_SCANCODE_A:      return engine::input::Key::A;
		case SDL_SCANCODE_S:      return engine::input::Key::S;
		case SDL_SCANCODE_D:      return engine::input::Key::D;
		case SDL_SCANCODE_Q:      return engine::input::Key::Q;
		case SDL_SCANCODE_E:      return engine::input::Key::E;
		case SDL_SCANCODE_ESCAPE: return engine::input::Key::Escape;
		case SDL_SCANCODE_SPACE:  return engine::input::Key::Space;
		case SDL_SCANCODE_RETURN: return engine::input::Key::Enter;
		case SDL_SCANCODE_LSHIFT: return engine::input::Key::Shift;
		case SDL_SCANCODE_LCTRL:  return engine::input::Key::Ctrl;
		case SDL_SCANCODE_LALT:   return engine::input::Key::Alt;
		default:                  return engine::input::Key::Unknown;
	}
}

static inline engine::input::Key sdl_mouse_to_internal(uint8_t button){
	switch(button){
		case SDL_BUTTON_LEFT:		return engine::input::Key::MouseLeft;
		case SDL_BUTTON_RIGHT:		return engine::input::Key::MouseRight;
		case SDL_BUTTON_MIDDLE:		return engine::input::Key::MouseMiddle;
		default:					return engine::input::Key::Unknown;
	}
}

SDLWindow::~SDLWindow() {
	if(window_){
		SDL_DestroyWindow(window_);
		SDL_Quit();
	}
}

bool SDLWindow::init(const WindowDesc& desc) {
	if(!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) return false;

	window_ = SDL_CreateWindow(
		desc.title.c_str(),
		desc.width,
		desc.height,
		SDL_WINDOW_RESIZABLE
	);

	this->width_ = desc.width;
	this->height_ = desc.height;

	return window_ != nullptr;
}

void SDLWindow::poll_events(std::unique_ptr<engine::input::Input>& input) {
	SDL_Event e;
	while(SDL_PollEvent(&e)){
		if(e.type == SDL_EVENT_QUIT){
			should_close_ = true;
		}

		engine::event::Event ev{};
		switch(e.type){
			case SDL_EVENT_QUIT:{
				ev.type = engine::event::EventType::WindowClose;
				break;
			}

			case SDL_EVENT_WINDOW_RESIZED:{
				ev.type = engine::event::EventType::WindowResize;
				ev.payload.wr = { 
					static_cast<uint32_t>(e.window.data1),
					static_cast<uint32_t>(e.window.data2) 
				};
				break;
			}

			case SDL_EVENT_KEY_DOWN: {
				engine::input::Key k = sdl_to_internal(
					e.key.scancode
				);

				if(k != engine::input::Key::Unknown){
					ev.type = engine::event::EventType::KeyDown;
					ev.payload.key = { k, e.key.repeat != 0};
				} else continue;
				break;
			}

			case SDL_EVENT_KEY_UP: {
				engine::input::Key k = sdl_to_internal(
					e.key.scancode
				);

				if(k != engine::input::Key::Unknown){
					ev.type = engine::event::EventType::KeyUp;
					ev.payload.key = { k, false };
				} else continue;
				break;
			}

			case SDL_EVENT_MOUSE_BUTTON_DOWN:{
				engine::input::Key k = engine::input::Key::Unknown;

				if(e.button.button == SDL_BUTTON_LEFT){
					k = engine::input::Key::MouseLeft;
				}
				if(e.button.button == SDL_BUTTON_RIGHT){
					k = engine::input::Key::MouseRight;
				}
				if(e.button.button == SDL_BUTTON_MIDDLE){
					k = engine::input::Key::MouseMiddle;
				}

				if(k != engine::input::Key::Unknown){
					ev.type = engine::event::EventType::KeyDown;
					ev.payload.key = {k,false};
				}

				break;
			}

			case SDL_EVENT_MOUSE_BUTTON_UP:{
				engine::input::Key k = engine::input::Key::Unknown;

				if(e.button.button == SDL_BUTTON_LEFT){
					k = engine::input::Key::MouseLeft;
				}
				if(e.button.button == SDL_BUTTON_RIGHT){
					k = engine::input::Key::MouseRight;
				}
				if(e.button.button == SDL_BUTTON_MIDDLE){
					k = engine::input::Key::MouseMiddle;
				}

				if(k != engine::input::Key::Unknown){
					ev.type = engine::event::EventType::KeyUp;
					ev.payload.key = {k,false};
				}

				break;
			}

			case SDL_EVENT_MOUSE_MOTION:{
				ev.type = engine::event::EventType::MouseMove;
				ev.payload.mm = {
					e.motion.x,
					e.motion.y,
					e.motion.xrel,
					e.motion.yrel
				};
				break;
			}

			case SDL_EVENT_MOUSE_WHEEL:{
				ev.type = engine::event::EventType::MouseWheel;
				ev.payload.mm = {0,0, e.wheel.x, e.wheel.y };
				break;
			}

			case SDL_EVENT_TEXT_INPUT:{
				ev.type = engine::event::EventType::TextInput;
				strncpy(
					ev.payload.txt.text, 
					e.text.text, 
					sizeof(ev.payload.txt.text)-1
				);
				ev.payload.txt.text[sizeof(ev.payload.txt.text)-1] = '\0';
				break;
			}

			default:
				continue;
		}

		input->process_event(ev);

		if(input->key_pressed(engine::input::Key::W)){
			std::cout << "W pressed" << std::endl;
		}
	}
}

bool SDLWindow::should_close() const{
	return should_close_;
}

void SDLWindow::swap_buffers(){ 
	std::cout << "swap buffers" << std::endl;
}

void* SDLWindow::native_handle() const{ 
	return (void*)window_;
}

} // namespace engine::window

