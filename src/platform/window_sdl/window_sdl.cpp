#include<iostream>

#include"platform/window/window.hpp"
#include"platform/window/window_desc.hpp"
#include"platform/window_sdl/window_sdl.hpp"

#include<SDL3/SDL.h>

namespace engine::window{

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
	return window_ != nullptr;
}

void SDLWindow::poll_events() {
	SDL_Event event;
	while(SDL_PollEvent(&event)){
		if(event.type == SDL_EVENT_QUIT){
			should_close_ = true;
		}
		if(event.type == SDL_EVENT_KEY_DOWN){
			//Input::set_key_state(event.key.scancode, true);
		}
		if(event.type == SDL_EVENT_KEY_UP){
			//Input::set_key_state(event.key.scancode, false);
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

