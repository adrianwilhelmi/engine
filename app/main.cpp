#include<string>
#include<iostream>
#include<memory>
#include<cassert>
#include<vector>

#include<core/memory/default_heap.hpp>
#include<core/memory/linear_arena.hpp>
#include<core/memory/pool_allocator.hpp>

#include<platform/window/window.hpp>
#include<platform/window_sdl/window_sdl.hpp>
#include<platform/input/input.hpp>
#include<platform/input_sdl/input_sdl.hpp>
#include<platform/input/key_codes.hpp>
#include<render/renderer.hpp>

#include<SDL3/SDL.h>
#include<SDL3/SDL_vulkan.h>
#include<vulkan/vulkan.h>

const char* key_to_name(engine::input::Key key) {
    using namespace engine::input;
    switch (key) {
        case Key::Q: return "Q";
        case Key::W: return "W";
        case Key::E: return "E";
        case Key::A: return "A";
        case Key::S: return "S";
        case Key::D: return "D";
        case Key::Escape: return "Escape";
        case Key::Space:  return "Space";
        case Key::Enter:  return "Enter";
        case Key::MouseLeft:  return "MouseLeft";
        case Key::MouseRight:  return "MouseRight";
        case Key::MouseMiddle:  return "MouseMiddle";
        default: return "Unknown";
    }
}

int main(){

	engine::window::WindowDesc desc;
	desc.title = "engine testin";
	desc.width = 1280;
	desc.height = 720;

	std::unique_ptr<engine::input::Input> input = 
		std::make_unique<engine::input::SDLInput>();

	std::unique_ptr<engine::window::Window> window =
		std::make_unique<engine::window::SDLWindow>();

	if(!window->init(desc)){
		std::cerr << "failed to init window" << std::endl;
		return -1;
	}

	std::cout << "window started" << std::endl;


	auto renderer = render::create_vulkan_renderer();

	render::RenderInitInfo init_info;
	init_info.window_handle = window.get();
	init_info.width = 1280;
	init_info.height = 720;

	if(!renderer->init(init_info)){
		return -1;
	}

	std::cout << "renderer started" << std::endl;

	float prev_mouse_x = 0.0;
	float prev_mouse_y = 0.0;
	float new_mouse_x = 0.0;
	float new_mouse_y = 0.0;

	float prev_mouse_wheel_x = 0.0f;
	float prev_mouse_wheel_y = 0.0f;
	float new_mouse_wheel_x = 0.0f;
	float new_mouse_wheel_y = 0.0f;

	while (!window->should_close()) {
		input->new_frame();
        window->poll_events(input);
		input->process_events();

		// keys
		for(int i = 0; i < (int)engine::input::Key::Count; ++i){
			auto k = static_cast<engine::input::Key>(i);

			if(input->key_pressed(k)){
				std::cout << key_to_name(k) << " key pressed" << std::endl;
			}
			if(input->key_released(k)){
				std::cout << key_to_name(k) << " key released" << std::endl;
			}
		}

		// mouse
		new_mouse_x = input->mouse_x();
		new_mouse_y = input->mouse_y();
		if(new_mouse_x != prev_mouse_x){
			std::cout << "mouse movement detected:" << std::endl;
			std::cout << "new mouse x: " << new_mouse_x << std::endl;
		}
		if(new_mouse_y != prev_mouse_y){
			std::cout << "mouse movement detected:" << std::endl;
			std::cout << "new mouse y: " << new_mouse_y << std::endl;
		}

		prev_mouse_x = new_mouse_x;
		prev_mouse_y = new_mouse_y;

		// mouse wheel
		new_mouse_wheel_x = input->mouse_wheel_x();
		new_mouse_wheel_y = input->mouse_wheel_y();
		if(new_mouse_wheel_x != prev_mouse_wheel_x){
			std::cout << "mouse WHEEL movement detected:" << std::endl;
			std::cout << "new mousewheel x: " << new_mouse_wheel_x << std::endl;
		}
		if(new_mouse_wheel_y != prev_mouse_wheel_y){
			std::cout << "mouse WHEEL movement detected:" << std::endl;
			std::cout << "new mousewheel y: " << new_mouse_wheel_y << std::endl;
		}

		prev_mouse_wheel_x = new_mouse_wheel_x;
		prev_mouse_wheel_y = new_mouse_wheel_y;

		renderer->render_frame();
    }

	bool running = true;

	return 0;
}
