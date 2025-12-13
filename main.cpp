#include<string>
#include<iostream>
#include<memory>

#include<SDL3/SDL.h>

struct SdlDeleter{
	void operator()(SDL_Window*p) const {SDL_DestroyWindow(p);}
};

int main(){
	if(!SDL_Init(SDL_INIT_VIDEO)){
		std::cerr << "SDL_Init err: " << SDL_GetError() << std::endl;
		return 1;
	}

	const std::string window_name = "SDL3_Window";
	const uint64_t width = 800;
	const uint64_t height = 600;
	auto pwindow = std::unique_ptr<SDL_Window, SdlDeleter>(
		SDL_CreateWindow(window_name.c_str(), width, height, SDL_WINDOW_RESIZABLE),
		SdlDeleter()
	);

	if(!pwindow){
		std::cerr << "SDL_CreateWindow err: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}

	bool running = true;
	SDL_Event e;
	while(running){
		while(SDL_PollEvent(&e)){
			if(e.type == SDL_EVENT_QUIT) running = false;
		}
	}

	SDL_Quit();

	return 0;
}
