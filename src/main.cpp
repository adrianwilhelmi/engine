#include<string>
#include<iostream>
#include<memory>
#include<cassert>
#include<vector>

#include<SDL3/SDL.h>
#include<SDL3/SDL_vulkan.h>

#include<vulkan/vulkan.h>

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



	Uint32 ext_count = 0;
	const char* const* instance_exts = SDL_Vulkan_GetInstanceExtensions(&ext_count);

	if(instance_exts == NULL){
			std::cerr << "instance exts is null: " << SDL_GetError() << std::endl;
			return 1;
	}

	std::vector<const char*> extensions(instance_exts, instance_exts + ext_count);

	// vulkan instance
	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "SDL3 Vulkan Test";
	appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
	appInfo.pEngineName = "CustomEngine";
	appInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
	appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkInstance instance;
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance!" << std::endl;
        return 1;
    }

    std::cout << "Vulkan instance created successfully" << std::endl;

	assert(pwindow != nullptr);


	bool running = true;
	SDL_Event e;
	while(running){
		while(SDL_PollEvent(&e)){
			if(e.type == SDL_EVENT_QUIT) running = false;
		}
	}

	vkDestroyInstance(instance, nullptr);
	SDL_Quit();

	return 0;
}
