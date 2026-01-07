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

#include<SDL3/SDL.h>
#include<SDL3/SDL_vulkan.h>
#include<vulkan/vulkan.h>

int main(){
	engine::window::WindowDesc desc;
	desc.title = "engine testin";
	desc.width = 1280;
	desc.height = 720;

	std::unique_ptr<engine::window::Window> window = std::make_unique<engine::window::SDLWindow>();

	if(!window->init(desc)){
		std::cerr << "failed to init window" << std::endl;
		return -1;
	}

	std::cout << "engine started" << std::endl;

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

	while (!window->should_close()) {
        window->poll_events();
    }

	bool running = true;
	vkDestroyInstance(instance, nullptr);


	return 0;
}
