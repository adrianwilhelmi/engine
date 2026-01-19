#pragma once

#include<vector>
#include<memory>

#include"render/renderer.hpp"

#include<vulkan/vulkan.h>

namespace render{

class VulkanRenderer : public Renderer{
public:
	VulkanRenderer();
	~VulkanRenderer() override;

	bool init(const RenderInitInfo& info) override;
	//bool resize(int w, int h) override;
	void render_frame() override;
	void shutdown() override;

private:
	bool create_instance();
	bool pick_physical_device();
	bool create_logical_device();
	bool create_surface();
	bool create_swapchain();
	bool create_image_views();
	bool create_render_pass();
	bool create_framebuffers();
	bool create_command_pool_and_buffers();
	bool create_sync_objects();
	void record_command_buffers();

	//utils
	uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
	int rate_device_suitability(VkPhysicalDevice device);
	bool check_device_extensions(VkPhysicalDevice device);


	void* window_ptr_ = nullptr;
	int width_ = 0;
	int height_ = 0;

	VkInstance instance_ = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
	VkDevice device_ = VK_NULL_HANDLE;
	VkQueue graphics_queue_ = VK_NULL_HANDLE;
	VkQueue present_queue_ = VK_NULL_HANDLE;
	VkSurfaceKHR surface_ = VK_NULL_HANDLE;

	VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
	std::vector<VkImage> swapchain_images_;
	VkFormat swapchain_image_format_ = VK_FORMAT_UNDEFINED;
	VkExtent2D swapchain_extent_{};
	std::vector<VkImageView> swapchain_image_views_;

	VkRenderPass render_pass_ = VK_NULL_HANDLE;
	std::vector<VkFramebuffer> frame_buffers_;

	VkCommandPool command_pool_ = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> command_buffers_;

	std::vector<VkSemaphore> image_available_semaphores_;
	std::vector<VkSemaphore> render_finished_semaphores_;
	std::vector<VkFence> in_flight_fences_;
	std::size_t current_frame_ = 0;
	const int MAX_FRAMES_IN_FLIGHT = 2;

	const std::vector<const char*> device_extensions_ {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	bool initialized_ = false;
};

} // namespace render
