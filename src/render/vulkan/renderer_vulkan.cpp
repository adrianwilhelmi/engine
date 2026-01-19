#include<iostream>
#include<set>
#include<limits>
#include<algorithm>
#include<cstring>

#include"platform/window/window.hpp"
#include"render/vulkan/renderer_vulkan.hpp"

#define VK_CHECK(call) \
    do { VkResult res = (call); if (res != VK_SUCCESS) { std::cerr << "Vulkan error: " << res << " at " << __FILE__ << ":" << __LINE__ << std::endl; return false; } } while(0)

namespace render{

VulkanRenderer::VulkanRenderer() = default;
VulkanRenderer::~VulkanRenderer() {
	shutdown();
}

bool VulkanRenderer::init(const RenderInitInfo& info){
	this->window_ptr_ = info.window_handle;
	this->width_ = info.width;
	this->height_ = info.height;

	if(!create_instance()) return false;
	if(!create_surface()) return false;
	if(!pick_physical_device()) return false;
	if(!create_logical_device()) return false;
	if(!create_swapchain()) return false;
	if(!create_image_views()) return false;
	if(!create_render_pass()) return false;
	if(!create_framebuffers()) return false;
	if(!create_command_pool_and_buffers()) return false;
	if(!create_sync_objects()) return false;

	record_command_buffers();

	this->initialized_ = true;
	return true;
}

/*
bool VulkanRenderer::resize(int w, int h){
	//...
}
*/

void VulkanRenderer::render_frame(){
	if(!this->initialized_) return;

	vkWaitForFences(
		device_,
		1,
		&in_flight_fences_[current_frame_],
		VK_TRUE,
		UINT64_MAX
	);

	uint32_t image_index;
	VkResult result = vkAcquireNextImageKHR(
		device_,
		swapchain_,
		UINT64_MAX,
		image_available_semaphores_[current_frame_],
		VK_NULL_HANDLE,
		&image_index
	);

	if(result == VK_ERROR_OUT_OF_DATE_KHR){
		//resize(width_, height_);
		return;
	}
	else if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR){
		std::cerr << "faield to acquire swap chain image!" << std::endl;
		return;
	}

	VkSubmitInfo submit_info{};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	VkSemaphore wait_semaphores[] = {image_available_semaphores_[current_frame_]};
	VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = wait_semaphores;
	submit_info.pWaitDstStageMask = wait_stages;

	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &command_buffers_[image_index];

	VkSemaphore signal_semaphores[] = {render_finished_semaphores_[current_frame_] };
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = signal_semaphores;

	vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);

	if(vkQueueSubmit(
		graphics_queue_, 
		1, 
		&submit_info, 
		in_flight_fences_[current_frame_]
	) != VK_SUCCESS){
		std::cerr << "failed to submit draw command buffer!!" << std::endl;
		return;
	}

	VkPresentInfoKHR present_info{};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = signal_semaphores;

	VkSwapchainKHR swapchains[] = {swapchain_};
	present_info.swapchainCount = 1;
	present_info.pSwapchains = swapchains;
	present_info.pImageIndices = &image_index;

	VkResult present_res = vkQueuePresentKHR(present_queue_, &present_info);

	if (present_res == VK_ERROR_OUT_OF_DATE_KHR || present_res == VK_SUBOPTIMAL_KHR) {
		//resize(width_, height_);
	} else if (present_res != VK_SUCCESS) {
		std::cerr << "failed to present swap chain image!" << std::endl;
	}

	current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanRenderer::shutdown(){
	if(device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);

	for(std::size_t i = 0; i < in_flight_fences_.size(); ++i){
		if(in_flight_fences_[i] != VK_NULL_HANDLE)
			vkDestroyFence(device_, in_flight_fences_[i], nullptr);
	}

	for(std::size_t i = 0; i < image_available_semaphores_.size(); ++i){
		if(image_available_semaphores_[i] != VK_NULL_HANDLE)
			vkDestroySemaphore(device_, image_available_semaphores_[i], nullptr);
		if(render_finished_semaphores_[i] != VK_NULL_HANDLE)
			vkDestroySemaphore(device_, render_finished_semaphores_[i], nullptr);
	}

	if(command_pool_ != VK_NULL_HANDLE){
		vkDestroyCommandPool(
			device_,
			command_pool_,
			nullptr
		);
	}

	for(auto fb : frame_buffers_) vkDestroyFramebuffer(device_, fb, nullptr);
	frame_buffers_.clear();

	if(render_pass_ != VK_NULL_HANDLE){
		vkDestroyRenderPass(
			device_,
			render_pass_,
			nullptr
		);
	}

	for(auto iv : swapchain_image_views_) vkDestroyImageView(device_, iv, nullptr);
	swapchain_image_views_.clear();

	if(swapchain_ != VK_NULL_HANDLE){
		vkDestroySwapchainKHR(
			device_,
			swapchain_,
			nullptr
		);
	}

	if(device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
	device_ = VK_NULL_HANDLE;
	physical_device_ = VK_NULL_HANDLE;

	if(surface_ != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance_, surface_, nullptr);
	if(instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
	instance_ = VK_NULL_HANDLE;

	initialized_ = false;
}

bool VulkanRenderer::create_instance() {
    engine::window::Window* win = reinterpret_cast<engine::window::Window*>(
		window_ptr_
	);

    if(!win){
        std::cerr << "Window pointer is null!" << std::endl;
        return false;
    }
    auto inst_exts = win->get_vulkan_instance_extensions();

    std::vector<const char*> extensions = inst_exts;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "EngineApp";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName = "CustomEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

	#ifdef __APPLE__
		extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	#endif


    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

    createInfo.enabledLayerCount = 0;

    VkResult res = vkCreateInstance(&createInfo, nullptr, &instance_);
    if(res != VK_SUCCESS){
        std::cerr << "failed to create instance: " << res << std::endl;
        return false;
    }
    return true;
}

bool VulkanRenderer::create_surface() {
    engine::window::Window* win = reinterpret_cast<engine::window::Window*>(
		window_ptr_
	);

    if (!win) return false;
    if (!win->create_vulkan_surface(instance_, &surface_)) {
        std::cerr << "Window backend failed to create Vulkan surface" << std::endl;
        return false;
    }
    return true;
}

bool VulkanRenderer::check_device_extensions(VkPhysicalDevice device) {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, available.data());

    std::set<std::string> required(device_extensions_.begin(), device_extensions_.end());
    for (const auto& ext : available) {
        required.erase(ext.extensionName);
    }
    return required.empty();
}

int VulkanRenderer::rate_device_suitability(VkPhysicalDevice device) {
    //simple heuristic: prefer discrete GPUs

    VkPhysicalDeviceProperties prop;
    vkGetPhysicalDeviceProperties(device, &prop);
    if (prop.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return 1000;
    if (prop.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return 500;
    return 0;
}

bool VulkanRenderer::pick_physical_device() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "failed to find GPUs with Vulkan support!" << std::endl;
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    int bestScore = -1;
	VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
    for (const auto& dev : devices) {
        if (!check_device_extensions(dev)) continue;

        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, qprops.data());

        bool hasGraphics = false;
        bool hasPresent = false;
        for (uint32_t i = 0; i < qcount; ++i) {
            if(qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) hasGraphics = true;
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_, &present);
            if (present) hasPresent = true;
        }
        if (!hasGraphics || !hasPresent) continue;

        int score = rate_device_suitability(dev);
        if (score > bestScore) {
            bestScore = score;
            bestDevice = dev;
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        std::cerr << "failed to find a suitable GPU" << std::endl;
        return false;
    }

    physical_device_ = bestDevice;
    return true;
}

bool VulkanRenderer::create_logical_device() {
    // find queue families
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qcount, qprops.data());

    int graphicsFamily = -1;
    int presentFamily = -1;
    for (uint32_t i = 0; i < qcount; ++i) {
        if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && graphicsFamily == -1) {
            graphicsFamily = (int)i;
        }
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, i, surface_, &present);
        if (present && presentFamily == -1) {
            presentFamily = (int)i;
        }
    }
    if (graphicsFamily == -1 || presentFamily == -1) {
        std::cerr << "failed to find required queue families" << std::endl;
        return false;
    }

    std::vector<uint32_t> uniqueFamilies;
	//= { (uint32_t)graphicsFamily, (uint32_t)presentFamily };
	
	uniqueFamilies.push_back((uint32_t)graphicsFamily);
	if(presentFamily != graphicsFamily) uniqueFamilies.push_back((uint32_t)presentFamily);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(qci);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = static_cast<uint32_t>(device_extensions_.size());
    createInfo.ppEnabledExtensionNames = device_extensions_.data();

    if (vkCreateDevice(physical_device_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
        std::cerr << "failed to create logical device!" << std::endl;
        return false;
    }

    vkGetDeviceQueue(device_, uniqueFamilies[0], 0, &graphics_queue_);
    vkGetDeviceQueue(device_, uniqueFamilies.back(), 0, &present_queue_);

    return true;
}

bool VulkanRenderer::create_swapchain() {
    // query surface capabilities
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &formatCount, formats.data());

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &presentModeCount, presentModes.data());

    VkSurfaceFormatKHR surfaceFormat = formats[0];
    for (const auto& availableFormat : formats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surfaceFormat = availableFormat;
            break;
        }
    }

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR; // guaranteed
    for (const auto& availablePresentMode : presentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            presentMode = availablePresentMode;
            break;
        }
    }

    VkExtent2D extent;
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent = capabilities.currentExtent;
    } else {
        extent.width = static_cast<uint32_t>(width_);
        extent.height = static_cast<uint32_t>(height_);
        extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, extent.width));
        extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, extent.height));
    }

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // find queue family indices again
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qcount, qprops.data());

    int graphicsFamily = -1;
    int presentFamily = -1;
    for (uint32_t i = 0; i < qcount; ++i) {
        if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && graphicsFamily == -1) graphicsFamily = (int)i;
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, i, surface_, &present);
        if (present && presentFamily == -1) presentFamily = (int)i;
    }

    if (graphicsFamily != presentFamily) {
        uint32_t queueFamilyIndices[] = { (uint32_t)graphicsFamily, (uint32_t)presentFamily };
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_) != VK_SUCCESS) {
        std::cerr << "failed to create swap chain!" << std::endl;
        return false;
    }

    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
    swapchain_images_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchain_images_.data());

    swapchain_image_format_ = surfaceFormat.format;
    swapchain_extent_ = extent;

    return true;
}

bool VulkanRenderer::create_image_views() {
    swapchain_image_views_.resize(swapchain_images_.size());
    for (size_t i = 0; i < swapchain_images_.size(); ++i) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapchain_images_[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapchain_image_format_;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device_, &createInfo, nullptr, &swapchain_image_views_[i]) != VK_SUCCESS) {
            std::cerr << "failed to create image views!" << std::endl;
            return false;
        }
    }
    return true;
}

bool VulkanRenderer::create_render_pass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchain_image_format_;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device_, &renderPassInfo, nullptr, &render_pass_) != VK_SUCCESS) {
        std::cerr << "failed to create render pass!" << std::endl;
        return false;
    }
    return true;
}

bool VulkanRenderer::create_framebuffers() {
    frame_buffers_.resize(swapchain_image_views_.size());

    for (size_t i = 0; i < swapchain_image_views_.size(); ++i) {
        VkImageView attachments[] = { swapchain_image_views_[i] };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = render_pass_;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapchain_extent_.width;
        framebufferInfo.height = swapchain_extent_.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &frame_buffers_[i]) != VK_SUCCESS) {
            std::cerr << "failed to create framebuffer!" << std::endl;
            return false;
        }
    }

    return true;
}

bool VulkanRenderer::create_command_pool_and_buffers() {
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qcount, qprops.data());

    int graphicsFamily = -1;
    for (uint32_t i = 0; i < qcount; ++i) {
        if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphicsFamily = (int)i;
            break;
        }
    }
    if (graphicsFamily < 0) return false;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphicsFamily;

    if(vkCreateCommandPool(device_, &poolInfo, nullptr, &command_pool_) != VK_SUCCESS) {
        std::cerr << "failed to create command pool!" << std::endl;
        return false;
    }

    // allocate one command buffer per swapchain image
    command_buffers_.resize(swapchain_image_views_.size());
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = command_pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(command_buffers_.size());

    if (vkAllocateCommandBuffers(device_, &allocInfo, command_buffers_.data()) != VK_SUCCESS) {
        std::cerr << "failed to allocate command buffers!" << std::endl;
        return false;
    }

    return true;
}

bool VulkanRenderer::create_sync_objects() {
    image_available_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    render_finished_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &image_available_semaphores_[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &render_finished_semaphores_[i]) != VK_SUCCESS ||
            vkCreateFence(device_, &fenceInfo, nullptr, &in_flight_fences_[i]) != VK_SUCCESS) {
            std::cerr << "failed to create synchronization objects for a frame!" << std::endl;
            return false;
        }
    }
    return true;
}

void VulkanRenderer::record_command_buffers() {
    for (size_t i = 0; i < command_buffers_.size(); ++i) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        vkResetCommandBuffer(command_buffers_[i], 0);
        vkBeginCommandBuffer(command_buffers_[i], &beginInfo);

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = render_pass_;
        renderPassInfo.framebuffer = frame_buffers_[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchain_extent_;

        VkClearValue clearColor = { {{0.1f, 0.1f, 0.12f, 1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(command_buffers_[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // currently no drawing commands .. just clear the framebuffer.
        // later: bind pipelines, vertex buffers, draw calls here.

        vkCmdEndRenderPass(command_buffers_[i]);
        if (vkEndCommandBuffer(command_buffers_[i]) != VK_SUCCESS) {
            std::cerr << "failed to record command buffer!" << std::endl;
        }
    }
}


} //namespace render
