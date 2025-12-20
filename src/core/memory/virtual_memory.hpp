#pragma once

#include<cstddef>
#include<string>

namespace engine::mem::os{

struct VirtualMemory{
	[[nodiscard]] static std::size_t get_page_size();

	//reserve virtual mem
	[[nodiscard]] static void*reserve(std::size_t size);

	//commit RAM under reserved addr
	//	ptr must be a result of reserve function
	//	size must be divisable by pagesize
	[[nodiscard]] static bool commit(void*ptr, std::size_t size);

	//decommit RAM (back to system), ptr addr is still reserved
	static void decommit(void* ptr, std::size_t size);

	//release memory 
	//	ptr must be a result of reserve function
	//	size arg is needed only on linux
	static void release(void* ptr, std::size_t size);

	static void* os_aligned_alloc(std::size_t alignment, std::size_t size);

	static void os_aligned_free(void*p);

	[[nodiscard]] static std::string detected_os();
};

}	//namespace engine::mem::os
