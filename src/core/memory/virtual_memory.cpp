#include<cstdlib>

#include"virtual_memory.hpp"
#include"allocator_utils.hpp"

using namespace engine::mem;

#if defined(WIN32) || defined(_WIN64)
	#define WIN32_LEAN_AND_MEAN
	#include<windows.h>
	#include<malloc.h>
#elif defined(__linux__) || defined(__unix__)
	#include<sys/mman.h>
	#include<unistd.h>
	#include<errno.h>
	#include<stdlib.h>
#endif

namespace engine::mem::os{
std::size_t VirtualMemory::get_page_size(){
#if defined(WIN32) || defined(_WIN64)
	SYSTEM_INFO si;
	GetSystemInfo(&si);
	return si.dwPageSize;
#elif defined(__linux__) || defined(__unix__)
	return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#else
	return 0;
#endif
}

void* VirtualMemory::reserve(std::size_t size){
#if defined(WIN32) || defined(_WIN64)
	return VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
#elif defined(__linux__) || defined(__unix__)
	void*ptr = mmap(nullptr, size, PROT_NONE,
			MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	return ptr == MAP_FAILED ? nullptr : ptr;
#else
	return nullptr;
#endif
}

bool VirtualMemory::commit(void* ptr, std::size_t size){
#if defined(WIN32) || defined(_WIN64)
	void*result = VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE);
	return result != nullptr;
#elif defined(__linux__) || defined(__unix__)
	int result = mprotect(ptr, size, PROT_READ | PROT_WRITE);
	return result == 0;
#else
	return false;
#endif
}

void VirtualMemory::decommit(void* ptr, std::size_t size){
#if defined(WIN32) || defined(_WIN64)
	VirtualFree(ptr, size, MEM_DECOMMIT);
#elif defined(__linux__) || defined(__unix__)
	madvise(ptr, size, MADV_DONTNEED);
	mprotect(ptr, size, PROT_NONE);
#endif
}

void VirtualMemory::release(void* ptr, std::size_t size){
#if defined(WIN32) || defined(_WIN64)
	(void)size;
	VirtualFree(ptr, 0, MEM_RELEASE);
#elif defined(__linux__) || defined(__unix__)
	munmap(ptr, size);
#endif
}

void* VirtualMemory::os_aligned_alloc(
		const std::size_t alignment, 
		const std::size_t size){
#if defined(WIN32) || defined(_WIN64)
	return _aligned_malloc(size, alignment);
#elif defined(__linux__) || defined(__unix__)
	void*ptr = nullptr;
	if(alignment < sizeof(void*)) alignment = sizeof(void*);
	int r = posix_memalign(&ptr, alignment, utils::align_up(size, alignment));
	return r == 0 ? ptr : nullptr;
#else
	std::size_t s = utils::align_up(size, alignment);
	return std::aligned_alloc(alignment, s); 
#endif
} 

void VirtualMemory::os_aligned_free(void*p) {
#if defined(WIN32) || defined(_WIN64)
	_aligned_free(p);
#else
	std::free(p);
#endif
}

std::string VirtualMemory::detected_os(){
#if defined(WIN32) || defined(_WIN64)
		return "win";
#elif defined(__linux__) || defined(__unix__)
		return "lin";
#else
		return "other";
#endif
}

}	// namespace engine::mem::os
