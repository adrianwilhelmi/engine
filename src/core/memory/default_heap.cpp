#include<new>

#include"default_heap.hpp"
#include"allocator_utils.hpp"
#include"virtual_memory.hpp"

namespace engine::mem::allocator{

void* DefaultHeap::allocate(
		const std::size_t size, 
		const std::size_t alignment) noexcept{
	void*p = engine::mem::os::VirtualMemory::os_aligned_alloc(alignment, size);
	if(p) ++allocs_;
	return p;
}

void DefaultHeap::deallocate(void* p) noexcept{
	VirtualMemory::os_aligned_free(p);
}

} // namespace engine::mem::allocator
