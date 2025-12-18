#include<cstdlib>
#include<string>
#include<algorithm>
#include<cassert>

#include"allocator.hpp"

namespace engine::allocator{

#if defined(WIN32) || defined(_WIN64)
	#include<malloc.h>
	static inline void* os_aligned_alloc(
			const std::size_t alignment, 
			const std::size_t size){
		return _aligned_malloc(size, alignment);
	}

	static inline void os_aligned_free(void*p) {_aligned_free(p); }

	std::string detected_os(){
		return "win";
	}
#elif defined(__linux__) || defined(__unix__)
	static inline void* os_aligned_alloc(
			std::size_t alignment, 
			const std::size_t size){
		void*ptr = nullptr;
		if(alignment < sizeof(void*)) alignment = sizeof(void*);
		int r = posix_memalign(&ptr, alignment, align_up(size, alignment));
		return r == 0 ? ptr : nullptr;
	}

	static inline void os_aligned_free(void*p) { std::free(p); }

	std::string detected_os(){
		return "lin";
	}
#else
	static inline void* os_aligned_alloc(
			std::size_t alignment, 
			std::size_t size){ 
		std::size_t s = size; 
		if(s % alignment != 0){ 
			s = ((s + alignment - 1) / alignment) * alignment; 
		} 
		return std::aligned_alloc(alignment, s); } 

	static inline void os_aligned_free(void*p) { std::free(p); }

	std::string detected_os(){
		return "other";
	}
#endif


void* DefaultHeap::allocate(
		const std::size_t size, 
		const std::size_t alignment) noexcept{
	void*p = os_aligned_alloc(alignment, size);
	if(p) ++allocs_;
	return p;
}

void DefaultHeap::deallocate(void* p) noexcept{
	os_aligned_free(p);
}

LinearArena::LinearArena(void* external_buffer, const std::size_t bytes) noexcept
	: buffer_(static_cast<std::byte*>(external_buffer)), 
	capacity_(bytes),
	offset_(0),
	owns_memory_(false){}

LinearArena::LinearArena(const std::size_t bytes) noexcept
		: capacity_(bytes), offset_(0), owns_memory_(true){
	buffer_ = static_cast<std::byte*>(DefaultHeap{}.allocate(bytes));
}

LinearArena::~LinearArena() noexcept{
	if(owns_memory_ && buffer_) DefaultHeap{}.deallocate(buffer_);
}

void* LinearArena::allocate(
			const std::size_t size, 
			const std::size_t alignment) noexcept{
	if(!buffer_) return nullptr;

	std::uintptr_t base = reinterpret_cast<std::uintptr_t>(buffer_);
	std::uintptr_t cur_ptr = base + offset_;

	const std::size_t mask = alignment - 1;
	std::size_t padding = 0;
	if(cur_ptr & mask){
		padding = alignment - (cur_ptr & mask);
	}

	if(offset_ + padding + size > capacity_) return nullptr;

	offset_ += padding;
	void*result = buffer_ + offset_;
	offset_ += size;

	return result;
}

void LinearArena::reset() noexcept {offset_ = 0; }

std::size_t LinearArena::in_use() const {return offset_;}

PoolAllocator::PoolAllocator(
			void* buf, 
			const std::size_t elem_size, 
			const std::size_t count, 
			std::size_t alignment)
		: memory_(static_cast<std::byte*>(buf)),
		capacity_count_(count),
		alignment_(alignment) {

	assert(alignment >= alignof(void*) && 
		"alignment must be at least pointer size");

	const std::size_t min_size = (elem_size > sizeof(void*)) ? 
		elem_size : sizeof(void*);
	elem_size_ = align_up(min_size, alignment);

	init_free_list();
}

void PoolAllocator::init_free_list() noexcept{
	if(!memory_) {
		free_head_ = nullptr; 
		return;
	}
	free_head_ = memory_;
	std::byte* ptr = memory_;
	for(std::size_t i = 0; i < capacity_count_ - 1; ++i){
		void* next = ptr + elem_size_;
		*reinterpret_cast<void**>(ptr) = next;
		ptr += elem_size_;
	}
	*reinterpret_cast<void**>(ptr) = nullptr;
}

void* PoolAllocator::allocate(
		const std::size_t size, 
		const std::size_t align){
	assert(size <= elem_size_ && "object too large for this pool");
	assert(align <= alignment_ && "too strict alignment required");

	if(!free_head_) return nullptr;

	void*r = free_head_;
	free_head_ = *reinterpret_cast<void**>(free_head_);
	return r;
}

void PoolAllocator::deallocate(void* p) noexcept{
	if(!p) return;
	*reinterpret_cast<void**>(p) = free_head_;
	free_head_ = p;
}

void PoolAllocator::reset() noexcept { 
	init_free_list(); 
}

std::size_t PoolAllocator::free_count() const noexcept{
	std::size_t cnt = 0;
	void* cur = free_head_;
	while(cur) {++cnt; cur = *reinterpret_cast<void**>(cur);}
	return cnt;
}

} // namespace engine::allocator
