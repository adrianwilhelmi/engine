#include<cassert>
#include<algorithm>

#include"pool_allocator.hpp"
#include"page_allocator.hpp"
#include"allocator_utils.hpp"

namespace engine::mem::allocator{

PoolAllocator::PoolAllocator(
			PageAllocator& backing,
			const std::size_t elem_size, 
			const std::size_t count, 
			std::size_t alignment)
		: elem_size_(std::max(elem_size, sizeof(void*))),
		capacity_count_(count),
		alignment_(alignment),
		backing_allocator_(&backing){
	assert(alignment >= alignof(void*) && 
		"alignment must be at least pointer size");
	assert(count > 0 && "pool count must be > 0");

	std::size_t stride = utils::align_up(elem_size_, alignment_);
	total_bytes_ = stride * count;
	memory_ = static_cast<std::byte*>(
		backing.allocate(total_bytes_, alignment)
	);

	assert(memory_ && "failed to allocate pool memory");

	init_free_list();
}

void PoolAllocator::init_free_list() noexcept{
	if(!memory_) return;

	std::size_t stride = utils::align_up(elem_size_, alignment_);

	for(std::size_t i = 0; i < capacity_count_ - 1; ++i){
		void* curr = utils::ptr_add<void>(memory_, i * stride);
		void* next = utils::ptr_add<void>(memory_, (i+1) * stride);
		*static_cast<void**>(curr) = next;
	}

	void* last = utils::ptr_add<void>(memory_, (capacity_count_ - 1)*stride);
	*static_cast<void**>(last) = nullptr;

	free_head_ = memory_;
}

PoolAllocator::~PoolAllocator() noexcept{
	if(backing_allocator_ && memory_){
		backing_allocator_->deallocate(memory_, total_bytes_);
	}
}

void* PoolAllocator::allocate(
		const std::size_t size, 
		const std::size_t align){
	(void)align;

	assert(size <= elem_size_ && "object too large for this pool");

	if(!free_head_) return nullptr;

	void*r = free_head_;
	//move head to next element from list
	free_head_ = *reinterpret_cast<void**>(free_head_);
	return r;
}

void PoolAllocator::deallocate(void* p) noexcept{
	if(!p) return;
	*static_cast<void**>(p) = free_head_;
	free_head_ = p;
}

void PoolAllocator::reset() noexcept { 
	init_free_list(); 
}

std::size_t PoolAllocator::free_count() const noexcept{
	std::size_t cnt = 0;
	void* cur = free_head_;
	while(cur) {
		++cnt;
		cur = *reinterpret_cast<void**>(cur);
	}
	return cnt;
}

}// namespace engine::mem::allocator
