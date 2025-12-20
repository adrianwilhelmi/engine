#include<cassert>
#include<cstdint>

#include"linear_arena.hpp"
#include"page_allocator.hpp"

namespace engine::mem::allocator{

LinearArena::LinearArena(void* external_buffer, const std::size_t bytes) noexcept
	: buffer_(static_cast<std::byte*>(external_buffer)), 
	capacity_(bytes),
	offset_(0),
	backing_allocator_(nullptr){}

LinearArena::LinearArena(PageAllocator& backing, const std::size_t bytes) noexcept
		: capacity_(bytes), offset_(0), backing_allocator_(&backing){
	buffer_ = static_cast<std::byte*>(backing.allocate(bytes, 16));
	assert(buffer_ 
			&& "failed to allocate memory for LinearArena from PageAllocator");
}

LinearArena::~LinearArena() noexcept{
	if(backing_allocator_ && buffer_){
		backing_allocator_->deallocate(buffer_, capacity_);
	}
}

LinearArena::LinearArena(LinearArena&& other) noexcept
		: buffer_(other.buffer_),
		capacity_(other.capacity_),
		offset_(other.offset_),
		backing_allocator_(other.backing_allocator_){
	other.backing_allocator_ = nullptr;
	other.buffer_ = nullptr;
	other.capacity_ = 0;
	other.offset_ = 0;
}


void* LinearArena::allocate(
			const std::size_t size, 
			const std::size_t alignment) noexcept{
	if(!buffer_) return nullptr;

	std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(buffer_);
	std::uintptr_t current_addr = base_addr + offset_;

	std::uintptr_t aligned_addr = (current_addr + alignment - 1)
		& ~(alignment - 1);

	std::size_t padding = aligned_addr - current_addr;
	std::size_t total_req = padding + size;

	if(offset_ + total_req > capacity_){
		return nullptr;
	}

	offset_ += total_req;
	return reinterpret_cast<void*>(aligned_addr);
}

void LinearArena::reset() noexcept {offset_ = 0; }

}// namespace engine::mem::allocator
