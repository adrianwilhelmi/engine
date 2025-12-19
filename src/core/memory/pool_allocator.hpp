#pragma once

#include"allocator_utils.hpp"
#include"page_allocator.hpp"

namespace engine::mem::allocator{

class PoolAllocator{
public:
	PoolAllocator(PageAllocator& backing,
				const std::size_t elem_size, 
				const std::size_t count, 
				std::size_t alignment = alignof(std::max_align_t));

	~PoolAllocator() noexcept;
	PoolAllocator(const PoolAllocator&) = delete;

	[[nodiscard]] void* allocate(const std::size_t size, const std::size_t align);
	void deallocate(void* p) noexcept;
	void reset() noexcept;

	std::size_t capacity() const noexcept {return capacity_count_;}
	std::size_t free_count() const noexcept;

private:
	void init_free_list() noexcept;

	std::byte* memory_ = nullptr;
	void* free_head_ = nullptr;

	std::size_t elem_size_ = 0;
	std::size_t capacity_count_ = 0;
	std::size_t alignment_ = 0;
	std::size_t total_bytes_ = 0;

	PageAllocator* backing_allocator_ = nullptr;
};
static_assert(utils::PoolLike<PoolAllocator>);

} // namespace engine::mem::allocator
