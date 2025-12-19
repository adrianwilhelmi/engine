#pragma once

#include<algorithm>
#include<mutex>

#include"virtual_memory.hpp"
#include"allocator_utils.hpp"

namespace engine::mem::allocator{

class PageAllocator{
public:
	PageAllocator() = default;
	~PageAllocator() { shutdown(); }

	PageAllocator(const PageAllocator&) = delete;
	PageAllocator& operator=(const PageAllocator&) = delete;

	void init(std::size_t max_size_bytes);
	void shutdown();

	[[nodiscard]] void*allocate(std::size_t size, std::size_t alignment);
	void deallocate(void* ptr, std::size_t size);

	void reset(bool decommit_unused = false);

	std::size_t committed_bytes() const {return committed_head_;}
	std::size_t reserved_bytes() const {return reserved_size_;}

private:
	void* base_ptr_ = nullptr;
	std::size_t reserved_size_ = 0;
	std::size_t current_offset_ = 0;
	std::size_t committed_head_ = 0;
	std::size_t page_size_ = 0;
	std::mutex mutex_;
};
static_assert(utils::AllocatorLike<PageAllocator>);

} //namespace engine::mem::allocator
