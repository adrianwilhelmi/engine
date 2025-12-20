#pragma once

#include"allocator_utils.hpp"
#include"page_allocator.hpp"

namespace engine::mem::allocator{

class LinearArena{
public:
	LinearArena(void* external_buffer, const std::size_t bytes) noexcept;
	explicit LinearArena(PageAllocator& backing, const std::size_t bytes) noexcept;
	~LinearArena() noexcept;

	LinearArena(const LinearArena&) = delete;
	LinearArena& operator=(const LinearArena&) = delete;

	LinearArena(LinearArena&& other) noexcept;
	LinearArena& operator=(LinearArena&& other) = delete;

	[[nodiscard]] void* allocate(const std::size_t size, const std::size_t alignment = alignof(std::max_align_t)) noexcept;
	void reset() noexcept;
	[[nodiscard]] std::size_t in_use() const {return offset_;}
	[[nodiscard]] std::size_t capacity() const {return capacity_;}

private:
	std::byte* buffer_ = nullptr;
	std::size_t capacity_ = 0;
	std::size_t offset_ = 0;

	PageAllocator* backing_allocator_ = nullptr;
};
static_assert(utils::ArenaLike<LinearArena>);

} // namespace engine::mem::allocator
