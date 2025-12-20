#pragma once

#include<new>

#include"allocator_utils.hpp"

namespace engine::mem::allocator{

struct DefaultHeap{
	std::size_t allocs_ = 0;

	DefaultHeap() noexcept = default;

	[[nodiscard]] void* allocate(
			const std::size_t size,
			const std::size_t alignment = alignof(std::max_align_t)) noexcept;
	void deallocate(void* p) noexcept;
};
static_assert(utils::AllocatorLike<DefaultHeap>);

} // namespace engine::mem::allocator
