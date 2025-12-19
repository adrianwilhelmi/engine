#pragma once

#include<cstddef>
#include<cstdint>
#include<concepts>
#include<type_traits>

namespace engine::mem::utils{

constexpr std::size_t align_up(const std::size_t size,
		const std::size_t alignment) noexcept{
	return (size + alignment - 1) & ~(alignment - 1);
}

template<typename T>
inline T* ptr_add(void* p, std::size_t bytes){
	return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(p) + bytes);
}

template<typename T>
inline std::size_t ptr_diff(void*end, void*start){
	return reinterpret_cast<std::uintptr_t>(end) -
		reinterpret_cast<std::uintptr_t>(start);
}

template<typename T>
concept AllocatorLike = requires(
		T& a, 
		const std::size_t n, 
		const std::size_t al, 
		void*p){
	{a.allocate(n, al) } -> std::convertible_to<void*>;
};

template<typename T>
concept ArenaLike = AllocatorLike<T> && requires(T& a) {a.reset();};

template<typename T>
concept PoolLike = AllocatorLike<T> && requires(T& p) { 
	{p.capacity() } -> std::convertible_to<std::size_t>; 
};

} // namespace engine::mem
