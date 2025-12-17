#pragma once

#include<cstddef>
#include<cstdint>
#include<new>
#include<utility>
#include<concepts>
#include<bit>

constexpr std::size_t align_up(const std::size_t size,
		std::size_t alignment) noexcept{
	return (size + alignment - 1) & ~(alignment - 1);
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

struct DefaultHeap{
	std::size_t allocs_ = 0;

	DefaultHeap() noexcept = default;

	void* allocate(
			const std::size_t size,
			const std::size_t alignment = alignof(std::max_align_t)) noexcept;
	void deallocate(void* p) noexcept;
};
static_assert(AllocatorLike<DefaultHeap>);

class LinearArena{
public:
	LinearArena(void* external_buffer, const std::size_t bytes) noexcept;
	explicit LinearArena(const std::size_t bytes) noexcept;
	~LinearArena() noexcept;

	LinearArena(const LinearArena&) = delete;
	LinearArena& operator=(const LinearArena&) = delete;
	LinearArena(LinearArena&& other) noexcept;
	LinearArena& operator=(LinearArena&& other) = delete;

	void* allocate(const std::size_t size, const std::size_t alignment = alignof(std::max_align_t)) noexcept;
	void reset() noexcept;
	std::size_t in_use() const;

private:
	std::byte* buffer_ = nullptr;
	std::size_t capacity_ = 0;
	std::size_t offset_ = 0;
	bool owns_memory_ = false;
};
static_assert(ArenaLike<LinearArena>);

class PoolAllocator{
public:
	PoolAllocator(void* buf,
				const std::size_t elem_size, 
				const std::size_t count, 
				std::size_t alignment = alignof(std::max_align_t));
	PoolAllocator(const PoolAllocator&) = delete;


	void* allocate(const std::size_t size, const std::size_t align);
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
};
static_assert(PoolLike<PoolAllocator>);


struct AllocatorHandle{
	void* impl = nullptr;
	void* (*alloc_fn)(void*, const std::size_t, const std::size_t) = nullptr;
	void (*free_fn)(void*, void*) = nullptr;
	void (*reset_fn)(void*) = nullptr;

	bool can_free = false;
	bool can_reset = false;

	template<typename T>
	static AllocatorHandle from_heap(T& allocator){
		return AllocatorHandle{
			&allocator,
			[](void*i, const std::size_t s,const std::size_t a){
				return static_cast<T*>(i)->allocate(s,a);
			},
			[](void*i, void*p){
				static_cast<T*>(i)->deallocate(p);
			},
			nullptr,
			true,
			false
		};
	}

	template<typename T>
	static AllocatorHandle from_arena(T& allocator){
		return AllocatorHandle{
			&allocator,
			[](void* i, const std::size_t s,const std::size_t a){
				return static_cast<T*>(i)->allocate(s,a);
			},
			[](void* /*i*/, void* /*p*/) {
				/**/
			},
			[](void* i) {static_cast<T*>(i)->reset();},
			false,
			true
		};
	}

	template<typename T>
	static AllocatorHandle from_pool(T& allocator){
		return AllocatorHandle{
			&allocator,
			[](void* i, const std::size_t s,const std::size_t a){
				return static_cast<T*>(i)->allocate(s,a);
			},
			[](void* i, void* p) {
				static_cast<T*>(i)->deallocate(p);
			},
			[](void* i) {static_cast<T*>(i)->reset();},
			true,
			true
		};
	}

	void* allocate(
			const std::size_t size, 
			const std::size_t align = alignof(std::max_align_t)) const{
		return alloc_fn(impl, size, align);
	}

	void deallocate(void* ptr) const{
		if(can_free && free_fn) free_fn(impl, ptr);
	}

	void reset() const{
		if(can_reset && reset_fn) reset_fn(impl);
	}
};

template<typename T>
T* alloc_new(AllocatorHandle& a){
	void* mem = a.allocate(sizeof(T), alignof(T));
	if(!mem) throw std::bad_alloc();
	return new (mem) T();
}

template<typename T>
void free_delete(AllocatorHandle& a, T* obj) noexcept{
	if(!obj) return;
	obj->~T();
	a.deallocate(static_cast<void*>(obj));
}
