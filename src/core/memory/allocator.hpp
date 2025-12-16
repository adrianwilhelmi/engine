#pragma once

#include<string>
#include<cstddef>
#include<cstdint>
#include<cstdlib>
#include<new>
#include<type_traits>
#include<utility>
#include<cassert>
#include<optional>
#include<bit>

constexpr std::size_t align_up(const std::size_t size,
		std::size_t alignment) noexcept{
	return (size + alignment - 1) & ~(alignment - 1);
}

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
	DefaultHeap() noexcept = default;
	std::size_t allocs_ = 0;

	void* allocate(
			const std::size_t size,
			const std::size_t alignment = alignof(std::max_align_t)) noexcept{
		void*p = os_aligned_alloc(alignment, size);
		if(p) ++allocs_;
		return p;
	}

	void deallocate(void* p) noexcept{
		os_aligned_free(p);
	}
};
static_assert(AllocatorLike<DefaultHeap>);


class LinearArena{
public:
	LinearArena(void* external_buffer, const std::size_t bytes) noexcept
			: buffer_(static_cast<std::byte*>(external_buffer)), 
			capacity_(bytes),
			offset_(0),
			owns_memory_(false)
		{}

	explicit LinearArena(const std::size_t bytes) noexcept
			: capacity_(bytes), offset_(0), owns_memory_(true){
		buffer_ = static_cast<std::byte*>(DefaultHeap{}.allocate(bytes));
	}

	~LinearArena() noexcept{
		if(owns_memory_ && buffer_) DefaultHeap{}.deallocate(buffer_);
	}

	LinearArena(const LinearArena&) = delete;
	LinearArena& operator=(const LinearArena&) = delete;
	LinearArena(LinearArena&& other) noexcept
			: buffer_(other.buffer_), 
			capacity_(other.capacity_), 
			offset_(other.offset_),
			owns_memory_(other.owns_memory_){
		other.buffer_ = nullptr;
		other.owns_memory_ = false;
	}

	void* allocate(
			const std::size_t size, 
			const std::size_t alignment = alignof(std::max_align_t)) noexcept{
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

	inline void reset() noexcept {offset_ = 0; }

	std::size_t in_use() const {return offset_;}

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
				std::size_t alignment = alignof(std::max_align_t))
			: memory_(static_cast<std::byte*>(buf)), 
			elem_size_((elem_size + sizeof(void*) - 1) & ~(sizeof(void*) - 1)),
			capacity_count_(count){
		assert(alignment >= alignof(void*) && 
				"alignment must be at least pointer size");
		const std::size_t min_size = (elem_size > sizeof(void*)) ? 
			elem_size : sizeof(void*);
		elem_size_ = align_up(min_size, alignment);

		init_free_list();
	}

	PoolAllocator(const PoolAllocator&) = delete;

	void init_free_list() noexcept{
		if(!memory_) { free_head_ = nullptr; return; }
		free_head_ = memory_;
		std::byte* ptr = memory_;
		for(std::size_t i = 0; i < capacity_count_ - 1; ++i){
			void* next = ptr + elem_size_;
			*reinterpret_cast<void**>(ptr) = next;
			ptr += elem_size_;
		}
		*reinterpret_cast<void**>(ptr) = nullptr;
	}

	void* allocate(
			const std::size_t size, 
			const std::size_t align){
		assert(size <= elem_size_ && "object too large for this pool");
		assert(align <= alignment_ && "too strict alignment required");

		if(!free_head_) return nullptr;

		void*r = free_head_;
		free_head_ = *reinterpret_cast<void**>(free_head_);
		return r;
	}

	void deallocate(void* p) noexcept{
		if(!p) return;
		*reinterpret_cast<void**>(p) = free_head_;
		free_head_ = p;
	}

	inline void reset() noexcept { init_free_list(); }

	inline std::size_t capacity() const noexcept {return capacity_count_;}
	inline std::size_t free_count() const noexcept{
		std::size_t cnt = 0;
		void* cur = free_head_;
		while(cur) {++cnt; cur = *reinterpret_cast<void**>(cur);}
		return cnt;
	}

private:
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

	AllocatorHandle() noexcept = default;

	template<typename T>
	static AllocatorHandle fromHeap(T& allocator){
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
	static AllocatorHandle fromArena(T& allocator){
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
	static AllocatorHandle fromPool(T& allocator){
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
		if(can_free) free_fn(impl, ptr);
	}

	void reset() const{
		if(can_reset) reset_fn(impl);
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
