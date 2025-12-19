#pragma once

#include"allocator_utils.hpp"

namespace engine::mem::allocator{

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

	[[nodiscard]] void* allocate(
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

template<typename T, typename ...Args>
T* alloc_new(AllocatorHandle& a, Args&&... args){
	void* mem = a.allocate(sizeof(T), alignof(T));
	if(!mem) throw std::bad_alloc();
	return new (mem) T(std::forward<Args>(args)...);
}

template<typename T>
void free_delete(AllocatorHandle& a, T* obj) noexcept{
	if(!obj) return;
	obj->~T();
	a.deallocate(static_cast<void*>(obj));
}

} // namespace engine::mem::allocator
