#include<cstring>
#include<vector>
#include<algorithm>
#include<gtest/gtest.h>

#include<core/memory/default_heap.hpp>
#include<core/memory/linear_arena.hpp>
#include<core/memory/pool_allocator.hpp>
#include<core/memory/page_allocator.hpp>
#include<core/memory/allocator_handle.hpp>

#include<gtest/gtest.h>

using namespace engine::mem::allocator;

TEST(DefaultHeapTest, BasicAllocDealloc){
	const std::size_t maxa = alignof(std::max_align_t);
	const std::size_t buff_size = 1024;
	DefaultHeap heap;

	void* buff = nullptr;
	EXPECT_EQ(buff, nullptr);
	buff = heap.allocate(buff_size, maxa);
	EXPECT_EQ(heap.allocs_, 1);
	EXPECT_NE(buff, nullptr);

	heap.deallocate(buff);
}

TEST(DefaultHeapTest, AlignmentAndNullFree){
	DefaultHeap heap;
	void*p64 = heap.allocate(128,64);
	ASSERT_NE(p64,nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p64) % 64, 0u);

	void* p128 = heap.allocate(256,128);
	ASSERT_NE(p128,nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p128) % 128, 0u);

	heap.deallocate(nullptr);
	heap.deallocate(p64);
	heap.deallocate(p128);
}

TEST(LinearArenaTest, BasicAllocReset){
	const std::size_t buff_size = 1024;
	void* buffer = std::malloc(buff_size);
	ASSERT_NE(buffer, nullptr);
	LinearArena arena(buffer, buff_size);

	void*a = arena.allocate(16, alignof(std::max_align_t));
	EXPECT_NE(a, nullptr);
	void*b = arena.allocate(32, alignof(std::max_align_t));
	EXPECT_NE(b, nullptr);
	EXPECT_NE(a, b);
	EXPECT_LT(a, b);

	EXPECT_GE(arena.in_use(), 16 + 32);

	arena.reset();

	EXPECT_EQ(arena.in_use(), 0);

	void*c = arena.allocate(64, alignof(std::max_align_t));
	EXPECT_NE(c, nullptr);
	EXPECT_GE(arena.in_use(), 64);

	std::free(buffer);
}

TEST(LinearArenaTest, LinearArenaWithHeap){
	const std::size_t maxa = alignof(std::max_align_t);
	const std::size_t buff_size = 4 * 1024;
	DefaultHeap heap;

	void*buff = heap.allocate(buff_size, maxa);

	EXPECT_NE(buff, nullptr);

	LinearArena a1(buff, buff_size / 2);
	LinearArena a2(static_cast<std::byte*>(buff)+buff_size/2, buff_size / 2);

	EXPECT_EQ(a1.in_use(), 0);
	EXPECT_EQ(a2.in_use(), 0);
	EXPECT_EQ(heap.allocs_, 1);
}

TEST(LinearArenaTest, RespectsAlignment){
	PageAllocator backing;
	backing.init(4096);

	LinearArena arena(backing, 1024);
	arena.allocate(1);

	void*p = arena.allocate(100,64);
	ASSERT_NE(p,nullptr);
	std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
	EXPECT_EQ(addr % 64, 0);
}

TEST(LinearArenaTest, ReturnsNullOnOOM){
	PageAllocator backing;
	backing.init(4096);

	LinearArena arena(backing, 100);
	void*p = arena.allocate(200);
	EXPECT_EQ(p,nullptr);
}

TEST(LinearArenaTest, ReuseAfterReset){
	const std::size_t BUF = 1024;
	void*buffer = std::malloc(BUF);
	ASSERT_NE(buffer, nullptr);
	LinearArena arena(buffer, BUF);

	void*a1 = arena.allocate(64, alignof(std::max_align_t));
	ASSERT_NE(a1,nullptr);
	std::uintptr_t addr1 = reinterpret_cast<std::uintptr_t>(a1);

	arena.reset();

	void*a2 = arena.allocate(64, alignof(std::max_align_t));
	ASSERT_NE(a2, nullptr);
	std::uintptr_t addr2 = reinterpret_cast<std::uintptr_t>(a2);

	EXPECT_EQ(addr1, addr2);

	std::free(buffer);
}

TEST(LinearArenaTest, PointerOrderingViaUIntptr){
	PageAllocator backing;
	backing.init(4096);

	LinearArena arena(backing, 512);
	void*p1 = arena.allocate(8);
	void*p2 = arena.allocate(8);
	ASSERT_NE(p1,nullptr);
	ASSERT_NE(p2,nullptr);
	EXPECT_LT(reinterpret_cast<std::uintptr_t>(p1),
			reinterpret_cast<std::uintptr_t>(p2));
}

TEST(LinearArenaTest, ExactCapacityUsage){
	PageAllocator backing;
	backing.init(4096);

	LinearArena arena(backing, 100);

	void*p1 = arena.allocate(100,1);
	EXPECT_NE(p1,nullptr);
	EXPECT_EQ(arena.in_use(), 100);

	void*p2 = arena.allocate(1,1);
	EXPECT_EQ(p2,nullptr);
}

TEST(PoolAllocatorTest, ReusesMemory){
	PageAllocator backing;
	backing.init(4096);

	// elem_size = 32, count=2
	PoolAllocator pool(backing, 32, 2);

	void* p1 = pool.allocate(32,8);
	void* p2 = pool.allocate(32,8);
	void* p3 = pool.allocate(32,8);

	ASSERT_NE(p1,nullptr);
	ASSERT_NE(p2,nullptr);
	EXPECT_EQ(p3,nullptr);
	EXPECT_EQ(pool.free_count(),0);

	pool.deallocate(p1);
	EXPECT_EQ(pool.free_count(), 1);

	void*p4 = pool.allocate(32, 8);
	EXPECT_EQ(p4,p1);
}

TEST(PoolAllocatorTest, ResetRestoresPool){
	PageAllocator backing;
	backing.init(4096);

	const std::size_t elem = 32;
	const std::size_t count = 8;
	PoolAllocator pool(backing, elem, count);

	for(std::size_t i = 0; i < count; ++i){
		void*p = pool.allocate(elem, 1);
		ASSERT_NE(p,nullptr);
	}
	EXPECT_EQ(pool.free_count(), 0u);

	pool.reset();
	EXPECT_EQ(pool.free_count(), count);
}

TEST(PoolAllocatorTest, Stress){
	PageAllocator backing;
	backing.init(1000*64);

	const std::size_t count = 1000;
	PoolAllocator pool(backing, 32, count);
	std::vector<void*> items;
	items.reserve(count);
	for(std::size_t i = 0; i < count; ++i){
		void*p = pool.allocate(32, alignof(void*));
		ASSERT_NE(p,nullptr);
		items.push_back(p);
	}
	EXPECT_EQ(pool.free_count(), 0u);
	for(auto p : items) pool.deallocate(p);
	EXPECT_EQ(pool.free_count(), count);
}

TEST(PoolAllocatorTest, Stress2){
	PageAllocator backing;

	constexpr int pool_size = 100;
	backing.init(pool_size * 128);
	PoolAllocator pool(backing, 64, pool_size);

	std::vector<void*> ptrs;
	ptrs.reserve(pool_size);

	for(std::size_t i = 0; i < pool_size; ++i){
		void*p = pool.allocate(64,8);
		ASSERT_NE(p, nullptr);
		ptrs.push_back(p);
	}

	for(std::size_t i = 0; i < pool_size; i+=2){
		pool.deallocate(ptrs[i]);
		ptrs[i] = nullptr;
	}

	for(std::size_t i = 0; i < pool_size; i+=2){
		void*p = pool.allocate(64,8);
		ASSERT_NE(p,nullptr);
		ptrs[i] = p;
	}
}

TEST(PoolAllocatorTest, CorrectStridingWithAlignment){
	PageAllocator backing;
	backing.init(4096);

	PoolAllocator pool(backing,10,2,16);
	void*p1 = pool.allocate(10, 16);
	void*p2 = pool.allocate(10, 16);

	ASSERT_NE(p1, nullptr);
	ASSERT_NE(p2, nullptr);

	std::uintptr_t addr1 = reinterpret_cast<std::uintptr_t>(p1);
	std::uintptr_t addr2 = reinterpret_cast<std::uintptr_t>(p2);

	std::size_t distance = (addr2 > addr1) ? 
			(addr2 - addr1) : (addr1 - addr2);

	EXPECT_GE(distance,16u);
	EXPECT_EQ(distance % 16, 0u);
}

struct SpyObject{
	static int constructions;
	static int destructions;
	int value;

	SpyObject() : value(0) { constructions++; }
	SpyObject(int v) : value(v) { constructions++;}
	~SpyObject() {destructions++;}
};


int SpyObject::constructions = 0;
int SpyObject::destructions = 0;

TEST(AllocatorHandleTest, CallsConstructorAndDestructors){
	SpyObject::constructions = 0;
	SpyObject::destructions = 0;

	PageAllocator backing;
	backing.init(4096);
	LinearArena arena(backing, 1024);
	auto handle = AllocatorHandle::from_arena(arena);

	SpyObject* obj = alloc_new<SpyObject>(handle);
	ASSERT_NE(obj, nullptr);
	EXPECT_EQ(SpyObject::constructions, 1);
	EXPECT_EQ(SpyObject::destructions, 0);

	free_delete(handle, obj);
	EXPECT_EQ(SpyObject::constructions, 1);
	EXPECT_EQ(SpyObject::destructions, 1);
}

struct Entity{
	int x;
	int y;
	Entity(int x_val, int y_val) : x(x_val), y(y_val) {}
};

TEST(AllocatorHandleTest, ForwardsArguments){
	PageAllocator backing;
	backing.init(4096);
	LinearArena arena(backing, 1024);
	auto handle = AllocatorHandle::from_arena(arena);

	Entity* e = alloc_new<Entity>(handle, 10, 20);

	ASSERT_NE(e, nullptr);
	EXPECT_EQ(e->x, 10);
	EXPECT_EQ(e->y, 20);

	free_delete(handle,e);
}
