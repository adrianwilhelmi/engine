#include<cstring>

#include<core/memory/allocator.hpp>

#include<gtest/gtest.h>

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
	LinearArena arena(1024);
	arena.allocate(1);

	void*p = arena.allocate(100,64);
	ASSERT_NE(p,nullptr);
	std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
	EXPECT_EQ(addr % 64, 0);
}

TEST(LinearArenaTest, ReturnsNullOnOOM){
	LinearArena arena(100);
	void*p = arena.allocate(200);
	EXPECT_EQ(p,nullptr);
}

TEST(PoolAllocatorTest, ReusesMemory){
	std::byte buffer[200];
	PoolAllocator pool(buffer, 32, 2);

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

	LinearArena arena(1024);
	auto handle = AllocatorHandle::from_arena(arena);

	SpyObject* obj = alloc_new<SpyObject>(handle);
	ASSERT_NE(obj, nullptr);
	EXPECT_EQ(SpyObject::constructions, 1);
	EXPECT_EQ(SpyObject::destructions, 0);

	free_delete(handle, obj);
	EXPECT_EQ(SpyObject::constructions, 1);
	EXPECT_EQ(SpyObject::destructions, 1);
}
