#include<cstring>

#include<core/memory/allocator.hpp>
//#include<../src/core/memory/allocator.hpp>

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

	EXPECT_GE(arena.in_use(), 16 + 32);

	void* big = arena.allocate(buff_size, alignof(std::max_align_t));
	EXPECT_EQ(big, nullptr);

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
