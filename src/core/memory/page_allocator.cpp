#include<stdexcept>
#include<cassert>

#include"page_allocator.hpp"
#include"virtual_memory.hpp"

namespace engine::mem::allocator{

using engine::mem::os::VirtualMemory;

void PageAllocator::init(std::size_t max_size_bytes){
	page_size_ = VirtualMemory::get_page_size();

	reserved_size_ = utils::align_up(max_size_bytes, page_size_);
	base_ptr_ = VirtualMemory::reserve(reserved_size_);

	if(!base_ptr_){
		throw std::bad_alloc();
	}

	current_offset_ = 0;
	committed_head_ = 0;
}

void* PageAllocator::allocate(std::size_t size, std::size_t alignment){
	std::lock_guard<std::mutex> lock(mutex_);

	std::size_t base_addr = reinterpret_cast<std::size_t>(
		base_ptr_
	);

	std::size_t current_addr = base_addr + current_offset_;
	std::size_t aligned_addr = utils::align_up(current_addr, alignment);

	std::size_t padding = aligned_addr - current_addr;
	std::size_t new_offset = current_offset_ + padding + size;

	if(new_offset > reserved_size_)
		return nullptr;

	if(new_offset > committed_head_){
		std::size_t needed = new_offset - committed_head_;
		std::size_t pages_needed = utils::align_up(needed, page_size_);

		void* commit_ptr = utils::ptr_add<void>(base_ptr_, committed_head_);

		if(!VirtualMemory::commit(commit_ptr, pages_needed)){
			return nullptr;
		}

		committed_head_ += pages_needed;
	}

	void*result = reinterpret_cast<void*>(aligned_addr);
	current_offset_ = new_offset;

	return result;
}

void PageAllocator::deallocate(void*ptr, std::size_t size){
	//LIFO
	
	std::lock_guard<std::mutex> lock(mutex_);

	if(!ptr || size == 0) return;

	std::size_t addr = reinterpret_cast<std::size_t>(ptr);
	std::size_t base = reinterpret_cast<std::size_t>(base_ptr_);

	if(addr < base || addr >= base + reserved_size_) return;

	std::size_t chunk_end_offset = (addr + size) - base;
	if(chunk_end_offset == current_offset_){
		current_offset_ = addr - base;
	}
}

void PageAllocator::shutdown(){
	if(base_ptr_){
		VirtualMemory::release(base_ptr_, reserved_size_);
		base_ptr_ = nullptr;
		reserved_size_ = 0;
		committed_head_ = 0;
		current_offset_ = 0;
	}
}

void PageAllocator::reset(bool decommit_unused){
	std::lock_guard<std::mutex> lock(mutex_);
	current_offset_ = 0;

	if(decommit_unused && committed_head_ > 0){
		VirtualMemory::decommit(base_ptr_, committed_head_);
		committed_head_ = 0;
	}
}

} //namespace engine::mem
