#pragma once

#include<vector>
#include<cassert>
#include<cstddef>
#include<cstring>

#include"event.hpp"

namespace engine::event{

template<std::size_t CAP>
class EventQueue{
	static_assert((CAP & (CAP - 1)) == 0 && "capacity must be a power of two");

public:
	EventQueue() : mask_(CAP - 1), head_(0), tail_(0) {}

	inline bool push(const Event& e){
		const std::size_t next_ = (head_ + 1) & mask_;
		if(next_ == tail_) return false;
		buf_[head_] = e;
		head_ = next_;
		return true;
	}

	inline bool pop(Event& out){
		if(tail_ == head_) return false;
		out = buf_[tail_];
		tail_ = (tail_ + 1) & mask_;
		return true;
	}

	inline void clear() {head_ = tail_ = 0;}
	inline bool empty() const { return head_ == tail_;}
	inline std::size_t size() const {return (head_ - tail_) & mask_; }

private:
	std::array<Event, CAP> buf_;
	std::size_t mask_;
	std::size_t head_;
	std::size_t tail_;

};

} // namespace engine::event
