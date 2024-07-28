#include "../include/utils.h"

void synchronize() {
#if defined(__NVCOMPILER) || defined(__NVCC__)
	gpuErrchk(cudaDeviceSynchronize());
#elif defined(__HIPCC__)
	gpuErrchk(hipDeviceSynchronize());
#endif
}

void genericFree(void* ptr) {
#if defined(__NVCOMPILER) || defined(__NVCC__)
	cudaFree(ptr);
#elif defined(__HIPCC__)
	hipFree(ptr);
#else
	free(ptr);
#endif	
}

_PREC Range::get_nth(int n) {
	return start + step * n;
}

int Range::size() {
	return (int) floor((stop - start)/step) + 1;
}

RangeUnion::RangeUnion(vector<Range> input_ranges){
	for (int i = 0; i < input_ranges.size(); i++) {
		positions.push_back(0);
	}
	ranges = input_ranges;
}

void RangeUnion::concatenate(RangeUnion other) {
	for (int i = 0; i < other.ranges.size(); i++) {
		push_back(other.ranges[i]);
	}
}

void RangeUnion::push_back(Range range) {
	ranges.push_back(range);
	positions.push_back(0);
}

bool RangeUnion::hasNext() {
	for (int i = 0; i < ranges.size(); i++) {
		if (positions[i] < ranges[i].size()) {
			return true;
		}
	}
	return false;
}

_PREC RangeUnion::next(){
	_PREC min_val = std::numeric_limits<_PREC>::infinity();
	int min_range = -1;
	for (int i = 0; i < ranges.size(); i++) {
		if (positions[i] >= ranges[i].size()) {
			continue;
		}
		_PREC val = ranges[i].get_nth(positions[i]);
		if (val < min_val) {
			min_val = val;
			min_range = i;
		}
	}
	positions[min_range] += 1;
	return min_val;
}

int RangeUnion::size() {
	int total = 0;
	for (int i = 0; i < ranges.size(); i++) {
		total += ranges[i].size();
	}
	return total;
}
