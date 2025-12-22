# benchmarks

## matmul

To evaluate the performance of my 4x4 matrix multiplication implementation, a series of runtime experiments was conducted. A total of 1,000,000 pairs of random 4x4 matrices were generated. The execution time of the 4x4 matrix multiplication was measured for three approaches: my optimized implementation utilizing SIMD instructions and memory alignment, a naive C++ implementation, and the implementation provided by the GLM library. Each experiment was repeated ten times, and the mean execution time and standard deviation were computed across repetitions. The results are presented in the figure below.

[![Diagram](matmul/benchmark_results-1.png)](matmul/benchmark_results.pdf)

The performance difference observed for GLM may be caused by abstraction overhead, data layout, or conservative optimization choices aimed at portability. A deeper low-level analysis would be needed to pinpoint the exact reason. If anyone has insight into why GLM performs better than the naive implementation in this benchmark, feedback is very welcome.
