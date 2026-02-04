# Prerequisite Assignment – Average Color Calculation

This repository contains the solution for the prerequisite assignment, which implements a function to compute the average RGB color of a rectangular region in a bitmap image.

## Files and Directories

- `prereq.cc`
  Contains the implementation of the `calculate` function.

- `tests/`
  Test cases used by the grader.

- `benchmarks/`
  Benchmark-related files provided with the assignment.

- `grading/`
  Grading scripts and tools.

- `prereq`
  Compiled executable generated during grading.

- `prereq.dSYM`
  Debug symbols generated on macOS.

## Description

The function iterates over the pixels inside a given rectangle and computes the arithmetic mean of the red, green, and blue color components. All calculations are performed using double precision, and the final result is returned as single-precision floating-point values, as required.

## Build and Test

To run the grader:

```bash
./grading test
```
## On slower systems (e.g. macOS), you may need:

```bash
./grading test --no-timeout
```
## Notes

- The implementation prioritizes correctness and clarity over performance.
- No parallelism or advanced optimization techniques are used.

# End
