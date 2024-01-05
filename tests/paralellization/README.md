## Paralell writing

Test whether writing in paralell is working as expected.
Timing for 1, 2, 4 and 8 threads is printed to stdout.

### Big array

Write a single large array (20'000 x 20'000).

```bash
./test_big_array/run.sh out_directory
```

### Smaller arrays in steps

Write 100 steps, where in each writing an array (200'000 x 3).

```bash
./test_steps/run.sh out_directory
```
