import pstats

# Load the profiling data
p = pstats.Stats("output.prof")

# Sort by cumulative time and print the top 10 results
p.sort_stats("tottime").print_stats(100)
