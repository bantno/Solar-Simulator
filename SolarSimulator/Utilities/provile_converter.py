import pstats

stats = pstats.Stats('output.prof')
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()  # Optional: To see the output on the console

with open('profile_output.txt', 'w') as f:
    stats.stream = f
    stats.print_stats()
