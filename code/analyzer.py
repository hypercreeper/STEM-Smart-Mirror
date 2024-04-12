import pstats
p = pstats.Stats('profileing')
p.sort_stats('cumulative').print_stats()
