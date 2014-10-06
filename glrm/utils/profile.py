import cProfile, pstats, StringIO

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
#             s = StringIO.StringIO()
#             sortby = "cumulative"
#             ps = pstats.Stats(profile, stream = s).sort_stats(sortby)
#             ps.print_stats()
#             print s.getvalue()
            profile.print_stats()
    return profiled_func

