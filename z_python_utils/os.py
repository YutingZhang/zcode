

def self_memory_usage():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss
    return mem_usage
