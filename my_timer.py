import time

class mytimer:
    all_time = 0
    kernel_time = 0
    timer_time = 0
    linear_time = 0
    sdpa_time = 0
    trans_time = 0
    @staticmethod
    def record():
        return time.perf_counter()