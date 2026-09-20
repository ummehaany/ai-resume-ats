from backend.core.rate_limit import SlidingWindowLimiter


def test_allows_up_to_limit_then_blocks_with_retry_after():
    lim = SlidingWindowLimiter()
    rule = [('k', 2, 60)]
    assert lim.try_acquire(rule, now=0) is None
    assert lim.try_acquire(rule, now=10) is None
    retry = lim.try_acquire(rule, now=20)
    assert retry == 41                       # first hit expires at t=60 -> 40s + 1
    assert lim.try_acquire(rule, now=61) is None


def test_keys_are_independent():
    lim = SlidingWindowLimiter()
    assert lim.try_acquire([('a', 1, 60)], now=0) is None
    assert lim.try_acquire([('a', 1, 60)], now=1) is not None
    assert lim.try_acquire([('b', 1, 60)], now=1) is None


def test_zero_or_negative_limit_disables_a_rule():
    lim = SlidingWindowLimiter()
    for i in range(50):
        assert lim.try_acquire([('k', 0, 60), ('j', -1, 60)], now=i) is None


def test_blocked_request_does_not_consume_other_rules():
    lim = SlidingWindowLimiter()
    assert lim.try_acquire([('user', 1, 60)], now=0) is None
    assert lim.try_acquire([('user', 1, 60), ('global', 2, 60)], now=1) is not None   # user rule blocks
    # the global counter must not have been charged for the blocked call
    assert lim.try_acquire([('global', 2, 60)], now=2) is None
    assert lim.try_acquire([('global', 2, 60)], now=3) is None
    assert lim.try_acquire([('global', 2, 60)], now=4) is not None


def test_thread_safety_smoke():
    import threading
    lim = SlidingWindowLimiter()
    allowed = []

    def worker():
        for _ in range(50):
            if lim.try_acquire([('k', 100, 60)], now=1.0) is None:
                allowed.append(1)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(allowed) == 100
