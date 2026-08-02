#include "utilities/Channel.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

TEST(Channel, FifoOrder) {
    util::Channel<int> ch(8);
    EXPECT_TRUE(ch.Push(1));
    EXPECT_TRUE(ch.Push(2));
    EXPECT_TRUE(ch.Push(3));
    int v = 0;
    ASSERT_TRUE(ch.Pop(v));
    EXPECT_EQ(v, 1);
    ASSERT_TRUE(ch.Pop(v));
    EXPECT_EQ(v, 2);
    ASSERT_TRUE(ch.Pop(v));
    EXPECT_EQ(v, 3);
}

TEST(Channel, DropOldestWhenFull) {
    util::Channel<int> ch(2, /*dropOldestWhenFull=*/true);
    ch.Push(1);
    ch.Push(2);
    ch.Push(3); // drops 1
    ch.Push(4); // drops 2
    EXPECT_EQ(ch.Size(), 2u);
    EXPECT_EQ(ch.Dropped(), 2u);
    int v = 0;
    ASSERT_TRUE(ch.Pop(v));
    EXPECT_EQ(v, 3);
    ASSERT_TRUE(ch.Pop(v));
    EXPECT_EQ(v, 4);
}

TEST(Channel, CloseDrainsThenReturnsFalse) {
    util::Channel<int> ch(8);
    ch.Push(10);
    ch.Push(20);
    ch.Close();
    EXPECT_FALSE(ch.Push(30)); // closed -> rejected
    int v = 0;
    ASSERT_TRUE(ch.Pop(v));
    EXPECT_EQ(v, 10);
    ASSERT_TRUE(ch.Pop(v));
    EXPECT_EQ(v, 20);
    EXPECT_FALSE(ch.Pop(v)); // closed + drained
}

TEST(Channel, CloseUnblocksBlockedPop) {
    util::Channel<int> ch(8);
    std::atomic<bool> returned{false};
    std::atomic<bool> popResult{true};
    std::thread consumer([&] {
        int v = 0;
        popResult = ch.Pop(v); // blocks (empty)
        returned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(returned); // still blocked
    ch.Close();
    consumer.join();
    EXPECT_TRUE(returned);
    EXPECT_FALSE(popResult); // closed + empty -> false
}

TEST(Channel, BlockingPushUnblocksOnPop) {
    util::Channel<int> ch(1, /*dropOldestWhenFull=*/false);
    ch.Push(1); // fills capacity
    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        ch.Push(2); // blocks until a Pop frees the slot
        pushed = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(pushed); // blocked (full, no drop)
    int v = 0;
    ASSERT_TRUE(ch.Pop(v));
    EXPECT_EQ(v, 1);
    producer.join();
    EXPECT_TRUE(pushed);
    ASSERT_TRUE(ch.Pop(v));
    EXPECT_EQ(v, 2);
}

TEST(Channel, ConcurrentSpscNoLoss) {
    util::Channel<int> ch(16, /*dropOldestWhenFull=*/false); // blocking -> no drops
    const int N = 100000;
    std::thread producer([&] {
        for (int i = 1; i <= N; ++i) ch.Push(i);
        ch.Close();
    });
    long long sum = 0;
    int count = 0;
    int v = 0;
    while (ch.Pop(v)) {
        sum += v;
        ++count;
    }
    producer.join();
    EXPECT_EQ(count, N);
    EXPECT_EQ(sum, 1LL * N * (N + 1) / 2);
    EXPECT_EQ(ch.Dropped(), 0u);
}
