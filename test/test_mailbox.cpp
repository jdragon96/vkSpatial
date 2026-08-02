#include "utilities/Mailbox.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

TEST(Mailbox, EmptyReturnsNull) {
    util::Mailbox<int> mb;
    EXPECT_EQ(mb.Latest(), nullptr);
}

TEST(Mailbox, LatestWins) {
    util::Mailbox<int> mb;
    mb.Publish(std::make_shared<const int>(1));
    mb.Publish(std::make_shared<const int>(2));
    auto v = mb.Latest();
    ASSERT_NE(v, nullptr);
    EXPECT_EQ(*v, 2);
}

// A snapshot held by a consumer stays valid after the producer publishes newer ones.
TEST(Mailbox, HeldSnapshotSurvivesNewerPublish) {
    util::Mailbox<int> mb;
    mb.Publish(std::make_shared<const int>(10));
    auto held = mb.Latest();
    mb.Publish(std::make_shared<const int>(20));
    ASSERT_NE(held, nullptr);
    EXPECT_EQ(*held, 10);   // still valid
    EXPECT_EQ(*mb.Latest(), 20);
}

// Concurrent producer + consumer: consumer always sees a valid, monotonically-non-decreasing value.
TEST(Mailbox, ConcurrentProducerConsumer) {
    util::Mailbox<int> mb;
    std::atomic<bool> stop{false};
    std::thread producer([&] {
        for (int i = 1; i <= 100000; ++i) mb.Publish(std::make_shared<const int>(i));
        stop = true;
    });
    int last = 0;
    while (!stop) {
        auto v = mb.Latest();
        if (v) {
            EXPECT_GE(*v, last);
            last = *v;
        }
    }
    producer.join();
    ASSERT_NE(mb.Latest(), nullptr);
    EXPECT_EQ(*mb.Latest(), 100000);
}
