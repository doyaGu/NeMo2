#include <gtest/gtest.h>
#include <thread>
#include <chrono>

#include "VxTimeProfiler.h"

// Helper function to sleep for a specific duration in milliseconds.
void SleepFor(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

// Test Fixture for VxTimeProfiler
class VxTimeProfilerTest : public ::testing::Test {
protected:
    // A reasonable margin of error for timing tests, accounting for OS scheduling.
    // 25ms is a safe bet for short sleeps.
    const float timing_tolerance_ms = 25.0f;
};

// Test Case 1: Default Constructor
// Verifies that a newly created profiler starts at time zero.
TEST_F(VxTimeProfilerTest, DefaultConstructor_StartsAtZero) {
    // Action
    VxTimeProfiler profiler;

    // Verification
    // The time elapsed immediately after construction should be negligible.
    EXPECT_LT(profiler.Current(), 0.01f) << "A new profiler should have nearly zero elapsed time.";
}

// Test Case 2: Reset Method
// Verifies that Reset() sets the profiler's internal timer back to the current time.
TEST_F(VxTimeProfilerTest, Reset_ResetsTheTimer) {
    // Setup
    VxTimeProfiler profiler;
    SleepFor(50); // Let some time pass.
    ASSERT_GT(profiler.Current(), 0.04f) << "Profiler should have measured time before reset.";

    // Action
    profiler.Reset();

    // Verification
    // After reset, the elapsed time should be negligible again.
    EXPECT_LT(profiler.Current(), 0.01f) << "Profiler should be near zero after a reset.";
}

// Test Case 3: Copy Assignment Operator
// Verifies that one profiler can be assigned the state (start time) of another.
TEST_F(VxTimeProfilerTest, CopyAssignment_CopiesStartTime) {
    // Setup
    VxTimeProfiler profiler_A;
    SleepFor(50); // Let profiler A run for 50ms.

    VxTimeProfiler profiler_B;
    ASSERT_LT(profiler_B.Current(), profiler_A.Current()) << "Profilers should have different start times initially.";

    // Action
    profiler_B = profiler_A;

    // Verification
    // After assignment, both profilers should report roughly the same elapsed time
    // because they now share the same start time.
    EXPECT_NEAR(profiler_A.Current(), profiler_B.Current(), 0.01f)
        << "After assignment, profiler B should have the same start time as A.";

    // Let more time pass to ensure they continue timing from the same point.
    SleepFor(50);
    EXPECT_NEAR(profiler_A.Current(), profiler_B.Current(), 0.01f)
        << "Both profilers should still be in sync after more time has passed.";
}

// Test Case 4: Current() Method
// Verifies that Current() accurately measures elapsed time and is cumulative.
TEST_F(VxTimeProfilerTest, Current_MeasuresCumulativeTime) {
    // Setup
    VxTimeProfiler profiler;
    const int sleep_duration_ms = 100;

    // Action
    SleepFor(sleep_duration_ms);
    float time1 = profiler.Current();

    // Verification
    // Check if the first measurement is correct.
    EXPECT_NEAR(time1, sleep_duration_ms, timing_tolerance_ms)
        << "First time measurement should be around 100ms.";

    // Action 2: Let more time pass.
    SleepFor(sleep_duration_ms);
    float time2 = profiler.Current();

    // Verification 2
    // Check if the second measurement is cumulative (total time is ~200ms).
    EXPECT_NEAR(time2, 2 * sleep_duration_ms, timing_tolerance_ms)
        << "Second measurement should be cumulative, around 200ms.";
    EXPECT_GT(time2, time1) << "Time should be cumulative and continue to increase.";
}

// Test Case 5: Split() Method
// Verifies that Split() returns the elapsed time AND resets the timer.
TEST_F(VxTimeProfilerTest, Split_MeasuresTimeAndResets) {
    // Setup
    VxTimeProfiler profiler;
    const int sleep_duration_ms = 100;

    // Action
    SleepFor(sleep_duration_ms);
    float split_time = profiler.Split();

    // Verification
    // 1. Check if the returned split time is correct.
    EXPECT_NEAR(split_time, sleep_duration_ms, timing_tolerance_ms)
        << "Split() should return the elapsed time of ~100ms.";

    // 2. Check if the profiler was reset.
    float time_after_split = profiler.Current();
    EXPECT_LT(time_after_split, 0.01f)
        << "Immediately after a Split(), the current time should be near zero.";

    // 3. Let more time pass to confirm the timer restarted.
    SleepFor(50);
    float new_time = profiler.Current();
    EXPECT_NEAR(new_time, 50, timing_tolerance_ms)
        << "Profiler should measure time from the moment Split() was called.";
}

// Test Case 6: Edge Case - Rapid Calls
// Ensures that rapid, successive calls to the timing functions behave logically.
TEST_F(VxTimeProfilerTest, EdgeCase_RapidCalls) {
    // Setup
    VxTimeProfiler profiler;

    // Action & Verification
    float t1 = profiler.Current();
    float t2 = profiler.Current();
    EXPECT_GE(t2, t1) << "Successive calls to Current() should not go back in time.";

    profiler.Reset();
    float t3 = profiler.Split();
    float t4 = profiler.Current();
    EXPECT_LT(t3, 0.01f) << "Split() on a just-reset timer should be near zero.";
    EXPECT_LT(t4, 0.01f) << "Current() after a split on a just-reset timer should be near zero.";
}

