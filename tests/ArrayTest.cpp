#include <gtest/gtest.h>
#include "XArray.h"

// Test simple types with XArray
class XArrayTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
        for (int i = 0; i < 10; ++i) {
            test_data[i] = i * 2;
        }
    }

    int test_data[10];
};

TEST_F(XArrayTest, DefaultConstructor) {
    XArray<int> arr;
    EXPECT_EQ(arr.Size(), 0);
    EXPECT_TRUE(arr.IsEmpty());
    EXPECT_EQ(arr.Allocated(), 0);
}

TEST_F(XArrayTest, ConstructorWithSize) {
    XArray<int> arr(5);
    EXPECT_EQ(arr.Size(), 0);
    EXPECT_TRUE(arr.IsEmpty());
    EXPECT_GE(arr.Allocated(), 5);
}

TEST_F(XArrayTest, CopyConstructor) {
    XArray<int> original;
    original.PushBack(1);
    original.PushBack(2);
    original.PushBack(3);

    XArray<int> copy(original);
    EXPECT_EQ(copy.Size(), 3);
    EXPECT_EQ(copy[0], 1);
    EXPECT_EQ(copy[1], 2);
    EXPECT_EQ(copy[2], 3);

    // Verify deep copy
    copy[0] = 10;
    EXPECT_EQ(original[0], 1);
    EXPECT_EQ(copy[0], 10);
}

TEST_F(XArrayTest, Assignment) {
    XArray<int> original;
    original.PushBack(1);
    original.PushBack(2);

    XArray<int> assigned;
    assigned.PushBack(99);

    assigned = original;
    EXPECT_EQ(assigned.Size(), 2);
    EXPECT_EQ(assigned[0], 1);
    EXPECT_EQ(assigned[1], 2);
}

TEST_F(XArrayTest, PushBack) {
    XArray<int> arr;

    for (int i = 0; i < 5; ++i) {
        arr.PushBack(test_data[i]);
        EXPECT_EQ(arr.Size(), i + 1);
        EXPECT_EQ(arr[i], test_data[i]);
    }

    EXPECT_FALSE(arr.IsEmpty());
}

TEST_F(XArrayTest, PopBack) {
    XArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);

    EXPECT_EQ(arr.Size(), 3);

    arr.PopBack();
    EXPECT_EQ(arr.Size(), 2);
    EXPECT_EQ(arr[1], 20);

    arr.PopBack();
    EXPECT_EQ(arr.Size(), 1);
    EXPECT_EQ(arr[0], 10);
}

TEST_F(XArrayTest, IndexAccess) {
    XArray<int> arr;
    arr.PushBack(100);
    arr.PushBack(200);

    EXPECT_EQ(arr[0], 100);
    EXPECT_EQ(arr[1], 200);

    arr[0] = 500;
    EXPECT_EQ(arr[0], 500);

    // Test const access
    const XArray<int> &const_arr = arr;
    EXPECT_EQ(const_arr[0], 500);
}

TEST_F(XArrayTest, At) {
    XArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);

    EXPECT_EQ(*arr.At(0), 10);
    EXPECT_EQ(*arr.At(1), 20);

    *arr.At(0) = 15;
    EXPECT_EQ(*arr.At(0), 15);
}

TEST_F(XArrayTest, FrontBack) {
    XArray<int> arr;
    arr.PushBack(100);
    arr.PushBack(200);
    arr.PushBack(300);

    EXPECT_EQ(arr.Front(), 100);
    EXPECT_EQ(arr.Back(), 300);

    arr.Front() = 150;
    arr.Back() = 350;

    EXPECT_EQ(arr[0], 150);
    EXPECT_EQ(arr[2], 350);
}

TEST_F(XArrayTest, Insert) {
    XArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(30);

    // Insert at beginning
    arr.Insert(arr.Begin(), 5);
    EXPECT_EQ(arr.Size(), 3);
    EXPECT_EQ(arr[0], 5);
    EXPECT_EQ(arr[1], 10);
    EXPECT_EQ(arr[2], 30);

    // Insert in middle
    arr.Insert(arr.Begin() + 2, 20);
    EXPECT_EQ(arr.Size(), 4);
    EXPECT_EQ(arr[0], 5);
    EXPECT_EQ(arr[1], 10);
    EXPECT_EQ(arr[2], 20);
    EXPECT_EQ(arr[3], 30);
}

TEST_F(XArrayTest, Remove) {
    XArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);
    arr.PushBack(40);

    // Remove from middle
    XArray<int>::Iterator it = arr.Remove(arr.Begin() + 1);
    EXPECT_EQ(arr.Size(), 3);
    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 30);
    EXPECT_EQ(arr[2], 40);
    EXPECT_EQ(*it, 30);

    // Remove first element
    it = arr.Remove(arr.Begin());
    EXPECT_EQ(arr.Size(), 2);
    EXPECT_EQ(arr[0], 30);
    EXPECT_EQ(arr[1], 40);
}

TEST_F(XArrayTest, Clear) {
    XArray<int> arr;
    arr.PushBack(1);
    arr.PushBack(2);
    arr.PushBack(3);

    EXPECT_EQ(arr.Size(), 3);
    EXPECT_FALSE(arr.IsEmpty());

    arr.Clear();
    EXPECT_EQ(arr.Size(), 0);
    EXPECT_TRUE(arr.IsEmpty());
}

TEST_F(XArrayTest, Reserve) {
    XArray<int> arr;
    EXPECT_EQ(arr.Allocated(), 0);

    arr.Reserve(100);
    EXPECT_GE(arr.Allocated(), 100);
    EXPECT_EQ(arr.Size(), 0);

    // Adding elements should not require reallocation
    for (int i = 0; i < 50; ++i) {
        arr.PushBack(i);
    }
    EXPECT_GE(arr.Allocated(), 100);
}

TEST_F(XArrayTest, Resize) {
    XArray<int> arr;

    arr.Resize(5);
    EXPECT_EQ(arr.Size(), 5);

    // // All elements should be default-initialized (0 for int)
    // for (int i = 0; i < 5; ++i) {
    //     EXPECT_EQ(arr[i], 0);
    // }

    arr.Resize(10);
    EXPECT_EQ(arr.Size(), 10);

    // // First 5 elements should remain 0
    // for (int i = 0; i < 5; ++i) {
    //     EXPECT_EQ(arr[i], 0);
    // }

    // // New elements should be 42
    // for (int i = 5; i < 10; ++i) {
    //     EXPECT_EQ(arr[i], 42);
    // }

    // Shrink
    arr.Resize(3);
    EXPECT_EQ(arr.Size(), 3);
}

TEST_F(XArrayTest, Find) {
    XArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);
    arr.PushBack(20);

    XArray<int>::Iterator it = arr.Find(20);
    EXPECT_NE(it, arr.End());
    EXPECT_EQ(*it, 20);
    EXPECT_EQ(it - arr.Begin(), 1); // Should find first occurrence

    it = arr.Find(999);
    EXPECT_EQ(it, arr.End()); // Not found
}

TEST_F(XArrayTest, Sort) {
    XArray<int> arr;
    arr.PushBack(30);
    arr.PushBack(10);
    arr.PushBack(40);
    arr.PushBack(20);

    arr.Sort();

    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 20);
    EXPECT_EQ(arr[2], 30);
    EXPECT_EQ(arr[3], 40);
}

TEST_F(XArrayTest, Iterators) {
    XArray<int> arr;
    arr.PushBack(1);
    arr.PushBack(2);
    arr.PushBack(3);

    // Test iteration
    int expected = 1;
    for (XArray<int>::Iterator it = arr.Begin(); it != arr.End(); ++it) {
        EXPECT_EQ(*it, expected++);
    }

    // Test reverse iteration
    expected = 3;
    for (XArray<int>::Iterator it = arr.End() - 1; it >= arr.Begin(); --it) {
        EXPECT_EQ(*it, expected--);
    }
}

TEST_F(XArrayTest, SwapElements) {
    XArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);

    arr.Swap(0, 2);
    EXPECT_EQ(arr[0], 30);
    EXPECT_EQ(arr[2], 10);
    EXPECT_EQ(arr[1], 20); // Should remain unchanged
}

TEST_F(XArrayTest, FillArray) {
    XArray<int> arr;
    arr.Resize(5);

    arr.Fill(42);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(arr[i], 42);
    }
}

// Test XArray with custom objects
struct TestObject {
    int value;
    static int construction_count;
    static int destruction_count;

    TestObject() : value(0) { construction_count++; }
    TestObject(int v) : value(v) { construction_count++; }
    TestObject(const TestObject &other) : value(other.value) { construction_count++; }
    ~TestObject() { destruction_count++; }

    bool operator==(const TestObject &other) const { return value == other.value; }
    bool operator!=(const TestObject &other) const { return value != other.value; }
    bool operator>(const TestObject &other) const { return value > other.value; }
    bool operator<(const TestObject &other) const { return value < other.value; }

    static void ResetCounters() {
        construction_count = 0;
        destruction_count = 0;
    }
};

int TestObject::construction_count = 0;
int TestObject::destruction_count = 0;

class XArrayObjectTest : public ::testing::Test {
protected:
    void SetUp() override {
        TestObject::ResetCounters();
    }

    void TearDown() override {
        // Verify no memory leaks
        EXPECT_EQ(TestObject::construction_count, TestObject::destruction_count);
    }
};

TEST_F(XArrayObjectTest, ObjectLifetime) {
    {
        XArray<TestObject> arr;
        arr.PushBack(TestObject(10));
        arr.PushBack(TestObject(20));
    }

    // Objects should be properly destroyed
    EXPECT_EQ(TestObject::construction_count, TestObject::destruction_count);
}

TEST_F(XArrayObjectTest, ObjectOperations) {
    XArray<TestObject> arr;

    arr.PushBack(TestObject(30));
    arr.PushBack(TestObject(10));
    arr.PushBack(TestObject(20));

    EXPECT_EQ(arr[0].value, 30);
    EXPECT_EQ(arr[1].value, 10);
    EXPECT_EQ(arr[2].value, 20);

    arr.Sort();

    EXPECT_EQ(arr[0].value, 10);
    EXPECT_EQ(arr[1].value, 20);
    EXPECT_EQ(arr[2].value, 30);

    XArray<TestObject>::Iterator it = arr.Find(TestObject(20));
    EXPECT_NE(it, arr.End());
    EXPECT_EQ(it->value, 20);
}

// Performance and edge case tests
const int LARGE_SIZE = 10000;

class XArrayPerformanceTest : public ::testing::Test {
};

TEST_F(XArrayPerformanceTest, LargeArrayOperations) {
    XArray<int> arr;

    // Test growing capacity
    for (int i = 0; i < LARGE_SIZE; ++i) {
        arr.PushBack(i);
    }

    EXPECT_EQ(arr.Size(), LARGE_SIZE);

    // Verify all elements
    for (int i = 0; i < LARGE_SIZE; ++i) {
        EXPECT_EQ(arr[i], i);
    }

    // Test shrinking
    for (int i = 0; i < LARGE_SIZE / 2; ++i) {
        arr.PopBack();
    }

    EXPECT_EQ(arr.Size(), LARGE_SIZE / 2);
}

TEST_F(XArrayPerformanceTest, ReserveAndResize) {
    XArray<int> arr;

    // Reserve large capacity
    arr.Reserve(LARGE_SIZE);
    int initial_capacity = arr.Allocated();

    // Add elements without triggering reallocation
    for (int i = 0; i < LARGE_SIZE / 2; ++i) {
        arr.PushBack(i);
    }

    // Capacity should remain the same
    EXPECT_EQ(arr.Allocated(), initial_capacity);

    // Test resize
    arr.Resize(LARGE_SIZE);
    EXPECT_EQ(arr.Size(), LARGE_SIZE);

    // // Check that new elements have the correct value
    // for (int i = LARGE_SIZE / 2; i < LARGE_SIZE; ++i) {
    //     EXPECT_EQ(arr[i], 42);
    // }
}

TEST_F(XArrayPerformanceTest, SortLargeArray) {
    XArray<int> arr;

    // Add elements in reverse order
    for (int i = LARGE_SIZE - 1; i >= 0; --i) {
        arr.PushBack(i);
    }

    arr.Sort();

    // Verify sorted order
    for (int i = 0; i < LARGE_SIZE; ++i) {
        EXPECT_EQ(arr[i], i);
    }
}
