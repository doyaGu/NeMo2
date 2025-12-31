#include <gtest/gtest.h>
#include "XArray.h"
#include "XSArray.h"
#include "XClassArray.h"
#include "XBitArray.h"
#include <cstring>
#include <initializer_list>
#include <utility>

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

TEST_F(XArrayTest, AttachDetach) {
    int external_data[5] = {1, 2, 3, 4, 5};
    
    XArray<int> arr;
    arr.Attach(external_data, 5);
    
    EXPECT_EQ(arr.Size(), 5);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[4], 5);
    
    // Modify through array
    arr[2] = 99;
    EXPECT_EQ(external_data[2], 99);
    
    arr.Detach();
    EXPECT_EQ(arr.Size(), 0);
}

TEST_F(XArrayTest, BinaryFind) {
    XArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);
    arr.PushBack(40);
    arr.PushBack(50);
    
    int* found = arr.BinaryFind(30);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(*found, 30);
    
    found = arr.BinaryFind(25);
    EXPECT_EQ(found, nullptr);
}

TEST_F(XArrayTest, GetPosition) {
    XArray<int> arr;
    arr.PushBack(100);
    arr.PushBack(200);
    arr.PushBack(300);
    
    EXPECT_EQ(arr.GetPosition(100), 0);
    EXPECT_EQ(arr.GetPosition(200), 1);
    EXPECT_EQ(arr.GetPosition(300), 2);
    EXPECT_EQ(arr.GetPosition(999), static_cast<size_t>(-1));
}

TEST_F(XArrayTest, FastRemove) {
    XArray<int> arr;
    for (int i = 0; i < 5; ++i) {
        arr.PushBack(i * 10);
    }
    
    // Fast remove swaps with last element
    arr.FastRemove(10);
    EXPECT_EQ(arr.Size(), 4);
    // Element 40 (last) should now be at index 1
    EXPECT_EQ(arr[1], 40);
}

TEST_F(XArrayTest, ArraySubtraction) {
    XArray<int> arr1;
    arr1.PushBack(1);
    arr1.PushBack(2);
    arr1.PushBack(3);
    arr1.PushBack(4);
    
    XArray<int> arr2;
    arr2.PushBack(2);
    arr2.PushBack(4);
    
    arr1 -= arr2;
    
    EXPECT_EQ(arr1.Size(), 2);
    EXPECT_EQ(arr1[0], 1);
    EXPECT_EQ(arr1[1], 3);
}

TEST_F(XArrayTest, Memset) {
    XArray<int> arr;
    arr.Resize(10);
    arr.Memset(0);
    
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arr[i], 0);
    }
}

#if VX_HAS_CXX11
TEST_F(XArrayTest, MoveConstructor) {
    XArray<int> arr1;
    arr1.PushBack(1);
    arr1.PushBack(2);
    arr1.PushBack(3);
    
    XArray<int> arr2(std::move(arr1));
    
    EXPECT_EQ(arr2.Size(), 3);
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);
    EXPECT_EQ(arr1.Size(), 0);
}

TEST_F(XArrayTest, MoveAssignment) {
    XArray<int> arr1;
    arr1.PushBack(1);
    arr1.PushBack(2);
    arr1.PushBack(3);
    
    XArray<int> arr2;
    arr2 = std::move(arr1);
    
    EXPECT_EQ(arr2.Size(), 3);
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr1.Size(), 0);
}

TEST_F(XArrayTest, Cxx11InitializerListCtorAndAssign) {
    XArray<int> arr{1, 2, 3};
    EXPECT_EQ(arr.Size(), 3);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);

    arr = {7, 8};
    EXPECT_EQ(arr.Size(), 2);
    EXPECT_EQ(arr[0], 7);
    EXPECT_EQ(arr[1], 8);
}

TEST_F(XArrayTest, Cxx11RvalueInsertAndEmplaceAndRangeFor) {
    XArray<int> arr;

    int v = 10;
    arr.PushBack(std::move(v));
    arr.PushBack(30);

    int mid = 20;
    arr.Insert(arr.Begin() + 1, std::move(mid));

    arr.Emplace(arr.Begin(), 5);
    arr.EmplaceBack(40);

    EXPECT_EQ(arr.Size(), 5);
    EXPECT_EQ(arr[0], 5);
    EXPECT_EQ(arr[1], 10);
    EXPECT_EQ(arr[2], 20);
    EXPECT_EQ(arr[3], 30);
    EXPECT_EQ(arr[4], 40);

    int sum = 0;
    for (int x : arr) {
        sum += x;
    }
    EXPECT_EQ(sum, 5 + 10 + 20 + 30 + 40);
}
#endif

// ==============================================================================
// XSArray Tests - Space-Efficient Array (No Extra Capacity)
// ==============================================================================

class XSArrayTest : public ::testing::Test {
};

TEST_F(XSArrayTest, DefaultConstructor) {
    XSArray<int> arr;
    EXPECT_EQ(arr.Size(), 0);
}

TEST_F(XSArrayTest, CopyConstructor) {
    XSArray<int> original;
    original.PushBack(1);
    original.PushBack(2);
    original.PushBack(3);
    
    XSArray<int> copy(original);
    EXPECT_EQ(copy.Size(), 3);
    EXPECT_EQ(copy[0], 1);
    EXPECT_EQ(copy[1], 2);
    EXPECT_EQ(copy[2], 3);
    
    // Verify deep copy
    copy[0] = 10;
    EXPECT_EQ(original[0], 1);
}

TEST_F(XSArrayTest, Assignment) {
    XSArray<int> original;
    original.PushBack(1);
    original.PushBack(2);
    
    XSArray<int> assigned;
    assigned = original;
    
    EXPECT_EQ(assigned.Size(), 2);
    EXPECT_EQ(assigned[0], 1);
    EXPECT_EQ(assigned[1], 2);
}

TEST_F(XSArrayTest, PushBackAndAccess) {
    XSArray<int> arr;
    
    for (int i = 0; i < 5; ++i) {
        arr.PushBack(i * 10);
        EXPECT_EQ(arr.Size(), i + 1);
        EXPECT_EQ(arr[i], i * 10);
    }
}

TEST_F(XSArrayTest, PushFront) {
    XSArray<int> arr;
    arr.PushBack(20);
    arr.PushBack(30);
    
    arr.PushFront(10);
    
    EXPECT_EQ(arr.Size(), 3);
    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 20);
    EXPECT_EQ(arr[2], 30);
}

TEST_F(XSArrayTest, Insert) {
    XSArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(30);
    
    arr.Insert(1, 20);
    
    EXPECT_EQ(arr.Size(), 3);
    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 20);
    EXPECT_EQ(arr[2], 30);
}

TEST_F(XSArrayTest, PopBack) {
    XSArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);
    
    arr.PopBack();
    EXPECT_EQ(arr.Size(), 2);
    EXPECT_EQ(arr[1], 20);
}

TEST_F(XSArrayTest, Remove) {
    XSArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);
    
    int* it = arr.Remove(arr.Begin() + 1);
    
    EXPECT_EQ(arr.Size(), 2);
    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 30);
}

TEST_F(XSArrayTest, RemoveByValue) {
    XSArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);
    
    arr.Remove(20);
    
    EXPECT_EQ(arr.Size(), 2);
    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 30);
}

TEST_F(XSArrayTest, Clear) {
    XSArray<int> arr;
    arr.PushBack(1);
    arr.PushBack(2);
    arr.PushBack(3);
    
    arr.Clear();
    EXPECT_EQ(arr.Size(), 0);
}

TEST_F(XSArrayTest, Resize) {
    XSArray<int> arr;
    arr.Resize(5);
    EXPECT_EQ(arr.Size(), 5);
    
    arr.Resize(10);
    EXPECT_EQ(arr.Size(), 10);
    
    arr.Resize(3);
    EXPECT_EQ(arr.Size(), 3);
}

TEST_F(XSArrayTest, Fill) {
    XSArray<int> arr;
    arr.Resize(5);
    arr.Fill(42);
    
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(arr[i], 42);
    }
}

TEST_F(XSArrayTest, Find) {
    XSArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);
    
    int* found = arr.Find(20);
    EXPECT_NE(found, arr.End());
    EXPECT_EQ(*found, 20);
    
    found = arr.Find(999);
    EXPECT_EQ(found, arr.End());
}

TEST_F(XSArrayTest, IsHere) {
    XSArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    
    EXPECT_TRUE(arr.IsHere(10));
    EXPECT_TRUE(arr.IsHere(20));
    EXPECT_FALSE(arr.IsHere(30));
}

TEST_F(XSArrayTest, Sort) {
    XSArray<int> arr;
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

TEST_F(XSArrayTest, Swap) {
    XSArray<int> arr;
    arr.PushBack(10);
    arr.PushBack(20);
    arr.PushBack(30);
    
    arr.Swap(0, 2);
    
    EXPECT_EQ(arr[0], 30);
    EXPECT_EQ(arr[2], 10);
}

TEST_F(XSArrayTest, ArrayAddition) {
    XSArray<int> arr1;
    arr1.PushBack(1);
    arr1.PushBack(2);
    
    XSArray<int> arr2;
    arr2.PushBack(3);
    arr2.PushBack(4);
    
    arr1 += arr2;
    
    EXPECT_EQ(arr1.Size(), 4);
    EXPECT_EQ(arr1[0], 1);
    EXPECT_EQ(arr1[1], 2);
    EXPECT_EQ(arr1[2], 3);
    EXPECT_EQ(arr1[3], 4);
}

TEST_F(XSArrayTest, ArraySubtraction) {
    XSArray<int> arr1;
    arr1.PushBack(1);
    arr1.PushBack(2);
    arr1.PushBack(3);
    arr1.PushBack(4);
    
    XSArray<int> arr2;
    arr2.PushBack(2);
    arr2.PushBack(4);
    
    arr1 -= arr2;
    
    EXPECT_EQ(arr1.Size(), 2);
    EXPECT_EQ(arr1[0], 1);
    EXPECT_EQ(arr1[1], 3);
}

TEST_F(XSArrayTest, Iterators) {
    XSArray<int> arr;
    arr.PushBack(1);
    arr.PushBack(2);
    arr.PushBack(3);
    
    int expected = 1;
    for (int* it = arr.Begin(); it != arr.End(); ++it) {
        EXPECT_EQ(*it, expected++);
    }
}

TEST_F(XSArrayTest, BeginEnd) {
    XSArray<int> arr;
    arr.PushBack(100);
    arr.PushBack(200);
    arr.PushBack(300);
    
    EXPECT_EQ(*arr.Begin(), 100);
    EXPECT_EQ(*(arr.End() - 1), 300);
}

#if VX_HAS_CXX11
TEST_F(XSArrayTest, MoveConstructor) {
    XSArray<int> arr1;
    arr1.PushBack(1);
    arr1.PushBack(2);
    arr1.PushBack(3);
    
    XSArray<int> arr2(std::move(arr1));
    
    EXPECT_EQ(arr2.Size(), 3);
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr1.Size(), 0);
}

TEST_F(XSArrayTest, MoveAssignment) {
    XSArray<int> arr1;
    arr1.PushBack(1);
    arr1.PushBack(2);
    
    XSArray<int> arr2;
    arr2 = std::move(arr1);
    
    EXPECT_EQ(arr2.Size(), 2);
    EXPECT_EQ(arr1.Size(), 0);
}

TEST_F(XSArrayTest, Cxx11InitializerListCtorAndAssign) {
    XSArray<int> arr{1, 2, 3};
    EXPECT_EQ(arr.Size(), 3);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);

    arr = {7, 8};
    EXPECT_EQ(arr.Size(), 2);
    EXPECT_EQ(arr[0], 7);
    EXPECT_EQ(arr[1], 8);
}

TEST_F(XSArrayTest, Cxx11RvalueInsertAndEmplaceAndRangeFor) {
    XSArray<int> arr;

    int v = 10;
    arr.PushBack(std::move(v));
    arr.PushBack(30);

    int mid = 20;
    arr.Insert(1, std::move(mid));

    arr.Emplace(arr.Begin(), 5);
    arr.EmplaceBack(40);

    EXPECT_EQ(arr.Size(), 5);
    EXPECT_EQ(arr[0], 5);
    EXPECT_EQ(arr[1], 10);
    EXPECT_EQ(arr[2], 20);
    EXPECT_EQ(arr[3], 30);
    EXPECT_EQ(arr[4], 40);

    int sum = 0;
    for (int x : arr) {
        sum += x;
    }
    EXPECT_EQ(sum, 5 + 10 + 20 + 30 + 40);
}
#endif

// ==============================================================================
// XClassArray Tests - Array for Classes with Proper Construction/Destruction
// ==============================================================================

struct ComplexObject {
    int value;
    int* ptr;
    static int construction_count;
    static int destruction_count;
    
    ComplexObject() : value(0), ptr(new int(0)) { construction_count++; }
    ComplexObject(int v) : value(v), ptr(new int(v)) { construction_count++; }
    
    ComplexObject(const ComplexObject& other) : value(other.value), ptr(new int(*other.ptr)) {
        construction_count++;
    }
    
    ComplexObject& operator=(const ComplexObject& other) {
        if (this != &other) {
            value = other.value;
            *ptr = *other.ptr;
        }
        return *this;
    }
    
    ~ComplexObject() {
        destruction_count++;
        delete ptr;
    }
    
    bool operator==(const ComplexObject& other) const { return value == other.value; }
    bool operator!=(const ComplexObject& other) const { return value != other.value; }
    bool operator<(const ComplexObject& other) const { return value < other.value; }
    bool operator>(const ComplexObject& other) const { return value > other.value; }
    
    static void ResetCounters() {
        construction_count = 0;
        destruction_count = 0;
    }
};

int ComplexObject::construction_count = 0;
int ComplexObject::destruction_count = 0;

class XClassArrayTest : public ::testing::Test {
protected:
    void SetUp() override {
        ComplexObject::ResetCounters();
    }
    
    void TearDown() override {
        // Verify no memory leaks
        EXPECT_EQ(ComplexObject::construction_count, ComplexObject::destruction_count);
    }
};

TEST_F(XClassArrayTest, DefaultConstructor) {
    XClassArray<ComplexObject> arr;
    EXPECT_EQ(arr.Size(), 0);
}

TEST_F(XClassArrayTest, ConstructorWithReserve) {
    XClassArray<ComplexObject> arr(10);
    EXPECT_EQ(arr.Size(), 0);
    EXPECT_GE(arr.Allocated(), 10);
}

TEST_F(XClassArrayTest, PushBackAndAccess) {
    XClassArray<ComplexObject> arr;
    
    arr.PushBack(ComplexObject(10));
    arr.PushBack(ComplexObject(20));
    arr.PushBack(ComplexObject(30));
    
    EXPECT_EQ(arr.Size(), 3);
    EXPECT_EQ(arr[0].value, 10);
    EXPECT_EQ(arr[1].value, 20);
    EXPECT_EQ(arr[2].value, 30);
}

TEST_F(XClassArrayTest, CopyConstructor) {
    XClassArray<ComplexObject> original;
    original.PushBack(ComplexObject(1));
    original.PushBack(ComplexObject(2));
    
    XClassArray<ComplexObject> copy(original);
    
    EXPECT_EQ(copy.Size(), 2);
    EXPECT_EQ(copy[0].value, 1);
    EXPECT_EQ(copy[1].value, 2);
    
    // Verify deep copy
    copy[0].value = 99;
    EXPECT_EQ(original[0].value, 1);
}

TEST_F(XClassArrayTest, Assignment) {
    XClassArray<ComplexObject> original;
    original.PushBack(ComplexObject(1));
    original.PushBack(ComplexObject(2));
    
    XClassArray<ComplexObject> assigned;
    assigned = original;
    
    EXPECT_EQ(assigned.Size(), 2);
    EXPECT_EQ(assigned[0].value, 1);
}

TEST_F(XClassArrayTest, Resize) {
    XClassArray<ComplexObject> arr;
    
    arr.Resize(5);
    EXPECT_EQ(arr.Size(), 5);
    
    arr.Resize(10);
    EXPECT_EQ(arr.Size(), 10);
    
    arr.Resize(3);
    EXPECT_EQ(arr.Size(), 3);
}

TEST_F(XClassArrayTest, Reserve) {
    XClassArray<ComplexObject> arr;
    
    arr.Reserve(100);
    EXPECT_GE(arr.Allocated(), 100);
    EXPECT_EQ(arr.Size(), 0);
    
    for (int i = 0; i < 50; ++i) {
        arr.PushBack(ComplexObject(i));
    }
    
    EXPECT_GE(arr.Allocated(), 100);
    EXPECT_EQ(arr.Size(), 50);
}

TEST_F(XClassArrayTest, Expand) {
    XClassArray<ComplexObject> arr;
    
    arr.Expand(5);
    EXPECT_EQ(arr.Size(), 5);
    
    arr.Expand(3);
    EXPECT_EQ(arr.Size(), 8);
}

TEST_F(XClassArrayTest, Insert) {
    XClassArray<ComplexObject> arr;
    arr.PushBack(ComplexObject(10));
    arr.PushBack(ComplexObject(30));
    
    arr.Insert(arr.Begin() + 1, ComplexObject(20));
    
    EXPECT_EQ(arr.Size(), 3);
    EXPECT_EQ(arr[0].value, 10);
    EXPECT_EQ(arr[1].value, 20);
    EXPECT_EQ(arr[2].value, 30);
}

TEST_F(XClassArrayTest, Remove) {
    XClassArray<ComplexObject> arr;
    arr.PushBack(ComplexObject(10));
    arr.PushBack(ComplexObject(20));
    arr.PushBack(ComplexObject(30));
    
    arr.Remove(arr.Begin() + 1);
    
    EXPECT_EQ(arr.Size(), 2);
    EXPECT_EQ(arr[0].value, 10);
    EXPECT_EQ(arr[1].value, 30);
}

TEST_F(XClassArrayTest, Clear) {
    XClassArray<ComplexObject> arr;
    arr.PushBack(ComplexObject(1));
    arr.PushBack(ComplexObject(2));
    
    arr.Clear();
    EXPECT_EQ(arr.Size(), 0);
}

TEST_F(XClassArrayTest, Find) {
    XClassArray<ComplexObject> arr;
    arr.PushBack(ComplexObject(10));
    arr.PushBack(ComplexObject(20));
    arr.PushBack(ComplexObject(30));
    
    ComplexObject* found = arr.Find(ComplexObject(20));
    EXPECT_NE(found, arr.End());
    EXPECT_EQ(found->value, 20);
}

TEST_F(XClassArrayTest, Sort) {
    XClassArray<ComplexObject> arr;
    arr.PushBack(ComplexObject(30));
    arr.PushBack(ComplexObject(10));
    arr.PushBack(ComplexObject(20));
    
    arr.Sort();
    
    EXPECT_EQ(arr[0].value, 10);
    EXPECT_EQ(arr[1].value, 20);
    EXPECT_EQ(arr[2].value, 30);
}

TEST_F(XClassArrayTest, Swap) {
    XClassArray<ComplexObject> arr;
    arr.PushBack(ComplexObject(10));
    arr.PushBack(ComplexObject(20));
    arr.PushBack(ComplexObject(30));
    
    arr.Swap(0, 2);
    
    EXPECT_EQ(arr[0].value, 30);
    EXPECT_EQ(arr[2].value, 10);
}

TEST_F(XClassArrayTest, BeginEnd) {
    XClassArray<ComplexObject> arr;
    arr.PushBack(ComplexObject(100));
    arr.PushBack(ComplexObject(200));
    arr.PushBack(ComplexObject(300));
    
    EXPECT_EQ(arr.Begin()->value, 100);
    EXPECT_EQ((arr.End() - 1)->value, 300);
}

TEST_F(XClassArrayTest, Iterators) {
    XClassArray<ComplexObject> arr;
    arr.PushBack(ComplexObject(1));
    arr.PushBack(ComplexObject(2));
    arr.PushBack(ComplexObject(3));
    
    int expected = 1;
    for (auto it = arr.Begin(); it != arr.End(); ++it) {
        EXPECT_EQ(it->value, expected++);
    }
}

#if VX_HAS_CXX11
TEST_F(XClassArrayTest, MoveConstructor) {
    XClassArray<ComplexObject> arr1;
    arr1.PushBack(ComplexObject(1));
    arr1.PushBack(ComplexObject(2));
    
    XClassArray<ComplexObject> arr2(std::move(arr1));
    
    EXPECT_EQ(arr2.Size(), 2);
    EXPECT_EQ(arr1.Size(), 0);
}

TEST_F(XClassArrayTest, MoveAssignment) {
    XClassArray<ComplexObject> arr1;
    arr1.PushBack(ComplexObject(1));
    arr1.PushBack(ComplexObject(2));
    
    XClassArray<ComplexObject> arr2;
    arr2 = std::move(arr1);
    
    EXPECT_EQ(arr2.Size(), 2);
    EXPECT_EQ(arr1.Size(), 0);
}

TEST_F(XClassArrayTest, Cxx11InitializerListCtorAndAssign) {
    XClassArray<ComplexObject> arr{ComplexObject(1), ComplexObject(2), ComplexObject(3)};
    EXPECT_EQ(arr.Size(), 3);
    EXPECT_EQ(arr[0].value, 1);
    EXPECT_EQ(arr[1].value, 2);
    EXPECT_EQ(arr[2].value, 3);

    arr = {ComplexObject(7), ComplexObject(8)};
    EXPECT_EQ(arr.Size(), 2);
    EXPECT_EQ(arr[0].value, 7);
    EXPECT_EQ(arr[1].value, 8);
}

TEST_F(XClassArrayTest, Cxx11RvalueInsertAndEmplaceAndRangeFor) {
    XClassArray<ComplexObject> arr;

    ComplexObject a(10);
    arr.PushBack(std::move(a));
    arr.EmplaceBack(30);

    arr.Insert(arr.Begin() + 1, ComplexObject(20));
    arr.Emplace(arr.Begin(), 5);

    EXPECT_EQ(arr.Size(), 4);
    EXPECT_EQ(arr[0].value, 5);
    EXPECT_EQ(arr[1].value, 10);
    EXPECT_EQ(arr[2].value, 20);
    EXPECT_EQ(arr[3].value, 30);

    int sum = 0;
    for (const auto &x : arr) {
        sum += x.value;
    }
    EXPECT_EQ(sum, 5 + 10 + 20 + 30);
}
#endif

// ==============================================================================
// XBitArray Tests - Efficient Bit Flag Management
// ==============================================================================

class XBitArrayTest : public ::testing::Test {
};

TEST_F(XBitArrayTest, DefaultConstructor) {
    XBitArray arr;
    EXPECT_EQ(arr.Size(), 32); // Default is 1 DWORD = 32 bits
}

TEST_F(XBitArrayTest, ConstructorWithSize) {
    XBitArray arr(4); // 4 DWORDs = 128 bits
    EXPECT_EQ(arr.Size(), 128);
}

TEST_F(XBitArrayTest, SetAndIsSet) {
    XBitArray arr;
    
    EXPECT_EQ(arr.IsSet(5), 0);
    
    arr.Set(5);
    EXPECT_NE(arr.IsSet(5), 0);
    
    arr.Set(10);
    arr.Set(20);
    EXPECT_NE(arr.IsSet(10), 0);
    EXPECT_NE(arr.IsSet(20), 0);
    EXPECT_EQ(arr.IsSet(15), 0);
}

TEST_F(XBitArrayTest, UnsetBit) {
    XBitArray arr;
    
    arr.Set(5);
    EXPECT_NE(arr.IsSet(5), 0);
    
    arr.Unset(5);
    EXPECT_EQ(arr.IsSet(5), 0);
}

TEST_F(XBitArrayTest, TestSet) {
    XBitArray arr;
    
    // First set should return 1 (was previously 0)
    EXPECT_EQ(arr.TestSet(5), 1);
    
    // Second set should return 0 (was already 1)
    EXPECT_EQ(arr.TestSet(5), 0);
}

TEST_F(XBitArrayTest, TestUnset) {
    XBitArray arr;
    
    arr.Set(5);
    
    // First unset should return 1 (was previously 1)
    EXPECT_EQ(arr.TestUnset(5), 1);
    
    // Second unset should return 0 (was already 0)
    EXPECT_EQ(arr.TestUnset(5), 0);
}

TEST_F(XBitArrayTest, ClearAll) {
    XBitArray arr;
    arr.Set(0);
    arr.Set(10);
    arr.Set(20);
    
    arr.Clear();
    
    EXPECT_EQ(arr.IsSet(0), 0);
    EXPECT_EQ(arr.IsSet(10), 0);
    EXPECT_EQ(arr.IsSet(20), 0);
}

TEST_F(XBitArrayTest, FillAll) {
    XBitArray arr;
    
    arr.Fill();
    
    // Check several bits are set
    for (int i = 0; i < 32; ++i) {
        EXPECT_NE(arr.IsSet(i), 0);
    }
}

TEST_F(XBitArrayTest, AutomaticResize) {
    XBitArray arr(1); // 32 bits initially
    
    EXPECT_EQ(arr.Size(), 32);
    
    // Setting a bit beyond current size should auto-resize
    arr.Set(100);
    EXPECT_GE(arr.Size(), 101);
    EXPECT_NE(arr.IsSet(100), 0);
}

TEST_F(XBitArrayTest, OperatorBracket) {
    XBitArray arr;
    
    arr.Set(5);
    arr.Set(10);
    
    EXPECT_NE(arr[5], 0);
    EXPECT_NE(arr[10], 0);
    EXPECT_EQ(arr[7], 0);
}

TEST_F(XBitArrayTest, AppendBits) {
    XBitArray arr;
    
    // Append bits from value 0b1010 (10 in decimal), 4 bits
    arr.AppendBits(0, 0b1010, 4);
    
    EXPECT_EQ(arr.IsSet(0), 0);  // bit 0 of 1010
    EXPECT_NE(arr.IsSet(1), 0);  // bit 1 of 1010
    EXPECT_EQ(arr.IsSet(2), 0);  // bit 2 of 1010
    EXPECT_NE(arr.IsSet(3), 0);  // bit 3 of 1010
}

TEST_F(XBitArrayTest, BitwiseAnd) {
    XBitArray arr1(2);
    XBitArray arr2(2);
    
    arr1.Set(0);
    arr1.Set(1);
    arr1.Set(2);
    
    arr2.Set(1);
    arr2.Set(2);
    arr2.Set(3);
    
    arr1.And(arr2);
    
    EXPECT_EQ(arr1.IsSet(0), 0);   // 1 & 0 = 0
    EXPECT_NE(arr1.IsSet(1), 0);   // 1 & 1 = 1
    EXPECT_NE(arr1.IsSet(2), 0);   // 1 & 1 = 1
    EXPECT_EQ(arr1.IsSet(3), 0);   // 0 & 1 = 0
}

TEST_F(XBitArrayTest, BitwiseOr) {
    XBitArray arr1(2);
    XBitArray arr2(2);
    
    arr1.Set(0);
    arr1.Set(1);
    
    arr2.Set(1);
    arr2.Set(2);
    
    arr1.Or(arr2);
    
    EXPECT_NE(arr1.IsSet(0), 0);   // 1 | 0 = 1
    EXPECT_NE(arr1.IsSet(1), 0);   // 1 | 1 = 1
    EXPECT_NE(arr1.IsSet(2), 0);   // 0 | 1 = 1
    EXPECT_EQ(arr1.IsSet(3), 0);   // 0 | 0 = 0
}

TEST_F(XBitArrayTest, BitwiseXor) {
    XBitArray arr1(2);
    XBitArray arr2(2);
    
    arr1.Set(0);
    arr1.Set(1);
    arr1.Set(2);
    
    arr2.Set(1);
    arr2.Set(2);
    arr2.Set(3);
    
    arr1.XOr(arr2);
    
    EXPECT_NE(arr1.IsSet(0), 0);   // 1 ^ 0 = 1
    EXPECT_EQ(arr1.IsSet(1), 0);   // 1 ^ 1 = 0
    EXPECT_EQ(arr1.IsSet(2), 0);   // 1 ^ 1 = 0
    EXPECT_NE(arr1.IsSet(3), 0);   // 0 ^ 1 = 1
}

TEST_F(XBitArrayTest, BitwiseSubtraction) {
    XBitArray arr1(2);
    XBitArray arr2(2);
    
    arr1.Set(0);
    arr1.Set(1);
    arr1.Set(2);
    
    arr2.Set(1);
    arr2.Set(3);
    
    arr1 -= arr2;
    
    EXPECT_NE(arr1.IsSet(0), 0);   // 1 & ~0 = 1
    EXPECT_EQ(arr1.IsSet(1), 0);   // 1 & ~1 = 0
    EXPECT_NE(arr1.IsSet(2), 0);   // 1 & ~0 = 1
    EXPECT_EQ(arr1.IsSet(3), 0);   // 0 & ~1 = 0
}

TEST_F(XBitArrayTest, Invert) {
    XBitArray arr(1); // 32 bits
    
    arr.Set(0);
    arr.Set(5);
    arr.Set(10);
    
    arr.Invert();
    
    EXPECT_EQ(arr.IsSet(0), 0);
    EXPECT_EQ(arr.IsSet(5), 0);
    EXPECT_EQ(arr.IsSet(10), 0);
    EXPECT_NE(arr.IsSet(1), 0);
    EXPECT_NE(arr.IsSet(6), 0);
}

TEST_F(XBitArrayTest, CheckCommon) {
    XBitArray arr1(2);
    XBitArray arr2(2);
    
    arr1.Set(0);
    arr1.Set(5);
    
    arr2.Set(5);
    arr2.Set(10);
    
    EXPECT_TRUE(arr1.CheckCommon(arr2)); // Both have bit 5 set
    
    XBitArray arr3(2);
    arr3.Set(20);
    
    EXPECT_FALSE(arr1.CheckCommon(arr3)); // No common bits
}

TEST_F(XBitArrayTest, BitSet) {
    XBitArray arr(2);
    
    arr.Set(0);
    arr.Set(5);
    arr.Set(10);
    arr.Set(15);
    
    EXPECT_EQ(arr.BitSet(), 4);
}

TEST_F(XBitArrayTest, GetSetBitPosition) {
    XBitArray arr(2);
    
    arr.Set(5);
    arr.Set(10);
    arr.Set(20);
    
    EXPECT_EQ(arr.GetSetBitPosition(0), 5);   // First set bit
    EXPECT_EQ(arr.GetSetBitPosition(1), 10);  // Second set bit
    EXPECT_EQ(arr.GetSetBitPosition(2), 20);  // Third set bit
    EXPECT_EQ(arr.GetSetBitPosition(3), static_cast<size_t>(-1));  // Fourth set bit (doesn't exist)
}

TEST_F(XBitArrayTest, GetUnsetBitPosition) {
    XBitArray arr(1);
    arr.Fill();
    
    arr.Unset(5);
    arr.Unset(10);
    
    EXPECT_EQ(arr.GetUnsetBitPosition(0), static_cast<size_t>(5));   // First unset bit
    EXPECT_EQ(arr.GetUnsetBitPosition(1), static_cast<size_t>(10));  // Second unset bit
}

TEST_F(XBitArrayTest, ConvertToString) {
    XBitArray arr(1);
    
    arr.Set(0);
    arr.Set(2);
    arr.Set(4);
    
    char buffer[33];
    arr.ConvertToString(buffer);
    
    EXPECT_EQ(buffer[0], '1');
    EXPECT_EQ(buffer[1], '0');
    EXPECT_EQ(buffer[2], '1');
    EXPECT_EQ(buffer[3], '0');
    EXPECT_EQ(buffer[4], '1');
    EXPECT_EQ(buffer[31], '0');
    EXPECT_EQ(buffer[32], '\0');
}

TEST_F(XBitArrayTest, CopyConstructor) {
    XBitArray original(2);
    original.Set(5);
    original.Set(10);
    original.Set(50);
    
    XBitArray copy(original);
    
    EXPECT_EQ(copy.Size(), original.Size());
    EXPECT_NE(copy.IsSet(5), 0);
    EXPECT_NE(copy.IsSet(10), 0);
    EXPECT_NE(copy.IsSet(50), 0);
    EXPECT_EQ(copy.IsSet(20), 0);
}

TEST_F(XBitArrayTest, Assignment) {
    XBitArray original(2);
    original.Set(5);
    original.Set(10);
    
    XBitArray assigned(1);
    assigned = original;
    
    EXPECT_EQ(assigned.Size(), original.Size());
    EXPECT_NE(assigned.IsSet(5), 0);
    EXPECT_NE(assigned.IsSet(10), 0);
}

#if VX_HAS_CXX11
TEST_F(XBitArrayTest, MoveConstructor) {
    XBitArray arr1(2);
    arr1.Set(5);
    arr1.Set(10);
    arr1.Set(50);
    
    XBitArray arr2(std::move(arr1));
    
    EXPECT_NE(arr2.IsSet(5), 0);
    EXPECT_NE(arr2.IsSet(10), 0);
    EXPECT_NE(arr2.IsSet(50), 0);
    EXPECT_EQ(arr1.Size(), 0);
}

TEST_F(XBitArrayTest, MoveAssignment) {
    XBitArray arr1(2);
    arr1.Set(5);
    arr1.Set(10);
    
    XBitArray arr2(1);
    arr2 = std::move(arr1);
    
    EXPECT_NE(arr2.IsSet(5), 0);
    EXPECT_NE(arr2.IsSet(10), 0);
    EXPECT_EQ(arr1.Size(), 0);
}

TEST_F(XBitArrayTest, Cxx11InitializerListCtorAndAssign) {
    XBitArray bits{0, 2, 5};
    EXPECT_GE(bits.Size(), 6);
    EXPECT_NE(bits.IsSet(0), 0);
    EXPECT_EQ(bits.IsSet(1), 0);
    EXPECT_NE(bits.IsSet(2), 0);
    EXPECT_NE(bits.IsSet(5), 0);

    bits = {1, 3};
    EXPECT_EQ(bits.IsSet(0), 0);
    EXPECT_NE(bits.IsSet(1), 0);
    EXPECT_EQ(bits.IsSet(2), 0);
    EXPECT_NE(bits.IsSet(3), 0);
    EXPECT_EQ(bits.IsSet(5), 0);
}

TEST_F(XBitArrayTest, Cxx11Swap) {
    XBitArray a{1, 4};
    XBitArray b{0, 31};

    a.Swap(b);

    EXPECT_NE(a.IsSet(0), 0);
    EXPECT_NE(a.IsSet(31), 0);
    EXPECT_EQ(a.IsSet(1), 0);
    EXPECT_EQ(a.IsSet(4), 0);

    EXPECT_EQ(b.IsSet(0), 0);
    EXPECT_EQ(b.IsSet(31), 0);
    EXPECT_NE(b.IsSet(1), 0);
    EXPECT_NE(b.IsSet(4), 0);
}
#endif

TEST_F(XBitArrayTest, GetMemoryOccupation) {
    XBitArray arr(4); // 4 DWORDs
    
    size_t mem = arr.GetMemoryOccupation(FALSE);
    EXPECT_EQ(mem, 4 * sizeof(XDWORD));
    
    size_t memWithStatic = arr.GetMemoryOccupation(TRUE);
    EXPECT_GT(memWithStatic, mem);
}

TEST_F(XBitArrayTest, CheckSameSize) {
    XBitArray arr1(2); // 64 bits
    XBitArray arr2(4); // 128 bits
    
    EXPECT_EQ(arr1.Size(), 64);
    
    arr1.CheckSameSize(arr2);
    
    EXPECT_EQ(arr1.Size(), arr2.Size());
}

TEST_F(XBitArrayTest, LargeBitArray) {
    XBitArray arr(32); // 1024 bits
    
    for (int i = 0; i < 1000; i += 10) {
        arr.Set(i);
    }
    
    EXPECT_EQ(arr.BitSet(), 100);
    
    for (int i = 0; i < 1000; i += 10) {
        EXPECT_NE(arr.IsSet(i), 0);
    }
}
