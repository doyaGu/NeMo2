#include "gtest/gtest.h"

#include "XList.h"

#if VX_HAS_CXX11
#include <string>
#include <utility>
#endif

// ============================================================================
// Basic Operations Tests
// ============================================================================

TEST(XListTest, DefaultConstructor) {
    XList<int> list;
    EXPECT_EQ(list.Size(), 0);
    EXPECT_TRUE(list.IsEmpty());
    EXPECT_EQ(list.Begin(), list.End());
}

TEST(XListTest, BasicPushPopAndIteration) {
    XList<int> list;
    EXPECT_EQ(list.Size(), 0);

    list.PushBack(1);
    list.PushBack(2);
    list.PushFront(0);

    EXPECT_EQ(list.Size(), 3);
    EXPECT_EQ(list.Front(), 0);
    EXPECT_EQ(list.Back(), 2);

    int sum = 0;
    for (XList<int>::Iterator it = list.Begin(); it != list.End(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, 3);

    list.PopFront();
    EXPECT_EQ(list.Size(), 2);
    list.PopBack();
    EXPECT_EQ(list.Size(), 1);
    EXPECT_EQ(list.Front(), 1);
}

TEST(XListTest, PushBackMultipleElements) {
    XList<int> list;
    for (int i = 0; i < 10; ++i) {
        list.PushBack(i);
    }
    EXPECT_EQ(list.Size(), 10);
    EXPECT_EQ(list.Front(), 0);
    EXPECT_EQ(list.Back(), 9);
}

TEST(XListTest, PushFrontMultipleElements) {
    XList<int> list;
    for (int i = 0; i < 10; ++i) {
        list.PushFront(i);
    }
    EXPECT_EQ(list.Size(), 10);
    EXPECT_EQ(list.Front(), 9);
    EXPECT_EQ(list.Back(), 0);
}

TEST(XListTest, FrontAndBackReferences) {
    XList<int> list;
    list.PushBack(10);
    list.PushBack(20);

    // Test non-const reference
    list.Front() = 15;
    list.Back() = 25;

    EXPECT_EQ(list.Front(), 15);
    EXPECT_EQ(list.Back(), 25);
}

TEST(XListTest, Clear) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);
    EXPECT_EQ(list.Size(), 3);

    list.Clear();
    EXPECT_EQ(list.Size(), 0);
    EXPECT_TRUE(list.IsEmpty());
    EXPECT_EQ(list.Begin(), list.End());

    // Should be able to add elements after clear
    list.PushBack(4);
    EXPECT_EQ(list.Size(), 1);
    EXPECT_EQ(list.Front(), 4);
}

TEST(XListTest, IsEmpty) {
    XList<int> list;
    EXPECT_TRUE(list.IsEmpty());

    list.PushBack(1);
    EXPECT_FALSE(list.IsEmpty());

    list.PopBack();
    EXPECT_TRUE(list.IsEmpty());
}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST(XListTest, IteratorDefaultConstructor) {
    XList<int>::Iterator it;
    XList<int>::Iterator it2;
    EXPECT_EQ(it, it2);
}

TEST(XListTest, IteratorCopyConstructor) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);

    XList<int>::Iterator it1 = list.Begin();
    XList<int>::Iterator it2(it1);

    EXPECT_EQ(it1, it2);
    EXPECT_EQ(*it1, *it2);
}

TEST(XListTest, IteratorPreIncrement) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    XList<int>::Iterator it = list.Begin();
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 2);
    ++it;
    EXPECT_EQ(*it, 3);
    ++it;
    EXPECT_EQ(it, list.End());
}

TEST(XListTest, IteratorPostIncrement) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);

    XList<int>::Iterator it = list.Begin();
    XList<int>::Iterator old = it++;
    EXPECT_EQ(*old, 1);
    EXPECT_EQ(*it, 2);
}

TEST(XListTest, IteratorPreDecrement) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    XList<int>::Iterator it = list.End();
    --it;
    EXPECT_EQ(*it, 3);
    --it;
    EXPECT_EQ(*it, 2);
    --it;
    EXPECT_EQ(*it, 1);
    EXPECT_EQ(it, list.Begin());
}

TEST(XListTest, IteratorPostDecrement) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);

    XList<int>::Iterator it = list.End();
    XList<int>::Iterator old = it--;
    EXPECT_EQ(old, list.End());
    EXPECT_EQ(*it, 2);
}

TEST(XListTest, IteratorPlusOffset) {
    XList<int> list;
    for (int i = 0; i < 5; ++i) {
        list.PushBack(i);
    }

    XList<int>::Iterator it = list.Begin();
    XList<int>::Iterator it2 = it + 2;
    EXPECT_EQ(*it2, 2);

    XList<int>::Iterator it3 = it + 4;
    EXPECT_EQ(*it3, 4);
}

TEST(XListTest, IteratorMinusOffset) {
    XList<int> list;
    for (int i = 0; i < 5; ++i) {
        list.PushBack(i);
    }

    XList<int>::Iterator it = list.End();
    it = it - 1;
    EXPECT_EQ(*it, 4);

    it = it - 2;
    EXPECT_EQ(*it, 2);
}

TEST(XListTest, IteratorEquality) {
    XList<int> list;
    list.PushBack(1);

    XList<int>::Iterator it1 = list.Begin();
    XList<int>::Iterator it2 = list.Begin();
    EXPECT_TRUE(it1 == it2);
    EXPECT_FALSE(it1 != it2);

    ++it2;
    EXPECT_FALSE(it1 == it2);
    EXPECT_TRUE(it1 != it2);
}

TEST(XListTest, IteratorDereference) {
    XList<int> list;
    list.PushBack(42);

    XList<int>::Iterator it = list.Begin();
    EXPECT_EQ(*it, 42);

    *it = 100;
    EXPECT_EQ(*it, 100);
    EXPECT_EQ(list.Front(), 100);
}

TEST(XListTest, ConstIterator) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    const XList<int> &constList = list;
    int sum = 0;
    for (XList<int>::Iterator it = constList.Begin(); it != constList.End(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, 6);
}

// ============================================================================
// Copy Constructor and Assignment Tests
// ============================================================================

TEST(XListTest, CopyConstructor) {
    XList<int> list1;
    list1.PushBack(1);
    list1.PushBack(2);
    list1.PushBack(3);

    XList<int> list2(list1);
    EXPECT_EQ(list2.Size(), 3);
    EXPECT_EQ(list2.Front(), 1);
    EXPECT_EQ(list2.Back(), 3);

    // Verify deep copy - modifying list2 shouldn't affect list1
    list2.PushBack(4);
    EXPECT_EQ(list1.Size(), 3);
    EXPECT_EQ(list2.Size(), 4);
}

TEST(XListTest, CopyConstructorEmptyList) {
    XList<int> list1;
    XList<int> list2(list1);

    EXPECT_TRUE(list2.IsEmpty());
    EXPECT_EQ(list2.Size(), 0);
}

TEST(XListTest, CopyAssignment) {
    XList<int> list1;
    list1.PushBack(1);
    list1.PushBack(2);

    XList<int> list2;
    list2.PushBack(99);

    list2 = list1;
    EXPECT_EQ(list2.Size(), 2);
    EXPECT_EQ(list2.Front(), 1);
    EXPECT_EQ(list2.Back(), 2);

    // Verify deep copy
    list2.PushBack(3);
    EXPECT_EQ(list1.Size(), 2);
    EXPECT_EQ(list2.Size(), 3);
}

TEST(XListTest, SelfAssignment) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    list = list;
    EXPECT_EQ(list.Size(), 3);
    EXPECT_EQ(list.Front(), 1);
    EXPECT_EQ(list.Back(), 3);
}

// ============================================================================
// Search and Find Tests
// ============================================================================

TEST(XListTest, FindExistingElement) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    XList<int>::Iterator it = list.Find(2);
    EXPECT_NE(it, list.End());
    EXPECT_EQ(*it, 2);
}

TEST(XListTest, FindNonExistingElement) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);

    XList<int>::Iterator it = list.Find(99);
    EXPECT_EQ(it, list.End());
}

TEST(XListTest, FindFirstOccurrence) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(2);
    list.PushBack(3);

    XList<int>::Iterator it = list.Find(2);
    EXPECT_NE(it, list.End());
    EXPECT_EQ(*it, 2);

    // Verify it's the first occurrence
    ++it;
    EXPECT_EQ(*it, 2);
}

TEST(XListTest, FindWithStartPosition) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);
    list.PushBack(2);

    XList<int>::Iterator start = list.Begin();
    ++start; // Start from second element
    ++start; // Start from third element

    XList<int>::Iterator it = list.Find(start, 2);
    EXPECT_NE(it, list.End());
    EXPECT_EQ(*it, 2);

    // Verify it's the second occurrence
    --it;
    EXPECT_EQ(*it, 3);
}

TEST(XListTest, IsHereTrue) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    EXPECT_TRUE(list.IsHere(2));
    EXPECT_TRUE(list.IsHere(1));
    EXPECT_TRUE(list.IsHere(3));
}

TEST(XListTest, IsHereFalse) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);

    EXPECT_FALSE(list.IsHere(99));
    EXPECT_FALSE(list.IsHere(0));
}

TEST(XListTest, IsHereEmptyList) {
    XList<int> list;
    EXPECT_FALSE(list.IsHere(1));
}

// ============================================================================
// Remove Tests
// ============================================================================

TEST(XListTest, RemoveByValue) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    XBOOL removed = list.Remove(2);
    EXPECT_TRUE(removed);
    EXPECT_EQ(list.Size(), 2);
    EXPECT_FALSE(list.IsHere(2));
}

TEST(XListTest, RemoveNonExistingElement) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);

    XBOOL removed = list.Remove(99);
    EXPECT_FALSE(removed);
    EXPECT_EQ(list.Size(), 2);
}

TEST(XListTest, RemoveFirstOccurrenceOnly) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(2);
    list.PushBack(3);

    XBOOL removed = list.Remove(2);
    EXPECT_TRUE(removed);
    EXPECT_EQ(list.Size(), 3);
    EXPECT_TRUE(list.IsHere(2)); // Second occurrence still present
}

TEST(XListTest, RemoveByIterator) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    XList<int>::Iterator it = list.Begin();
    ++it; // Point to 2

    XList<int>::Iterator next = list.Remove(it);
    EXPECT_EQ(list.Size(), 2);
    EXPECT_EQ(*next, 3);
    EXPECT_FALSE(list.IsHere(2));
}

TEST(XListTest, RemoveFirstElement) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    XList<int>::Iterator it = list.Begin();
    list.Remove(it);

    EXPECT_EQ(list.Size(), 2);
    EXPECT_EQ(list.Front(), 2);
}

TEST(XListTest, RemoveLastElement) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);
    list.PushBack(3);

    XList<int>::Iterator it = list.End();
    --it; // Point to last element

    XList<int>::Iterator next = list.Remove(it);
    EXPECT_EQ(next, list.End());
    EXPECT_EQ(list.Size(), 2);
    EXPECT_EQ(list.Back(), 2);
}

// ============================================================================
// Insert Tests
// ============================================================================

TEST(XListTest, InsertAtBeginning) {
    XList<int> list;
    list.PushBack(2);
    list.PushBack(3);

    XList<int>::Iterator it = list.Begin();
    list.Insert(it, 1);

    EXPECT_EQ(list.Size(), 3);
    EXPECT_EQ(list.Front(), 1);
}

TEST(XListTest, InsertAtEnd) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);

    XList<int>::Iterator it = list.End();
    list.Insert(it, 3);

    EXPECT_EQ(list.Size(), 3);
    EXPECT_EQ(list.Back(), 3);
}

TEST(XListTest, InsertInMiddle) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(3);

    XList<int>::Iterator it = list.Begin();
    ++it; // Point to 3
    list.Insert(it, 2);

    EXPECT_EQ(list.Size(), 3);

    it = list.Begin();
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 2);
    ++it;
    EXPECT_EQ(*it, 3);
}

TEST(XListTest, InsertIntoEmptyList) {
    XList<int> list;
    XList<int>::Iterator it = list.Begin();
    list.Insert(it, 1);

    EXPECT_EQ(list.Size(), 1);
    EXPECT_EQ(list.Front(), 1);
}

// ============================================================================
// Swap Tests
// ============================================================================

TEST(XListTest, Swap) {
    XList<int> list1;
    list1.PushBack(1);
    list1.PushBack(2);

    XList<int> list2;
    list2.PushBack(3);
    list2.PushBack(4);
    list2.PushBack(5);

    list1.Swap(list2);

    EXPECT_EQ(list1.Size(), 3);
    EXPECT_EQ(list1.Front(), 3);
    EXPECT_EQ(list1.Back(), 5);

    EXPECT_EQ(list2.Size(), 2);
    EXPECT_EQ(list2.Front(), 1);
    EXPECT_EQ(list2.Back(), 2);
}

TEST(XListTest, SwapWithEmptyList) {
    XList<int> list1;
    list1.PushBack(1);
    list1.PushBack(2);

    XList<int> list2;

    list1.Swap(list2);

    EXPECT_TRUE(list1.IsEmpty());
    EXPECT_EQ(list2.Size(), 2);
    EXPECT_EQ(list2.Front(), 1);
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST(XListTest, SingleElementList) {
    XList<int> list;
    list.PushBack(42);

    EXPECT_EQ(list.Size(), 1);
    EXPECT_EQ(list.Front(), 42);
    EXPECT_EQ(list.Back(), 42);
    EXPECT_FALSE(list.IsEmpty());

    XList<int>::Iterator it = list.Begin();
    EXPECT_EQ(*it, 42);
    ++it;
    EXPECT_EQ(it, list.End());
}

TEST(XListTest, SingleElementPopFront) {
    XList<int> list;
    list.PushBack(42);
    list.PopFront();

    EXPECT_TRUE(list.IsEmpty());
    EXPECT_EQ(list.Size(), 0);
}

TEST(XListTest, SingleElementPopBack) {
    XList<int> list;
    list.PushBack(42);
    list.PopBack();

    EXPECT_TRUE(list.IsEmpty());
    EXPECT_EQ(list.Size(), 0);
}

TEST(XListTest, LargeList) {
    XList<int> list;
    const int count = 1000;

    for (int i = 0; i < count; ++i) {
        list.PushBack(i);
    }

    EXPECT_EQ(list.Size(), count);
    EXPECT_EQ(list.Front(), 0);
    EXPECT_EQ(list.Back(), count - 1);

    int sum = 0;
    for (XList<int>::Iterator it = list.Begin(); it != list.End(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, (count - 1) * count / 2);
}

TEST(XListTest, AlternatingPushPopOperations) {
    XList<int> list;

    list.PushBack(1);
    list.PushFront(0);
    EXPECT_EQ(list.Size(), 2);

    list.PopBack();
    EXPECT_EQ(list.Size(), 1);

    list.PushBack(2);
    EXPECT_EQ(list.Size(), 2);

    list.PopFront();
    EXPECT_EQ(list.Size(), 1);
    EXPECT_EQ(list.Front(), 2);
}

TEST(XListTest, MultipleIdenticalElements) {
    XList<int> list;
    list.PushBack(5);
    list.PushBack(5);
    list.PushBack(5);

    EXPECT_EQ(list.Size(), 3);
    EXPECT_TRUE(list.IsHere(5));

    XList<int>::Iterator it = list.Find(5);
    EXPECT_EQ(*it, 5);
    ++it;
    EXPECT_EQ(*it, 5);
    ++it;
    EXPECT_EQ(*it, 5);
}

#if VX_HAS_CXX11
// ============================================================================
// C++11 Features Tests
// ============================================================================

TEST(XListTest, Cxx11MoveInitListEmplaceAndRangeFor) {
    XList<std::string> a({"a", "b", "c"});
    EXPECT_EQ(a.Size(), 3);
    EXPECT_FALSE(a.IsEmpty());

    // range-for via begin/end wrappers
    std::string joined;
    for (const std::string &s : a) {
        joined += s;
    }
    EXPECT_EQ(joined, "abc");

    // EmplaceFront/Back
    a.EmplaceFront(1, 'z');
    a.EmplaceBack(2, 'y');
    EXPECT_EQ(a.Size(), 5);

    // Insert (move) + Emplace before position
    XList<std::string>::Iterator pos = a.Begin();
    ++pos; // after "z"
    a.Insert(pos, std::string("X"));
    a.Emplace(pos, 1, 'Q');

    // Move ctor leaves source empty
    XList<std::string> moved(std::move(a));
    EXPECT_EQ(moved.Size(), 7);
    EXPECT_TRUE(a.IsEmpty());

    // Move assignment leaves source empty
    XList<std::string> b;
    b = std::move(moved);
    EXPECT_EQ(b.Size(), 7);
    EXPECT_TRUE(moved.IsEmpty());

    // initializer_list assignment
    b = {"u", "v"};
    EXPECT_EQ(b.Size(), 2);
    EXPECT_EQ(b.Front(), "u");
    EXPECT_EQ(b.Back(), "v");
}

TEST(XListTest, InitializerListConstructor) {
    XList<int> list = {1, 2, 3, 4, 5};

    EXPECT_EQ(list.Size(), 5);
    EXPECT_EQ(list.Front(), 1);
    EXPECT_EQ(list.Back(), 5);

    int expected = 1;
    for (const int &val : list) {
        EXPECT_EQ(val, expected++);
    }
}

TEST(XListTest, InitializerListAssignment) {
    XList<int> list;
    list.PushBack(99);

    list = {10, 20, 30};

    EXPECT_EQ(list.Size(), 3);
    EXPECT_EQ(list.Front(), 10);
    EXPECT_EQ(list.Back(), 30);
}

TEST(XListTest, EmptyInitializerList) {
    XList<int> list = {};

    EXPECT_TRUE(list.IsEmpty());
    EXPECT_EQ(list.Size(), 0);
}

TEST(XListTest, MoveConstructor) {
    XList<int> list1;
    list1.PushBack(1);
    list1.PushBack(2);
    list1.PushBack(3);

    XList<int> list2(std::move(list1));

    EXPECT_TRUE(list1.IsEmpty());
    EXPECT_EQ(list2.Size(), 3);
    EXPECT_EQ(list2.Front(), 1);
    EXPECT_EQ(list2.Back(), 3);
}

TEST(XListTest, MoveAssignment) {
    XList<int> list1;
    list1.PushBack(1);
    list1.PushBack(2);

    XList<int> list2;
    list2.PushBack(99);

    list2 = std::move(list1);

    EXPECT_TRUE(list1.IsEmpty());
    EXPECT_EQ(list2.Size(), 2);
    EXPECT_EQ(list2.Front(), 1);
    EXPECT_EQ(list2.Back(), 2);
}

TEST(XListTest, MoveSelfAssignment) {
    XList<int> list;
    list.PushBack(1);
    list.PushBack(2);

    list = std::move(list);

    // After self-move-assignment, the list should be empty
    // (Swap with self leaves it unchanged, then Clear empties it)
    EXPECT_TRUE(list.IsEmpty());
}

TEST(XListTest, PushBackMove) {
    XList<std::string> list;
    std::string s = "hello";
    list.PushBack(std::move(s));

    EXPECT_EQ(list.Front(), "hello");
    // s should be in moved-from state (typically empty for std::string)
}

TEST(XListTest, PushFrontMove) {
    XList<std::string> list;
    list.PushBack("world");

    std::string s = "hello";
    list.PushFront(std::move(s));

    EXPECT_EQ(list.Front(), "hello");
    EXPECT_EQ(list.Back(), "world");
}

TEST(XListTest, InsertMove) {
    XList<std::string> list;
    list.PushBack("first");
    list.PushBack("third");

    std::string s = "second";
    XList<std::string>::Iterator it = list.Begin();
    ++it;
    list.Insert(it, std::move(s));

    EXPECT_EQ(list.Size(), 3);
    it = list.Begin();
    EXPECT_EQ(*it, "first");
    ++it;
    EXPECT_EQ(*it, "second");
    ++it;
    EXPECT_EQ(*it, "third");
}

TEST(XListTest, EmplaceFront) {
    XList<std::string> list;
    list.PushBack("world");

    list.EmplaceFront(5, 'h'); // Create string with 5 'h's

    EXPECT_EQ(list.Front(), "hhhhh");
    EXPECT_EQ(list.Size(), 2);
}

TEST(XListTest, EmplaceBack) {
    XList<std::string> list;
    list.PushBack("hello");

    list.EmplaceBack(3, '!'); // Create string with 3 '!'s

    EXPECT_EQ(list.Back(), "!!!");
    EXPECT_EQ(list.Size(), 2);
}

TEST(XListTest, EmplaceAtPosition) {
    XList<std::string> list;
    list.PushBack("first");
    list.PushBack("third");

    XList<std::string>::Iterator it = list.Begin();
    ++it;
    list.Emplace(it, 6, 'x'); // Create string with 6 'x's

    EXPECT_EQ(list.Size(), 3);
    it = list.Begin();
    ++it;
    EXPECT_EQ(*it, "xxxxxx");
}

TEST(XListTest, RangeBasedForLoop) {
    XList<int> list = {1, 2, 3, 4, 5};

    int sum = 0;
    for (int val : list) {
        sum += val;
    }
    EXPECT_EQ(sum, 15);
}

TEST(XListTest, RangeBasedForLoopModify) {
    XList<int> list = {1, 2, 3};

    for (int &val : list) {
        val *= 2;
    }

    EXPECT_EQ(list.Front(), 2);
    XList<int>::Iterator it = list.Begin();
    ++it;
    EXPECT_EQ(*it, 4);
    ++it;
    EXPECT_EQ(*it, 6);
}

TEST(XListTest, ConstRangeBasedForLoop) {
    const XList<int> list = {1, 2, 3};

    int sum = 0;
    for (const int &val : list) {
        sum += val;
    }
    EXPECT_EQ(sum, 6);
}

TEST(XListTest, CBeginCEnd) {
    XList<int> list = {1, 2, 3};

    int sum = 0;
    for (XList<int>::Iterator it = list.cbegin(); it != list.cend(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, 6);
}

#endif
