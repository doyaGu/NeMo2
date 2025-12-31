#include <gtest/gtest.h>

#include <set>
#include <utility>
#include <vector>

#include "XHashTable.h"
#include "XNHashTable.h"
#include "XSHashTable.h"

namespace {

template <class Table>
static std::set<int> CollectKeys(const Table& t) {
    std::set<int> keys;
    int guard = 0;
    const int maxSteps = t.Size() + 10; // +slack for empty buckets
    for (auto it = t.Begin(); it != t.End(); ++it) {
        keys.insert(it.GetKey());
        if (++guard > maxSteps) {
            // If this trips, iterator likely entered a cycle.
            ADD_FAILURE() << "Iterator did not terminate (possible cycle)";
            break;
        }
    }
    return keys;
}

struct Key {
    int v;
};

struct KeyEq {
    int operator()(const Key& a, const Key& b) const { return a.v == b.v; }
};

struct KeyHashIdentity {
    int operator()(const Key& k) const { return k.v; }
};

struct KeyHashConstant {
    int operator()(const Key&) const { return 0; }
};

struct IntHashConstantZero {
    int operator()(const int&) const { return 0; }
};

struct IntHashNegativeOne {
    int operator()(const int&) const { return -1; }
};

static void ExpectAllPresent(const XHashTable<int, int>& t, int begin, int end, int scale) {
    for (int i = begin; i < end; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr) << "Missing key " << i;
        EXPECT_EQ(*p, i * scale);
    }
}

TEST(XHashTable, EmptyBeginEnd) {
    XHashTable<int, int> t;
    EXPECT_EQ(t.Size(), 0);
    EXPECT_EQ(t.Begin(), t.End());

    const XHashTable<int, int>& ct = t;
    EXPECT_EQ(ct.Begin(), ct.End());
}

TEST(XHashTable, InsertOverrideFlag) {
    XHashTable<int, int> t(4);
    EXPECT_TRUE(t.Insert(1, 111, TRUE));
    EXPECT_EQ(*t.FindPtr(1), 111);

    // override = FALSE should not update existing value.
    EXPECT_FALSE(t.Insert(1, 222, FALSE));
    EXPECT_EQ(*t.FindPtr(1), 111);

    // override = TRUE should update.
    EXPECT_TRUE(t.Insert(1, 333, TRUE));
    EXPECT_EQ(*t.FindPtr(1), 333);
}

TEST(XHashTable, InsertOverwritesByDefault) {
    XHashTable<int, int> t(4);
    auto it1 = t.Insert(7, 70);
    EXPECT_NE(it1, t.End());
    EXPECT_EQ(*it1, 70);

    auto it2 = t.Insert(7, 700);
    EXPECT_NE(it2, t.End());
    EXPECT_EQ(*it2, 700);
    EXPECT_EQ(t.Size(), 1);
}

TEST(XHashTable, InsertUniqueDoesNotOverwrite) {
    XHashTable<int, int> t(4);
    auto it1 = t.InsertUnique(9, 90);
    EXPECT_EQ(*it1, 90);
    auto it2 = t.InsertUnique(9, 900);
    EXPECT_EQ(*it2, 90);
    EXPECT_EQ(t.Size(), 1);
}

TEST(XHashTable, TestInsertReturnsNewFlagAndDoesNotOverwrite) {
    XHashTable<int, int> t(4);
    auto p1 = t.TestInsert(5, 50);
    EXPECT_TRUE(p1.m_New);
    EXPECT_EQ(*p1.m_Iterator, 50);
    EXPECT_EQ(t.Size(), 1);

    auto p2 = t.TestInsert(5, 500);
    EXPECT_FALSE(p2.m_New);
    EXPECT_EQ(*p2.m_Iterator, 50);
    EXPECT_EQ(t.Size(), 1);
}

TEST(XHashTable, OperatorBracketDefaultInsertion) {
    XHashTable<int, int> t(4);
    EXPECT_FALSE(t.IsHere(123));
    int& v = t[123];
    EXPECT_TRUE(t.IsHere(123));
    EXPECT_EQ(v, 0);
    v = 77;
    EXPECT_EQ(*t.FindPtr(123), 77);
}

TEST(XHashTable, RemoveAbsentKeyNoChange) {
    XHashTable<int, int> t(4);
    t.Insert(1, 10);
    t.Insert(2, 20);
    EXPECT_EQ(t.Size(), 2);

    t.Remove(999);
    EXPECT_EQ(t.Size(), 2);
    EXPECT_EQ(*t.FindPtr(1), 10);
    EXPECT_EQ(*t.FindPtr(2), 20);
}

TEST(XHashTable, ClearThenReuse) {
    XHashTable<int, int> t(8);
    for (int i = 0; i < 30; ++i)
        t.Insert(i, i * 2);
    ASSERT_EQ(t.Size(), 30);

    t.Clear();
    EXPECT_EQ(t.Size(), 0);
    EXPECT_EQ(t.Begin(), t.End());
    EXPECT_FALSE(t.IsHere(0));

    for (int i = 0; i < 30; ++i)
        t.Insert(i, i * 3);
    EXPECT_EQ(t.Size(), 30);
    ExpectAllPresent(t, 0, 30, 3);
}

TEST(XHashTable, ReserveOnEmptyDoesNotBreakInsertion) {
    XHashTable<int, int> t(4);
    t.Clear();
    t.Reserve(1000);
    EXPECT_EQ(t.Size(), 0);
    EXPECT_TRUE(t.Insert(1, 10, TRUE));
    EXPECT_EQ(*t.FindPtr(1), 10);
}

TEST(XHashTable, ReservePreservesElements) {
    XHashTable<int, int> t(4);

    for (int i = 0; i < 10; ++i)
        t.Insert(i, i * 10);
    ASSERT_EQ(t.Size(), 10);

    // Reserve should not lose entries.
    t.Reserve(1000);
    EXPECT_EQ(t.Size(), 10);
    ExpectAllPresent(t, 0, 10, 10);

    // Ensure we can keep inserting after reserve.
    for (int i = 10; i < 100; ++i)
        t.Insert(i, i * 10);
    EXPECT_EQ(t.Size(), 100);
    ExpectAllPresent(t, 0, 100, 10);
}

TEST(XHashTable, ReserveDoesNotShrinkAndKeepsElements) {
    XHashTable<int, int> t(16);
    for (int i = 0; i < 200; ++i)
        t.Insert(i, i);

    // Reserve smaller than current size should be a no-op on content.
    t.Reserve(10);
    EXPECT_EQ(t.Size(), 200);
    ExpectAllPresent(t, 0, 200, 1);
}

TEST(XHashTable, RehashTriggeredByPoolFullPreservesAllElementsAcrossManyRounds) {
    // initialize=4 => pool reserved ~3, so we hit rehash quickly.
    XHashTable<int, int> t(4);

    const int N = 5000;
    for (int i = 0; i < N; ++i) {
        t.Insert(i, i * 17);
        // Spot-check periodically to catch transient corruption during rehash.
        if ((i % 257) == 0) {
            int* p = t.FindPtr(i);
            ASSERT_NE(p, nullptr);
            EXPECT_EQ(*p, i * 17);
        }
    }

    EXPECT_EQ(t.Size(), N);
    ExpectAllPresent(t, 0, N, 17);

    // Iteration should cover exactly Size() elements without cycling.
    auto keys = CollectKeys(t);
    EXPECT_EQ((int)keys.size(), t.Size());
}

TEST(XHashTable, RehashWithAllKeysInSingleBucketDoesNotCorruptChains) {
    // Force worst-case chaining: all keys land in same bucket.
    XHashTable<int, int, IntHashConstantZero> t(4);

    const int N = 2000;
    for (int i = 0; i < N; ++i)
        t.Insert(i, i * 9);

    EXPECT_EQ(t.Size(), N);
    for (int i = 0; i < N; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr) << "Missing key " << i;
        EXPECT_EQ(*p, i * 9);
    }

    // Ensure iterator terminates and covers all keys.
    auto keys = CollectKeys(t);
    EXPECT_EQ((int)keys.size(), t.Size());
}

TEST(XHashTable, RemoveManyAfterHeavyRehashKeepsRemainingAccessible) {
    XHashTable<int, int> t(4);
    const int N = 3000;
    for (int i = 0; i < N; ++i)
        t.Insert(i, i * 5);
    ASSERT_EQ(t.Size(), N);

    // Remove every 3rd key to stress FastRemove remap on a large pool.
    for (int i = 0; i < N; i += 3)
        t.Remove(i);

    EXPECT_EQ(t.Size(), N - (N + 2) / 3);
    for (int i = 0; i < N; ++i) {
        if ((i % 3) == 0) {
            EXPECT_FALSE(t.IsHere(i));
        } else {
            int* p = t.FindPtr(i);
            ASSERT_NE(p, nullptr) << "Missing key " << i;
            EXPECT_EQ(*p, i * 5);
        }
    }
}

TEST(XHashTable, EraseWhileIteratingSurvivesRehashPressure) {
    // Create a table that rehashes multiple times.
    XHashTable<int, int> t(4);
    for (int i = 0; i < 5000; ++i)
        t.Insert(i, i);
    ASSERT_EQ(t.Size(), 5000);

    // Erase a pattern while iterating.
    for (auto it = t.Begin(); it != t.End();) {
        const int k = it.GetKey();
        if ((k % 5) == 0) {
            it = t.Remove(it);
        } else {
            ++it;
        }
    }

    EXPECT_EQ(t.Size(), 4000);
    for (int i = 0; i < 5000; ++i) {
        if ((i % 5) == 0) {
            EXPECT_FALSE(t.IsHere(i));
        } else {
            int* p = t.FindPtr(i);
            ASSERT_NE(p, nullptr);
            EXPECT_EQ(*p, i);
        }
    }
}

TEST(XHashTable, GetOccupationMatchesExpectedForSingleBucket) {
    // initialize=8 -> table has 8 buckets
    XHashTable<int, int, IntHashConstantZero> t(8);
    for (int i = 0; i < 5; ++i)
        t.Insert(i, i);

    XArray<size_t> occ;
    t.GetOccupation(occ);

    ASSERT_GE(occ.Size(), 6);
    EXPECT_EQ(occ[0], 7);   // 7 empty buckets
    EXPECT_EQ(occ[5], 1);   // 1 bucket contains 5 elements
    for (int i = 1; i < 5; ++i)
        EXPECT_EQ(occ[i], 0);
}

TEST(XHashTable, NegativeHashProducesValidIndexAndIsStableAcrossRehash) {
    // For power-of-two buckets, -1 & (size-1) yields (size-1).
    XHashTable<int, int, IntHashNegativeOne> t(4);
    for (int i = 0; i < 200; ++i)
        t.Insert(i, i * 2);

    EXPECT_EQ(t.Size(), 200);
    for (int i = 0; i < 200; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(*p, i * 2);
    }
}

TEST(XHashTable, RehashPreservesAllElements) {
    // Start small to force several rehash operations.
    XHashTable<int, int> t(4);

    const int N = 2000;
    for (int i = 0; i < N; ++i)
        t.Insert(i, i + 12345);
    ASSERT_EQ(t.Size(), N);

    for (int i = 0; i < N; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(*p, i + 12345);
    }

    // Iteration should cover exactly Size() elements.
    int count = 0;
    for (auto it = t.Begin(); it != t.End(); ++it)
        ++count;
    EXPECT_EQ(count, t.Size());
}

TEST(XHashTable, RemoveRemapsMovedEntryCorrectly) {
    XHashTable<int, int> t(8);

    for (int i = 1; i <= 200; ++i)
        t.Insert(i, i * 100);
    ASSERT_EQ(t.Size(), 200);

    // Removing a non-last element should move the last pool entry into its place.
    t.Remove(5);
    EXPECT_EQ(t.Size(), 199);
    EXPECT_FALSE(t.IsHere(5));

    // Key 200 is the last inserted; it is the one most likely to have been moved.
    int* p200 = t.FindPtr(200);
    ASSERT_NE(p200, nullptr) << "Key 200 missing after Remove(5)";
    EXPECT_EQ(*p200, 20000);

    for (int i = 1; i <= 200; ++i) {
        if (i == 5)
            continue;
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr) << "Key " << i << " missing after Remove";
        EXPECT_EQ(*p, i * 100);
    }
}

TEST(XHashTable, RemoveIteratorPatternInSingleBucketChain) {
    // Force all keys into the same bucket to stress linked-list + FastRemove remap.
    XHashTable<int, int> t(4);

    for (int i = 0; i < 200; ++i)
        t.Insert(i, i);
    ASSERT_EQ(t.Size(), 200);

    // Classic erase-while-iterating pattern.
    for (auto it = t.Begin(); it != t.End();) {
        const int k = it.GetKey();
        if ((k % 2) == 0) {
            it = t.Remove(it);
        } else {
            ++it;
        }
    }

    EXPECT_EQ(t.Size(), 100);
    for (int i = 0; i < 200; ++i) {
        if ((i % 2) == 0) {
            EXPECT_FALSE(t.IsHere(i));
        } else {
            int* p = t.FindPtr(i);
            ASSERT_NE(p, nullptr);
            EXPECT_EQ(*p, i);
        }
    }

    // Iteration still sound.
    int count = 0;
    for (auto it = t.Begin(); it != t.End(); ++it)
        ++count;
    EXPECT_EQ(count, t.Size());
}

TEST(XHashTable, ConstIterationMatchesFind) {
    XHashTable<int, int> t(4);
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i * 7);

    const XHashTable<int, int>& ct = t;
    std::set<int> seen;
    for (auto it = ct.Begin(); it != ct.End(); ++it) {
        const int k = it.GetKey();
        EXPECT_TRUE(ct.IsHere(k));
        EXPECT_EQ(*it, k * 7);
        seen.insert(k);
    }
    EXPECT_EQ((int)seen.size(), ct.Size());
}

TEST(XHashTable, CopyConstructorAndAssignmentIndependence) {
    XHashTable<int, int> t(4);
    for (int i = 0; i < 200; ++i)
        t.Insert(i, i * 11);

    XHashTable<int, int> copy(t);
    EXPECT_EQ(copy.Size(), t.Size());
    ExpectAllPresent(copy, 0, 200, 11);

    XHashTable<int, int> assigned;
    assigned = t;
    EXPECT_EQ(assigned.Size(), t.Size());
    ExpectAllPresent(assigned, 0, 200, 11);

    // Mutate original and ensure copies stay stable.
    t.Insert(0, 999);
    EXPECT_EQ(*t.FindPtr(0), 999);
    EXPECT_EQ(*copy.FindPtr(0), 0 * 11);
    EXPECT_EQ(*assigned.FindPtr(0), 0 * 11);
}

#if VX_HAS_CXX11
TEST(XHashTable, MoveConstructorAndAssignmentPreservesElements) {
    XHashTable<int, int> t(4);
    for (int i = 0; i < 300; ++i)
        t.Insert(i, i * 13);

    XHashTable<int, int> moved(std::move(t));
    EXPECT_EQ(moved.Size(), 300);
    ExpectAllPresent(moved, 0, 300, 13);

    XHashTable<int, int> t2(4);
    for (int i = 0; i < 100; ++i)
        t2.Insert(i, i * 2);
    t2 = std::move(moved);
    EXPECT_EQ(t2.Size(), 300);
    ExpectAllPresent(t2, 0, 300, 13);
}

TEST(XHashTable, Cxx11InitializerListAssignEmplaceRangeFor) {
    XHashTable<int, int> t{{{1, 10}, {2, 20}, {3, 30}}};
    EXPECT_EQ(t.Size(), 3);
    EXPECT_EQ(*t.FindPtr(1), 10);
    EXPECT_EQ(*t.FindPtr(2), 20);
    EXPECT_EQ(*t.FindPtr(3), 30);

    t = {{7, 70}, {8, 80}};
    EXPECT_EQ(t.Size(), 2);
    EXPECT_EQ(*t.FindPtr(7), 70);
    EXPECT_EQ(*t.FindPtr(8), 80);
    EXPECT_EQ(t.FindPtr(1), nullptr);

    t.Emplace(9, 90);
    EXPECT_EQ(*t.FindPtr(9), 90);

    int sum = 0;
    for (int v : t) {
        sum += v;
    }
    EXPECT_EQ(sum, 70 + 80 + 90);
}
#endif

TEST(XHashTable, CustomKeyHashEqBasicOperations) {
    XHashTable<int, Key, KeyHashIdentity, KeyEq> t(4);

    for (int i = 0; i < 100; ++i)
        t.Insert(Key{i}, i * 3);
    ASSERT_EQ(t.Size(), 100);

    for (int i = 0; i < 100; ++i) {
        auto it = t.Find(Key{i});
        ASSERT_NE(it, t.End());
        EXPECT_EQ(*it, i * 3);
    }

    // operator[] default insertion for absent key
    EXPECT_FALSE(t.IsHere(Key{1000}));
    int& v = t[Key{1000}];
    EXPECT_TRUE(t.IsHere(Key{1000}));
    EXPECT_EQ(v, 0);
    v = 42;
    EXPECT_EQ(*t.Find(Key{1000}), 42);
}

// ==============================================================================
// XNHashTable Tests - Chained Hash Table with Dynamic Node Allocation
// ==============================================================================

TEST(XNHashTable, EmptyTableBeginEqualsEnd) {
    XNHashTable<int, int> t;
    EXPECT_EQ(t.Size(), 0);
    EXPECT_EQ(t.Begin(), t.End());

    const XNHashTable<int, int>& ct = t;
    EXPECT_EQ(ct.Begin(), ct.End());
}

TEST(XNHashTable, BasicInsertAndFind) {
    XNHashTable<int, int> t(8);
    
    EXPECT_TRUE(t.Insert(1, 100, TRUE));
    EXPECT_TRUE(t.Insert(2, 200, TRUE));
    EXPECT_TRUE(t.Insert(3, 300, TRUE));
    EXPECT_EQ(t.Size(), 3);
    
    EXPECT_EQ(*t.FindPtr(1), 100);
    EXPECT_EQ(*t.FindPtr(2), 200);
    EXPECT_EQ(*t.FindPtr(3), 300);
}

TEST(XNHashTable, InsertWithOverrideFlag) {
    XNHashTable<int, int> t(8);
    
    EXPECT_TRUE(t.Insert(5, 50, TRUE));
    EXPECT_EQ(*t.FindPtr(5), 50);
    
    // override = FALSE should not update
    EXPECT_FALSE(t.Insert(5, 500, FALSE));
    EXPECT_EQ(*t.FindPtr(5), 50);
    
    // override = TRUE should update
    EXPECT_TRUE(t.Insert(5, 5000, TRUE));
    EXPECT_EQ(*t.FindPtr(5), 5000);
}

TEST(XNHashTable, InsertOverwritesByDefault) {
    XNHashTable<int, int> t(8);
    
    auto it1 = t.Insert(7, 70);
    EXPECT_NE(it1, t.End());
    EXPECT_EQ(*it1, 70);
    
    auto it2 = t.Insert(7, 700);
    EXPECT_NE(it2, t.End());
    EXPECT_EQ(*it2, 700);
    EXPECT_EQ(t.Size(), 1);
}

TEST(XNHashTable, InsertUniqueDoesNotOverwrite) {
    XNHashTable<int, int> t(8);
    
    auto it1 = t.InsertUnique(9, 90);
    EXPECT_EQ(*it1, 90);
    
    auto it2 = t.InsertUnique(9, 900);
    EXPECT_EQ(*it2, 90);
    EXPECT_EQ(t.Size(), 1);
}

TEST(XNHashTable, TestInsertReturnsCorrectNewFlag) {
    XNHashTable<int, int> t(8);
    
    auto p1 = t.TestInsert(5, 50);
    EXPECT_TRUE(p1.m_New);
    EXPECT_EQ(*p1.m_Iterator, 50);
    EXPECT_EQ(t.Size(), 1);
    
    auto p2 = t.TestInsert(5, 500);
    EXPECT_FALSE(p2.m_New);
    EXPECT_EQ(*p2.m_Iterator, 50);
    EXPECT_EQ(t.Size(), 1);
}

TEST(XNHashTable, OperatorBracketDefaultInsertion) {
    XNHashTable<int, int> t(8);
    
    EXPECT_FALSE(t.IsHere(123));
    int& v = t[123];
    EXPECT_TRUE(t.IsHere(123));
    EXPECT_EQ(v, 0);
    
    v = 77;
    EXPECT_EQ(*t.FindPtr(123), 77);
}

TEST(XNHashTable, RemoveKey) {
    XNHashTable<int, int> t(8);
    t.Insert(1, 10, TRUE);
    t.Insert(2, 20, TRUE);
    t.Insert(3, 30, TRUE);
    EXPECT_EQ(t.Size(), 3);
    
    t.Remove(2);
    EXPECT_EQ(t.Size(), 2);
    EXPECT_FALSE(t.IsHere(2));
    EXPECT_TRUE(t.IsHere(1));
    EXPECT_TRUE(t.IsHere(3));
}

TEST(XNHashTable, RemoveAbsentKeyNoChange) {
    XNHashTable<int, int> t(8);
    t.Insert(1, 10, TRUE);
    t.Insert(2, 20, TRUE);
    EXPECT_EQ(t.Size(), 2);
    
    t.Remove(999);
    EXPECT_EQ(t.Size(), 2);
    EXPECT_EQ(*t.FindPtr(1), 10);
    EXPECT_EQ(*t.FindPtr(2), 20);
}

TEST(XNHashTable, RemoveViaIterator) {
    XNHashTable<int, int> t(8);
    for (int i = 0; i < 10; ++i)
        t.Insert(i, i * 10, TRUE);
    ASSERT_EQ(t.Size(), 10);
    
    for (auto it = t.Begin(); it != t.End();) {
        const int k = it.GetKey();
        if ((k % 2) == 0) {
            it = t.Remove(it);
        } else {
            ++it;
        }
    }
    
    EXPECT_EQ(t.Size(), 5);
    for (int i = 0; i < 10; ++i) {
        if ((i % 2) == 0) {
            EXPECT_FALSE(t.IsHere(i));
        } else {
            EXPECT_TRUE(t.IsHere(i));
            EXPECT_EQ(*t.FindPtr(i), i * 10);
        }
    }
}

TEST(XNHashTable, ClearRemovesAllElements) {
    XNHashTable<int, int> t(8);
    for (int i = 0; i < 50; ++i)
        t.Insert(i, i * 2, TRUE);
    ASSERT_EQ(t.Size(), 50);
    
    t.Clear();
    EXPECT_EQ(t.Size(), 0);
    EXPECT_EQ(t.Begin(), t.End());
    EXPECT_FALSE(t.IsHere(0));
    EXPECT_FALSE(t.IsHere(25));
}

TEST(XNHashTable, ClearThenReuse) {
    XNHashTable<int, int> t(8);
    for (int i = 0; i < 30; ++i)
        t.Insert(i, i * 2, TRUE);
    ASSERT_EQ(t.Size(), 30);
    
    t.Clear();
    EXPECT_EQ(t.Size(), 0);
    
    for (int i = 0; i < 30; ++i)
        t.Insert(i, i * 3, TRUE);
    EXPECT_EQ(t.Size(), 30);
    
    for (int i = 0; i < 30; ++i) {
        EXPECT_EQ(*t.FindPtr(i), i * 3);
    }
}

TEST(XNHashTable, RehashTriggeredAutomatically) {
    XNHashTable<int, int> t(4);
    
    const int N = 1000;
    for (int i = 0; i < N; ++i) {
        t.Insert(i, i * 17, TRUE);
    }
    
    EXPECT_EQ(t.Size(), N);
    for (int i = 0; i < N; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(*p, i * 17);
    }
}

TEST(XNHashTable, RehashWithCollisions) {
    // All keys hash to same bucket (0)
    XNHashTable<int, int, IntHashConstantZero> t(4);
    
    const int N = 500;
    for (int i = 0; i < N; ++i)
        t.Insert(i, i * 9, TRUE);
    
    EXPECT_EQ(t.Size(), N);
    for (int i = 0; i < N; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr) << "Missing key " << i;
        EXPECT_EQ(*p, i * 9);
    }
    
    // Check iteration completes
    auto keys = CollectKeys(t);
    EXPECT_EQ((int)keys.size(), t.Size());
}

TEST(XNHashTable, IterationCoversAllElements) {
    XNHashTable<int, int> t(8);
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i * 7, TRUE);
    
    std::set<int> seenKeys;
    int count = 0;
    for (auto it = t.Begin(); it != t.End(); ++it) {
        seenKeys.insert(it.GetKey());
        ++count;
    }
    
    EXPECT_EQ(count, t.Size());
    EXPECT_EQ((int)seenKeys.size(), t.Size());
}

TEST(XNHashTable, ConstIterationMatchesFind) {
    XNHashTable<int, int> t(8);
    for (int i = 0; i < 50; ++i)
        t.Insert(i, i * 7, TRUE);
    
    const XNHashTable<int, int>& ct = t;
    std::set<int> seen;
    for (auto it = ct.Begin(); it != ct.End(); ++it) {
        const int k = it.GetKey();
        EXPECT_TRUE(ct.IsHere(k));
        EXPECT_EQ(*it, k * 7);
        seen.insert(k);
    }
    EXPECT_EQ((int)seen.size(), ct.Size());
}

TEST(XNHashTable, CopyConstructorCreatesIndependentCopy) {
    XNHashTable<int, int> t(8);
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i * 11, TRUE);
    
    XNHashTable<int, int> copy(t);
    EXPECT_EQ(copy.Size(), t.Size());
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(*copy.FindPtr(i), i * 11);
    }
    
    // Mutate original, copy should be unaffected
    t.Insert(0, 999, TRUE);
    EXPECT_EQ(*t.FindPtr(0), 999);
    EXPECT_EQ(*copy.FindPtr(0), 0 * 11);
}

TEST(XNHashTable, AssignmentOperatorCreatesIndependentCopy) {
    XNHashTable<int, int> t(8);
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i * 11, TRUE);
    
    XNHashTable<int, int> assigned;
    assigned = t;
    EXPECT_EQ(assigned.Size(), t.Size());
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(*assigned.FindPtr(i), i * 11);
    }
    
    // Mutate original, assigned should be unaffected
    t.Insert(50, 999, TRUE);
    EXPECT_EQ(*t.FindPtr(50), 999);
    EXPECT_EQ(*assigned.FindPtr(50), 50 * 11);
}

TEST(XNHashTable, SelfAssignmentIsHandled) {
    XNHashTable<int, int> t(8);
    for (int i = 0; i < 50; ++i)
        t.Insert(i, i * 3, TRUE);
    
    t = t;
    EXPECT_EQ(t.Size(), 50);
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(*t.FindPtr(i), i * 3);
    }
}

TEST(XNHashTable, LookUpFunctionality) {
    XNHashTable<int, int> t(8);
    t.Insert(10, 100, TRUE);
    t.Insert(20, 200, TRUE);
    
    int value;
    EXPECT_TRUE(t.LookUp(10, value));
    EXPECT_EQ(value, 100);
    
    EXPECT_TRUE(t.LookUp(20, value));
    EXPECT_EQ(value, 200);
    
    EXPECT_FALSE(t.LookUp(999, value));
}

TEST(XNHashTable, CustomKeyTypeWithCustomHashAndEq) {
    XNHashTable<int, Key, KeyHashIdentity, KeyEq> t(8);
    
    for (int i = 0; i < 50; ++i)
        t.Insert(Key{i}, i * 3, TRUE);
    ASSERT_EQ(t.Size(), 50);
    
    for (int i = 0; i < 50; ++i) {
        auto it = t.Find(Key{i});
        ASSERT_NE(it, t.End());
        EXPECT_EQ(*it, i * 3);
    }
    
    EXPECT_FALSE(t.IsHere(Key{1000}));
    int& v = t[Key{1000}];
    EXPECT_TRUE(t.IsHere(Key{1000}));
    EXPECT_EQ(v, 0);
}

TEST(XNHashTable, NegativeHashValuesHandledCorrectly) {
    XNHashTable<int, int, IntHashNegativeOne> t(8);
    
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i * 2, TRUE);
    
    EXPECT_EQ(t.Size(), 100);
    for (int i = 0; i < 100; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(*p, i * 2);
    }
}

TEST(XNHashTable, GetMemoryOccupation) {
    XNHashTable<int, int> t(8);
    
    size_t memBefore = t.GetMemoryOccupation(FALSE);
    EXPECT_GT(memBefore, 0);
    
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i, TRUE);
    
    size_t memAfter = t.GetMemoryOccupation(FALSE);
    EXPECT_GT(memAfter, memBefore);
    
    size_t memWithStatic = t.GetMemoryOccupation(TRUE);
    EXPECT_GT(memWithStatic, memAfter);
}

#if VX_HAS_CXX11
TEST(XNHashTable, Cxx11MoveCtorAssignInitializerListEmplaceRangeFor) {
    XNHashTable<int, int> t{{{1, 10}, {2, 20}}};
    EXPECT_EQ(t.Size(), 2);
    EXPECT_EQ(*t.FindPtr(1), 10);
    EXPECT_EQ(*t.FindPtr(2), 20);

    XNHashTable<int, int> moved(std::move(t));
    EXPECT_EQ(moved.Size(), 2);
    EXPECT_EQ(*moved.FindPtr(1), 10);
    EXPECT_EQ(*moved.FindPtr(2), 20);

    XNHashTable<int, int> assigned;
    assigned = std::move(moved);
    EXPECT_EQ(assigned.Size(), 2);

    assigned = {{7, 70}, {8, 80}};
    EXPECT_EQ(assigned.Size(), 2);
    EXPECT_EQ(*assigned.FindPtr(7), 70);

    assigned.Emplace(9, 90);
    EXPECT_EQ(*assigned.FindPtr(9), 90);

    int sum = 0;
    for (int v : assigned) {
        sum += v;
    }
    EXPECT_EQ(sum, 70 + 80 + 90);
}
#endif

// ==============================================================================
// XSHashTable Tests - Static Hash Table with Open Addressing
// ==============================================================================

TEST(XSHashTable, EmptyTableBeginEqualsEnd) {
    XSHashTable<int, int> t;
    EXPECT_EQ(t.Size(), 0);
    EXPECT_EQ(t.Begin(), t.End());
    
    const XSHashTable<int, int>& ct = t;
    EXPECT_EQ(ct.Begin(), ct.End());
}

TEST(XSHashTable, BasicInsertAndFind) {
    XSHashTable<int, int> t(8);
    
    EXPECT_TRUE(t.Insert(1, 100, TRUE));
    EXPECT_TRUE(t.Insert(2, 200, TRUE));
    EXPECT_TRUE(t.Insert(3, 300, TRUE));
    EXPECT_EQ(t.Size(), 3);
    
    EXPECT_EQ(*t.FindPtr(1), 100);
    EXPECT_EQ(*t.FindPtr(2), 200);
    EXPECT_EQ(*t.FindPtr(3), 300);
}

TEST(XSHashTable, InsertWithOverrideFlag) {
    XSHashTable<int, int> t(8);
    
    EXPECT_TRUE(t.Insert(5, 50, TRUE));
    EXPECT_EQ(*t.FindPtr(5), 50);
    
    // override = FALSE should not update
    EXPECT_FALSE(t.Insert(5, 500, FALSE));
    EXPECT_EQ(*t.FindPtr(5), 50);
    
    // override = TRUE should update
    EXPECT_TRUE(t.Insert(5, 5000, TRUE));
    EXPECT_EQ(*t.FindPtr(5), 5000);
}

TEST(XSHashTable, InsertUniqueDoesNotOverwrite) {
    XSHashTable<int, int> t(8);
    
    auto it1 = t.InsertUnique(9, 90);
    EXPECT_EQ(*it1, 90);
    
    auto it2 = t.InsertUnique(9, 900);
    EXPECT_EQ(*it2, 90);
    EXPECT_EQ(t.Size(), 1);
}

TEST(XSHashTable, OperatorBracketCreatesEntry) {
    XSHashTable<int, int> t(8);
    
    EXPECT_FALSE(t.IsHere(123));
    int& v = t[123];
    EXPECT_TRUE(t.IsHere(123));
    // Note: XSHashTable does not initialize the value to 0, just creates the entry
    
    v = 77;
    EXPECT_EQ(*t.FindPtr(123), 77);
}

TEST(XSHashTable, RemoveMarksAsDeleted) {
    XSHashTable<int, int> t(8);
    t.Insert(1, 10, TRUE);
    t.Insert(2, 20, TRUE);
    t.Insert(3, 30, TRUE);
    EXPECT_EQ(t.Size(), 3);
    
    t.Remove(2);
    EXPECT_EQ(t.Size(), 2);
    EXPECT_FALSE(t.IsHere(2));
    EXPECT_TRUE(t.IsHere(1));
    EXPECT_TRUE(t.IsHere(3));
}

TEST(XSHashTable, RemoveAbsentKeyNoChange) {
    XSHashTable<int, int> t(8);
    t.Insert(1, 10, TRUE);
    t.Insert(2, 20, TRUE);
    EXPECT_EQ(t.Size(), 2);
    
    t.Remove(999);
    EXPECT_EQ(t.Size(), 2);
    EXPECT_EQ(*t.FindPtr(1), 10);
    EXPECT_EQ(*t.FindPtr(2), 20);
}

TEST(XSHashTable, RemoveViaIterator) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 10; ++i)
        t.Insert(i, i * 10, TRUE);
    ASSERT_EQ(t.Size(), 10);
    
    for (auto it = t.Begin(); it != t.End();) {
        const int k = it.GetKey();
        if ((k % 2) == 0) {
            it = t.Remove(it);
        } else {
            ++it;
        }
    }
    
    EXPECT_EQ(t.Size(), 5);
    for (int i = 0; i < 10; ++i) {
        if ((i % 2) == 0) {
            EXPECT_FALSE(t.IsHere(i));
        } else {
            EXPECT_TRUE(t.IsHere(i));
            EXPECT_EQ(*t.FindPtr(i), i * 10);
        }
    }
}

TEST(XSHashTable, ClearRemovesAllElements) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 50; ++i)
        t.Insert(i, i * 2, TRUE);
    ASSERT_EQ(t.Size(), 50);
    
    t.Clear();
    EXPECT_EQ(t.Size(), 0);
    EXPECT_EQ(t.Begin(), t.End());
    EXPECT_FALSE(t.IsHere(0));
}

TEST(XSHashTable, ClearThenReuse) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 30; ++i)
        t.Insert(i, i * 2, TRUE);
    ASSERT_EQ(t.Size(), 30);
    
    t.Clear();
    EXPECT_EQ(t.Size(), 0);
    
    for (int i = 0; i < 30; ++i)
        t.Insert(i, i * 3, TRUE);
    EXPECT_EQ(t.Size(), 30);
    
    for (int i = 0; i < 30; ++i) {
        EXPECT_EQ(*t.FindPtr(i), i * 3);
    }
}

TEST(XSHashTable, AutomaticRehashOnLoadFactor) {
    XSHashTable<int, int> t(4);
    
    const int N = 1000;
    for (int i = 0; i < N; ++i) {
        t.Insert(i, i * 17, TRUE);
    }
    
    EXPECT_EQ(t.Size(), N);
    for (int i = 0; i < N; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(*p, i * 17);
    }
}

TEST(XSHashTable, LinearProbingWithCollisions) {
    // All keys hash to same initial position
    XSHashTable<int, int, IntHashConstantZero> t(16);
    
    const int N = 10;
    for (int i = 0; i < N; ++i)
        t.Insert(i, i * 9, TRUE);
    
    EXPECT_EQ(t.Size(), N);
    for (int i = 0; i < N; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(*p, i * 9);
    }
}

TEST(XSHashTable, IterationCoversAllElements) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i * 7, TRUE);
    
    std::set<int> seenKeys;
    int count = 0;
    for (auto it = t.Begin(); it != t.End(); ++it) {
        seenKeys.insert(it.GetKey());
        ++count;
    }
    
    EXPECT_EQ(count, t.Size());
    EXPECT_EQ((int)seenKeys.size(), t.Size());
}

TEST(XSHashTable, ConstIterationMatchesFind) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 50; ++i)
        t.Insert(i, i * 7, TRUE);
    
    const XSHashTable<int, int>& ct = t;
    std::set<int> seen;
    for (auto it = ct.Begin(); it != ct.End(); ++it) {
        const int k = it.GetKey();
        EXPECT_TRUE(ct.IsHere(k));
        EXPECT_EQ(*it, k * 7);
        seen.insert(k);
    }
    EXPECT_EQ((int)seen.size(), ct.Size());
}

TEST(XSHashTable, CopyConstructorCreatesIndependentCopy) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i * 11, TRUE);
    
    XSHashTable<int, int> copy(t);
    EXPECT_EQ(copy.Size(), t.Size());
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(*copy.FindPtr(i), i * 11);
    }
    
    // Mutate original, copy should be unaffected
    t.Insert(0, 999, TRUE);
    EXPECT_EQ(*t.FindPtr(0), 999);
    EXPECT_EQ(*copy.FindPtr(0), 0 * 11);
}

TEST(XSHashTable, AssignmentOperatorCreatesIndependentCopy) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i * 11, TRUE);
    
    XSHashTable<int, int> assigned;
    assigned = t;
    EXPECT_EQ(assigned.Size(), t.Size());
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(*assigned.FindPtr(i), i * 11);
    }
    
    // Mutate original, assigned should be unaffected
    t.Insert(50, 999, TRUE);
    EXPECT_EQ(*t.FindPtr(50), 999);
    EXPECT_EQ(*assigned.FindPtr(50), 50 * 11);
}

TEST(XSHashTable, SelfAssignmentIsHandled) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 50; ++i)
        t.Insert(i, i * 3, TRUE);
    
    t = t;
    EXPECT_EQ(t.Size(), 50);
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(*t.FindPtr(i), i * 3);
    }
}

#if VX_HAS_CXX11
TEST(XSHashTable, MoveConstructorPreservesElements) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 200; ++i)
        t.Insert(i, i * 13, TRUE);
    
    XSHashTable<int, int> moved(std::move(t));
    EXPECT_EQ(moved.Size(), 200);
    
    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(*moved.FindPtr(i), i * 13);
    }
}

TEST(XSHashTable, MoveAssignmentPreservesElements) {
    XSHashTable<int, int> t(16);
    for (int i = 0; i < 200; ++i)
        t.Insert(i, i * 13, TRUE);
    
    XSHashTable<int, int> t2;
    t2 = std::move(t);
    EXPECT_EQ(t2.Size(), 200);
    
    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(*t2.FindPtr(i), i * 13);
    }
}

TEST(XSHashTable, Cxx11InitializerListAssignTestInsertEmplaceRangeFor) {
    XSHashTable<int, int> t{{{1, 10}, {2, 20}, {3, 30}}};
    EXPECT_EQ(t.Size(), 3);
    EXPECT_EQ(*t.FindPtr(1), 10);
    EXPECT_EQ(*t.FindPtr(2), 20);
    EXPECT_EQ(*t.FindPtr(3), 30);

    auto p1 = t.TestInsert(3, 300);
    EXPECT_FALSE(p1.m_New);
    EXPECT_EQ(*p1.m_Iterator, 30);

    auto p2 = t.TestInsert(4, 40);
    EXPECT_TRUE(p2.m_New);
    EXPECT_EQ(*p2.m_Iterator, 40);

    t = {{7, 70}, {8, 80}};
    EXPECT_EQ(t.Size(), 2);
    EXPECT_EQ(*t.FindPtr(7), 70);
    EXPECT_EQ(*t.FindPtr(8), 80);

    t.EmplaceUnique(9, 90);
    EXPECT_EQ(*t.FindPtr(9), 90);

    int sum = 0;
    for (int v : t) {
        sum += v;
    }
    EXPECT_EQ(sum, 70 + 80 + 90);
}
#endif

TEST(XSHashTable, LookUpFunctionality) {
    XSHashTable<int, int> t(16);
    t.Insert(10, 100, TRUE);
    t.Insert(20, 200, TRUE);
    
    int value;
    EXPECT_TRUE(t.LookUp(10, value));
    EXPECT_EQ(value, 100);
    
    EXPECT_TRUE(t.LookUp(20, value));
    EXPECT_EQ(value, 200);
    
    EXPECT_FALSE(t.LookUp(999, value));
}

TEST(XSHashTable, CustomKeyTypeWithCustomHashAndEq) {
    XSHashTable<int, Key, KeyHashIdentity, KeyEq> t(16);
    
    for (int i = 0; i < 50; ++i)
        t.Insert(Key{i}, i * 3, TRUE);
    ASSERT_EQ(t.Size(), 50);
    
    for (int i = 0; i < 50; ++i) {
        auto it = t.Find(Key{i});
        ASSERT_NE(it, t.End());
        EXPECT_EQ(*it, i * 3);
    }
    
    EXPECT_FALSE(t.IsHere(Key{1000}));
    int& v = t[Key{1000}];
    EXPECT_TRUE(t.IsHere(Key{1000}));
    // Note: XSHashTable does not initialize value to 0
    v = 42;
    EXPECT_EQ(*t.FindPtr(Key{1000}), 42);
}

TEST(XSHashTable, NegativeHashValuesHandledCorrectly) {
    XSHashTable<int, int, IntHashNegativeOne> t(16);
    
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i * 2, TRUE);
    
    EXPECT_EQ(t.Size(), 100);
    for (int i = 0; i < 100; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(*p, i * 2);
    }
}

TEST(XSHashTable, DeletedEntriesReuseSlot) {
    XSHashTable<int, int> t(16);
    
    // Insert an element
    t.Insert(1, 10, TRUE);
    EXPECT_TRUE(t.IsHere(1));
    EXPECT_EQ(t.Size(), 1);
    
    // Remove it
    t.Remove(1);
    EXPECT_FALSE(t.IsHere(1));
    EXPECT_EQ(t.Size(), 0);
    
    // Re-insert should reuse the deleted slot
    t.Insert(1, 100, TRUE);
    EXPECT_TRUE(t.IsHere(1));
    EXPECT_EQ(*t.FindPtr(1), 100);
    EXPECT_EQ(t.Size(), 1);
}

TEST(XSHashTable, RehashAfterManyDeletions) {
    XSHashTable<int, int> t(16);
    
    for (int i = 0; i < 200; ++i)
        t.Insert(i, i, TRUE);
    
    // Delete every other element
    for (int i = 0; i < 200; i += 2)
        t.Remove(i);
    
    EXPECT_EQ(t.Size(), 100);
    
    // Add more elements to trigger rehash
    for (int i = 200; i < 400; ++i)
        t.Insert(i, i, TRUE);
    
    EXPECT_EQ(t.Size(), 300);
    
    // Verify all remaining elements
    for (int i = 1; i < 200; i += 2) {
        EXPECT_TRUE(t.IsHere(i));
        EXPECT_EQ(*t.FindPtr(i), i);
    }
    for (int i = 200; i < 400; ++i) {
        EXPECT_TRUE(t.IsHere(i));
        EXPECT_EQ(*t.FindPtr(i), i);
    }
}

TEST(XSHashTable, StressTestLargeNumberOfElements) {
    XSHashTable<int, int> t(4);
    
    const int N = 5000;
    for (int i = 0; i < N; ++i) {
        t.Insert(i, i * 7, TRUE);
    }
    
    EXPECT_EQ(t.Size(), N);
    
    // Verify all elements
    for (int i = 0; i < N; ++i) {
        int* p = t.FindPtr(i);
        ASSERT_NE(p, nullptr) << "Missing key " << i;
        EXPECT_EQ(*p, i * 7);
    }
    
    // Verify iteration
    auto keys = CollectKeys(t);
    EXPECT_EQ((int)keys.size(), t.Size());
}

TEST(XSHashTable, FindPtrReturnsNullForMissingKey) {
    XSHashTable<int, int> t(16);
    t.Insert(1, 10, TRUE);
    t.Insert(2, 20, TRUE);
    
    int* p = t.FindPtr(999);
    EXPECT_EQ(p, nullptr);
    
    const XSHashTable<int, int>& ct = t;
    const int* cp = ct.FindPtr(999);
    EXPECT_EQ(cp, nullptr);
}

TEST(XSHashTable, CustomLoadFactor) {
    // Test with a very low load factor (0.5) to trigger frequent rehashing
    XSHashTable<int, int> t(8, 0.5f);
    
    for (int i = 0; i < 100; ++i)
        t.Insert(i, i, TRUE);
    
    EXPECT_EQ(t.Size(), 100);
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(t.IsHere(i));
    }
}

TEST(XSHashTable, InvalidLoadFactorDefaultsToPointSevenFive) {
    XSHashTable<int, int> t1(8, 0.0f);    // Invalid
    XSHashTable<int, int> t2(8, -0.5f);   // Invalid
    
    // Both should still work correctly
    for (int i = 0; i < 50; ++i) {
        t1.Insert(i, i, TRUE);
        t2.Insert(i, i, TRUE);
    }
    
    EXPECT_EQ(t1.Size(), 50);
    EXPECT_EQ(t2.Size(), 50);
}

} // namespace
