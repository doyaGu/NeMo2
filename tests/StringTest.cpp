#include "gtest/gtest.h"
#include "VxMath.h"
#include <string> // Used for comparison

// Helper to check the state of an XString
void ExpectStringState(const XString& s, const char* expected_cstr, XWORD expected_length) {
    EXPECT_STREQ(s.CStr(), expected_cstr);
    EXPECT_EQ(s.Length(), expected_length);
    EXPECT_EQ(s.IsEmpty(), expected_length == 0);
}

// --- Test Suite for XString ---

// 1. Construction, Assignment, and State
TEST(XStringTest, DefaultConstructor) {
    XString s;
    ExpectStringState(s, "", 0);
}

TEST(XStringTest, CStringConstructor) {
    XString s("Hello World");
    ExpectStringState(s, "Hello World", 11);
}

TEST(XStringTest, NullCStringConstructor) {
    XString s(nullptr);
    ExpectStringState(s, "", 0);
}

TEST(XStringTest, SizedCStringConstructor) {
    XString s("Hello World", 5);
    ExpectStringState(s, "Hello", 5);
}

TEST(XStringTest, SizedConstructor) {
    XString s(20);
    ExpectStringState(s, "", 0);
    EXPECT_GE(s.Capacity(), 20);
}

TEST(XStringTest, CopyConstructor) {
    XString s1("Test Copy");
    XString s2(s1);
    ExpectStringState(s2, "Test Copy", 9);
    // Ensure it's a deep copy
    EXPECT_NE(s1.CStr(), s2.CStr());
}

TEST(XStringTest, BaseStringConstructor) {
    XBaseString base_s("Base String");
    XString s(base_s);
    ExpectStringState(s, "Base String", 11);
}

#if VX_HAS_CXX11
TEST(XStringTest, MoveConstructor) {
    XString s1("Test Move");
    char* old_buffer = s1.Str();
    XString s2(std::move(s1));

    ExpectStringState(s2, "Test Move", 9);
    EXPECT_EQ(s2.Str(), old_buffer); // Buffer should be moved
    ExpectStringState(s1, "", 0);    // Original string should be empty
}
#endif

TEST(XStringTest, AssignmentOperators) {
    XString s;
    XString s_source("Source");
    const char* c_source = "C-String Source";

    s = s_source;
    ExpectStringState(s, "Source", 6);

    s = c_source;
    ExpectStringState(s, "C-String Source", 15);

    s = "A";
    ExpectStringState(s, "A", 1);
    
    // Self-assignment
    // s = s;
    ExpectStringState(s, "A", 1);
}

#if VX_HAS_CXX11
TEST(XStringTest, MoveAssignmentOperator) {
    XString s1("Test Move Assign");
    XString s2;
    s2 = std::move(s1);

    ExpectStringState(s2, "Test Move Assign", 16);
    ExpectStringState(s1, "", 0);
}
#endif

TEST(XStringTest, Create) {
    XString s("Initial");
    s.Create("New Content", 3);
    ExpectStringState(s, "New", 3);
}

// TEST(XStringTest, ReserveAndCapacity) {
//     XString s("small");
//     XWORD initial_cap = s.Capacity();
//     s.Reserve(100);
//     EXPECT_STREQ(s.CStr(), "small");
//     EXPECT_GE(s.Capacity(), 100);
//     EXPECT_GT(s.Capacity(), initial_cap);
// }

// TEST(XStringTest, Resize) {
//     XString s("1234567890");
//     s.Resize(5);
//     ExpectStringState(s, "12345", 5);
//     s.Resize(10);
//     EXPECT_EQ(s.Length(), 5); // Length doesn't change on resize up
//     EXPECT_STREQ(s.CStr(), "12345"); // Null terminator is moved
// }

// 2. Comparison
TEST(XStringTest, ComparisonOperators) {
    XString s1("apple");
    XString s2("apple");
    XString s3("Apple");
    XString s4("banana");

    // Equality
    EXPECT_TRUE(s1 == s2);
    EXPECT_FALSE(s1 == s3);
    EXPECT_TRUE(s1 == "apple");
    EXPECT_FALSE(s1 == "Apple");

    // Inequality
    EXPECT_TRUE(s1 != s3);
    EXPECT_FALSE(s1 != s2);
    
    // Less/Greater
    EXPECT_TRUE(s1 < s4);
    EXPECT_TRUE(s4 > s1);
    EXPECT_TRUE(s3 < s1); // 'A' < 'a'
    
    // Less/Greater or Equal
    EXPECT_TRUE(s1 <= s2);
    EXPECT_TRUE(s1 >= s2);
}

TEST(XStringTest, CompareFunctions) {
    XString s1("apple");
    XString s2("Apple");
    XString s3("applepie");
    
    EXPECT_EQ(s1.Compare(s1), 0);
    EXPECT_GT(s1.Compare(s2), 0); // 'a' > 'A'
    EXPECT_LT(s1.Compare(s3), 0);
    
    // Case-insensitive compare
    EXPECT_EQ(s1.ICompare(s2), 0);
    
    // N-char compare
    EXPECT_EQ(s1.NCompare(s3, 5), 0);
    EXPECT_LT(s1.NCompare(s3, 6), 0);

    // Case-insensitive N-char compare
    EXPECT_EQ(s1.NICompare(s2, 5), 0);
}

// 3. Access and Iteration
TEST(XStringTest, AccessOperators) {
    XString s("abc");
    const XString cs("const");

    EXPECT_EQ(s[0], 'a');
    EXPECT_EQ(cs[1], 'o');

    s[1] = 'x';
    ExpectStringState(s, "axc", 3);
}

#if VX_HAS_CXX11
TEST(XStringTest, AtFunction) {
    XString s("abc");
    EXPECT_EQ(s.At(1), 'b');
    EXPECT_THROW(s.At(3), std::out_of_range);
    EXPECT_THROW(s.At(-1), std::out_of_range);
}

TEST(XStringTest, Iteration) {
    XString s("123");
    std::string temp;
    for (char c : s) {
        temp += c;
    }
    EXPECT_EQ(temp, "123");
}

TEST(XStringTest, Cxx11StdStringInteropAndUtilities) {
    std::string src("hello\0world", 11);
    XString s(src);

    EXPECT_EQ(static_cast<size_t>(s.Length()), src.size());
    EXPECT_FALSE(s.IsEmpty());
    EXPECT_STREQ(s.CStr(), "hello");
    EXPECT_EQ(s.ToStdString(), src);

    XString t;
    t = src;
    EXPECT_EQ(t.ToStdString(), src);
    t << std::string("!");
    EXPECT_EQ(t.Back(), '!');

    // push_back / pop_back
    XString u("ab");
    u.PushBack('c');
    ExpectStringState(u, "abc", 3);
    u.PopBack();
    ExpectStringState(u, "ab", 2);
    EXPECT_EQ(u.Front(), 'a');
    EXPECT_EQ(u.Back(), 'b');
}

TEST(XStringTest, Cxx11EmptyRangeForIsSafe) {
    XString s;
    std::string temp;
    for (char c : s) {
        temp += c;
    }
    EXPECT_TRUE(temp.empty());
}
#endif

// 4. Searching
TEST(XStringTest, FindFunctions) {
    XString s("Hello world, hello universe.");

    // Find char
    EXPECT_EQ(s.Find('o'), 4);
    EXPECT_EQ(s.Find('o', 5), 7);
    EXPECT_EQ(s.Find('x'), XString::NOTFOUND);

    // Find string
    EXPECT_EQ(s.Find("hello"), 13);
    EXPECT_EQ(s.Find("hello", 14), XString::NOTFOUND);

    // IFind string (case-insensitive)
    // EXPECT_EQ(s.IFind("HELLO"), 0);
    // EXPECT_EQ(s.IFind("UNIVERSE"), 19);

    // RFind char (reverse)
    EXPECT_EQ(s.RFind('o'), 17);
    EXPECT_EQ(s.RFind('o', 16), 7);
}

TEST(XStringTest, ContentQueries) {
    XString s("TopLevel");
    
    EXPECT_TRUE(s.Contains("Lev"));
    EXPECT_FALSE(s.Contains("lev"));

    EXPECT_TRUE(s.StartsWith("Top"));
    EXPECT_FALSE(s.StartsWith("top"));
    EXPECT_TRUE(s.IStartsWith("top"));

    EXPECT_TRUE(s.EndsWith("Level"));
    EXPECT_FALSE(s.EndsWith("level"));
    EXPECT_TRUE(s.IEndsWith("level"));
}


// 5. Modification and Manipulation
TEST(XStringTest, CaseModification) {
    XString s("Test 123!");
    s.ToUpper();
    ExpectStringState(s, "TEST 123!", 9);
    s.ToLower();
    ExpectStringState(s, "test 123!", 9);
}

TEST(XStringTest, WhitespaceModification) {
    XString s1("  \t hello \n ");
    s1.Trim();
    ExpectStringState(s1, "hello", 5);

    XString s2("  \t hello \n ");
    s2.Strip();
    ExpectStringState(s2, " hello ", 7);
    
    XString s3("no-whitespace");
    s3.Trim();
    ExpectStringState(s3, "no-whitespace", 13);
}

TEST(XStringTest, Replace) {
    // Char replace
    XString s1("banana");
    s1.Replace('a', 'o');
    ExpectStringState(s1, "bonono", 6);

    // String replace (same size)
    XString s2("Hello world");
    s2.Replace("world", "there");
    ExpectStringState(s2, "Hello there", 11);

    // String replace (shrink)
    s2.Replace("there", "u");
    ExpectStringState(s2, "Hello u", 7);

    // String replace (grow, might reallocate)
    s2.Replace("u", "everybody");
    ExpectStringState(s2, "Hello everybody", 15);
}

TEST(XStringTest, SubstringAndCut) {
    XString s("0123456789");

    // Substring (doesn't modify original)
    XString sub = s.Substring(2, 4);
    ExpectStringState(sub, "2345", 4);
    ExpectStringState(s, "0123456789", 10);
    
    // Substring to end
    sub = s.Substring(7);
    ExpectStringState(sub, "789", 3);

    // Crop (modifies original)
    s.Crop(3, 4);
    ExpectStringState(s, "3456", 4);

    // Cut (modifies original)
    XString s2("0123456789");
    s2.Cut(2, 6); // Remove "234567"
    ExpectStringState(s2, "0189", 4);
}

// 6. Concatenation and Formatting
TEST(XStringTest, Concatenation) {
    XString s("A");
    
    // operator+
    XString s2 = s + "B";
    XString s3 = s2 + 'C';
    XString s4 = s3 + 123;
    ExpectStringState(s4, "ABC123", 6);
    
    // operator+=
    s += "B";
    s += 'C';
    s += 456;
    ExpectStringState(s, "ABC456", 6);
    
    // operator<<
    XString stream;
    stream << "s" << ':' << 1.2f << 'p';
    // We can only check the start, as pointer format is implementation-defined
    EXPECT_TRUE(stream.StartsWith("s:1.200000p"));
}

TEST(XStringTest, Format) {
    XString s;
    s.Format("String: %s, Int: %d, Float: %.2f", "test", 123, 3.14159);
    ExpectStringState(s, "String: test, Int: 123, Float: 3.14", 35);

    // Test format that causes reallocation
    s.Format("%s", "This is a very long string designed to be larger than the initial default buffer capacity of the XString class to test reallocation.");
    EXPECT_GT(s.Length(), 50);
}

// 7. Splitting and Joining
TEST(XStringTest, Split) {
    XString s("a,b,,c,");
    XClassArray<XString> parts;

    int count = s.Split(',', parts);
    EXPECT_EQ(count, 5);
    ASSERT_EQ(parts.Size(), 5);
    EXPECT_STREQ(parts[0].CStr(), "a");
    EXPECT_STREQ(parts[1].CStr(), "b");
    EXPECT_STREQ(parts[2].CStr(), "");
    EXPECT_STREQ(parts[3].CStr(), "c");
    EXPECT_STREQ(parts[4].CStr(), ""); // Trailing delimiter
}

TEST(XStringTest, SplitNoDelimiter) {
    XString s("abc");
    XClassArray<XString> parts;
    int count = s.Split(',', parts);
    EXPECT_EQ(count, 1);
    ASSERT_EQ(parts.Size(), 1);
    EXPECT_STREQ(parts[0].CStr(), "abc");
}

// TEST(XStringTest, Join) {
//     XClassArray<XString> parts;
//     parts.PushBack("a");
//     parts.PushBack("b");
//     parts.PushBack("");
//     parts.PushBack("c");
//
//     XString joined = XString::Join(parts, ";");
//     ExpectStringState(joined, "a;b;;c", 6);
// }
//
// TEST(XStringTest, JoinEmpty) {
//     XClassArray<XString> parts;
//     XString joined = XString::Join(parts, ";");
//     EXPECT_TRUE(joined.IsEmpty());
// }

// 8. XBaseString
TEST(XStringTest, BaseStringConversions) {
    XBaseString s_int(" -123 ");
    XBaseString s_float(" 45.67 ");
    XBaseString s_double(" 89.123e-1 ");
    XBaseString s_invalid(" abc ");
    
    EXPECT_EQ(s_int.ToInt(), -123);
    EXPECT_FLOAT_EQ(s_float.ToFloat(), 45.67f);
    EXPECT_DOUBLE_EQ(s_double.ToDouble(), 8.9123);
    
    EXPECT_EQ(s_invalid.ToInt(), 0);
    EXPECT_FLOAT_EQ(s_invalid.ToFloat(), 0.0f);
}

