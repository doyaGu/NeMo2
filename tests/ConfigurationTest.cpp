#include <gtest/gtest.h>
#include <cstdio>
#include <string>

#include "VxConfiguration.h"

// Test fixture for VxConfiguration tests
class VxConfigurationTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = new VxConfiguration();
    }

    void TearDown() override {
        delete config;
        config = nullptr;

        // Clean up any test files
        std::remove("test_config.txt");
        std::remove("test_save.txt");
    }

    VxConfiguration *config;
};

// Test fixture for VxConfigurationSection tests
class VxConfigurationSectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = new VxConfiguration();
        section = config->CreateSubSection(nullptr, "TestSection");
    }

    void TearDown() override {
        delete config;
        config = nullptr;
        section = nullptr;
    }

    VxConfiguration *config;
    VxConfigurationSection *section;
};

// Test fixture for VxConfigurationEntry tests
class VxConfigurationEntryTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = new VxConfiguration();
        section = config->CreateSubSection(nullptr, "TestSection");
    }

    void TearDown() override {
        delete config;
        config = nullptr;
        section = nullptr;
    }

    VxConfiguration *config;
    VxConfigurationSection *section;
};

// Helper function to create a test configuration file
void CreateTestConfigFile(const std::string &filename, const std::string &content) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (fp) {
        fwrite(content.c_str(), sizeof(char), content.size(), fp);
        fclose(fp);
    }
}

// =============================================================================
// VxConfiguration Tests
TEST_F(VxConfigurationTest, ConstructorInitializesCorrectly) {
    EXPECT_NE(config, nullptr);
    EXPECT_EQ(config->GetNumberOfSubSections(), 0);
    EXPECT_EQ(config->GetNumberOfEntries(), 0);
}

TEST_F(VxConfigurationTest, AddStringEntryToRoot) {
    VxConfigurationEntry *result = nullptr;
    EXPECT_TRUE(config->AddEntry(nullptr, "TestKey", "TestValue", &result));
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(config->GetNumberOfEntries(), 1);

    VxConfigurationEntry *entry = config->GetEntry("TestKey", FALSE);
    EXPECT_NE(entry, nullptr);
    EXPECT_STREQ(entry->GetValue(), "TestValue");
}

TEST_F(VxConfigurationTest, AddIntegerEntry) {
    VxConfigurationEntry *result = nullptr;
    EXPECT_TRUE(config->AddEntry(nullptr, "IntKey", 42, &result));
    EXPECT_NE(result, nullptr);

    VxConfigurationEntry *entry = config->GetEntry("IntKey", FALSE);
    EXPECT_NE(entry, nullptr);

    int value;
    EXPECT_TRUE(entry->GetValueAsInteger(value));
    EXPECT_EQ(value, 42);
}

TEST_F(VxConfigurationTest, AddFloatEntry) {
    VxConfigurationEntry *result = nullptr;
    EXPECT_TRUE(config->AddEntry(nullptr, "FloatKey", 3.14f, &result));
    EXPECT_NE(result, nullptr);

    VxConfigurationEntry *entry = config->GetEntry("FloatKey", FALSE);
    EXPECT_NE(entry, nullptr);

    float value;
    EXPECT_TRUE(entry->GetValueAsFloat(value));
    EXPECT_FLOAT_EQ(value, 3.14f);
}

TEST_F(VxConfigurationTest, AddEntryToSubSection) {
    VxConfigurationEntry *result = nullptr;
    EXPECT_TRUE(config->AddEntry("Section1", "Key1", "Value1", &result));
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(config->GetNumberOfSubSections(), 1);

    VxConfigurationSection *section = config->GetSubSection("Section1", FALSE);
    EXPECT_NE(section, nullptr);
    EXPECT_EQ(section->GetNumberOfEntries(), 1);

    VxConfigurationEntry *entry = section->GetEntry("Key1");
    EXPECT_NE(entry, nullptr);
    EXPECT_STREQ(entry->GetValue(), "Value1");
}

TEST_F(VxConfigurationTest, CreateSubSection) {
    VxConfigurationSection *section = config->CreateSubSection(nullptr, "TestSection");
    EXPECT_NE(section, nullptr);
    EXPECT_EQ(config->GetNumberOfSubSections(), 1);
    EXPECT_STREQ(section->GetName(), "TestSection");
}

TEST_F(VxConfigurationTest, CreateNestedSubSection) {
    VxConfigurationSection *parent = config->CreateSubSection(nullptr, "Parent");
    EXPECT_NE(parent, nullptr);

    VxConfigurationSection *child = config->CreateSubSection("Parent", "Child");
    EXPECT_NE(child, nullptr);
    EXPECT_EQ(parent->GetNumberOfSubSections(), 1);
    EXPECT_STREQ(child->GetName(), "Child");
}

TEST_F(VxConfigurationTest, DotNotationEntry) {
    EXPECT_TRUE(config->AddEntry("Section1.SubSection", "Key1", "Value1"));

    VxConfigurationEntry *entry = config->GetEntry("Section1.SubSection.Key1", TRUE);
    EXPECT_NE(entry, nullptr);
    EXPECT_STREQ(entry->GetValue(), "Value1");
}

TEST_F(VxConfigurationTest, DeleteEntry) {
    config->AddEntry(nullptr, "TempKey", "TempValue");
    EXPECT_EQ(config->GetNumberOfEntries(), 1);

    EXPECT_TRUE(config->DeleteEntry(nullptr, "TempKey"));
    EXPECT_EQ(config->GetNumberOfEntries(), 0);

    VxConfigurationEntry *entry = config->GetEntry("TempKey", FALSE);
    EXPECT_EQ(entry, nullptr);
}

TEST_F(VxConfigurationTest, DeleteSection) {
    config->CreateSubSection(nullptr, "TempSection");
    EXPECT_EQ(config->GetNumberOfSubSections(), 1);

    EXPECT_TRUE(config->DeleteSection(nullptr, "TempSection"));
    EXPECT_EQ(config->GetNumberOfSubSections(), 0);

    VxConfigurationSection *section = config->GetSubSection("TempSection", FALSE);
    EXPECT_EQ(section, nullptr);
}

TEST_F(VxConfigurationTest, RemoveEntry) {
    config->AddEntry(nullptr, "TempKey", "TempValue");
    EXPECT_EQ(config->GetNumberOfEntries(), 1);

    VxConfigurationEntry *removed = config->RemoveEntry(nullptr, "TempKey");
    EXPECT_NE(removed, nullptr);
    EXPECT_EQ(config->GetNumberOfEntries(), 0);

    // Entry should be removed but not deleted
    EXPECT_STREQ(removed->GetValue(), "TempValue");
    delete removed; // Clean up manually
}

TEST_F(VxConfigurationTest, RemoveSection) {
    VxConfigurationSection *section = config->CreateSubSection(nullptr, "TempSection");
    EXPECT_EQ(config->GetNumberOfSubSections(), 1);

    VxConfigurationSection *removed = config->RemoveSection(nullptr, "TempSection");
    EXPECT_NE(removed, nullptr);
    EXPECT_EQ(config->GetNumberOfSubSections(), 0);

    // Section should be removed but not deleted
    EXPECT_STREQ(removed->GetName(), "TempSection");
    delete removed; // Clean up manually
}

TEST_F(VxConfigurationTest, DefaultConfiguration) {
    EXPECT_TRUE(config->AddDefaultEntry(nullptr, "DefaultKey", "DefaultValue"));
    EXPECT_TRUE(config->AddDefaultEntry("DefaultSection", "Key1", 100));

    // Should find default values when main config is empty
    VxConfigurationEntry *entry = config->GetEntry("DefaultKey", FALSE);
    EXPECT_NE(entry, nullptr);
    EXPECT_STREQ(entry->GetValue(), "DefaultValue");

    VxConfigurationEntry *sectionEntry = config->GetEntry("DefaultSection.Key1", TRUE);
    EXPECT_NE(sectionEntry, nullptr);

    int value;
    EXPECT_TRUE(sectionEntry->GetValueAsInteger(value));
    EXPECT_EQ(value, 100);
}

TEST_F(VxConfigurationTest, MainConfigOverridesDefault) {
    // Add default value
    config->AddDefaultEntry(nullptr, "TestKey", "DefaultValue");

    // Add main value
    config->AddEntry(nullptr, "TestKey", "MainValue");

    // Should get main value, not default
    VxConfigurationEntry *entry = config->GetEntry("TestKey", FALSE);
    EXPECT_NE(entry, nullptr);
    EXPECT_STREQ(entry->GetValue(), "MainValue");
}

TEST_F(VxConfigurationTest, RecursiveCounting) {
    config->AddEntry("Section1", "Key1", "Value1");
    config->AddEntry("Section1.SubSection", "Key2", "Value2");
    config->AddEntry("Section2", "Key3", "Value3");

    EXPECT_EQ(config->GetNumberOfSubSectionsRecursive(), 3); // Section1, SubSection, Section2
    EXPECT_EQ(config->GetNumberOfEntriesRecursive(), 3);
}

TEST_F(VxConfigurationTest, IterateEntries) {
    config->AddEntry(nullptr, "Key1", "Value1");
    config->AddEntry(nullptr, "Key2", "Value2");
    config->AddEntry(nullptr, "Key3", "Value3");

    ConstEntryIt it = config->BeginEntries();
    int count = 0;
    VxConfigurationEntry *entry;

    while ((entry = config->GetNextEntry(it)) != nullptr) {
        count++;
        EXPECT_NE(entry, nullptr);
        EXPECT_NE(entry->GetName(), nullptr);
        EXPECT_NE(entry->GetValue(), nullptr);
    }

    EXPECT_EQ(count, 3);
}

TEST_F(VxConfigurationTest, IterateSections) {
    config->CreateSubSection(nullptr, "Section1");
    config->CreateSubSection(nullptr, "Section2");
    config->CreateSubSection(nullptr, "Section3");

    ConstSectionIt it = config->BeginSections();
    int count = 0;
    VxConfigurationSection *section;

    while ((section = config->GetNextSection(it)) != nullptr) {
        count++;
        EXPECT_NE(section, nullptr);
        EXPECT_NE(section->GetName(), nullptr);
    }

    EXPECT_EQ(count, 3);
}

TEST_F(VxConfigurationTest, BuildFromMemory) {
    const char *configText =
        "# Test configuration\n"
        "RootKey = RootValue\n"
        "\n"
        "<Section1>\n"
        "  Key1 = Value1\n"
        "  Key2 = Value2\n"
        "</Section1>\n"
        "\n"
        "<Section2>\n"
        "  IntKey = 42\n"
        "  FloatKey = 3.14\n"
        "</Section2>\n";

    int line = 0;
    XString error;
    EXPECT_TRUE(config->BuildFromMemory(configText, line, error));

    // Check root entry
    VxConfigurationEntry *rootEntry = config->GetEntry("RootKey", FALSE);
    EXPECT_NE(rootEntry, nullptr);
    EXPECT_STREQ(rootEntry->GetValue(), "RootValue");

    // Check section 1
    VxConfigurationSection *section1 = config->GetSubSection("Section1", FALSE);
    EXPECT_NE(section1, nullptr);
    EXPECT_EQ(section1->GetNumberOfEntries(), 2);

    VxConfigurationEntry *key1 = section1->GetEntry("Key1");
    EXPECT_NE(key1, nullptr);
    EXPECT_STREQ(key1->GetValue(), "Value1");

    // Check section 2
    VxConfigurationSection *section2 = config->GetSubSection("Section2", FALSE);
    EXPECT_NE(section2, nullptr);

    VxConfigurationEntry *intKey = section2->GetEntry("IntKey");
    EXPECT_NE(intKey, nullptr);
    int intValue;
    EXPECT_TRUE(intKey->GetValueAsInteger(intValue));
    EXPECT_EQ(intValue, 42);

    VxConfigurationEntry *floatKey = section2->GetEntry("FloatKey");
    EXPECT_NE(floatKey, nullptr);
    float floatValue;
    EXPECT_TRUE(floatKey->GetValueAsFloat(floatValue));
    EXPECT_FLOAT_EQ(floatValue, 3.14f);
}

TEST_F(VxConfigurationTest, BuildFromFile) {
    const std::string configContent =
        "# Test configuration file\n"
        "FileKey = FileValue\n"
        "\n"
        "<FileSection>\n"
        "  Key1 = Value1\n"
        "  Key2 = 123\n"
        "</FileSection>\n";

    CreateTestConfigFile("test_config.txt", configContent);

    int line = 0;
    XString error;
    EXPECT_TRUE(config->BuildFromFile("test_config.txt", line, error));

    VxConfigurationEntry *entry = config->GetEntry("FileKey", FALSE);
    EXPECT_NE(entry, nullptr);
    EXPECT_STREQ(entry->GetValue(), "FileValue");

    VxConfigurationSection *section = config->GetSubSection("FileSection", FALSE);
    EXPECT_NE(section, nullptr);
    EXPECT_EQ(section->GetNumberOfEntries(), 2);
}

TEST_F(VxConfigurationTest, SaveToFile) {
    // Add some configuration data
    config->AddEntry(nullptr, "RootKey", "RootValue");
    config->AddEntry("Section1", "Key1", "Value1");
    config->AddEntry("Section1", "Key2", 42);

    // Save to file
    EXPECT_TRUE(config->SaveToFile("test_save.txt"));

    // Verify file was created and has content
    FILE *file = fopen("test_save.txt", "rb");
    EXPECT_TRUE(file != nullptr);

    // Get file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Read content
    std::string content(fileSize, '\0');
    fread(&content[0], sizeof(char), fileSize, file);
    fclose(file);

    EXPECT_FALSE(content.empty());
    EXPECT_NE(content.find("RootKey = RootValue"), std::string::npos);
    EXPECT_NE(content.find("<Section1>"), std::string::npos);
    EXPECT_NE(content.find("</Section1>"), std::string::npos);
    EXPECT_NE(content.find("Key1 = Value1"), std::string::npos);
    EXPECT_NE(content.find("Key2 = 42"), std::string::npos);
}

TEST_F(VxConfigurationTest, Clear) {
    config->AddEntry(nullptr, "Key1", "Value1");
    config->AddEntry("Section1", "Key2", "Value2");

    EXPECT_GT(config->GetNumberOfEntries(), 0);
    EXPECT_GT(config->GetNumberOfSubSections(), 0);

    config->Clear();

    EXPECT_EQ(config->GetNumberOfEntries(), 0);
    EXPECT_EQ(config->GetNumberOfSubSections(), 0);
}

// Edge cases and error conditions
TEST_F(VxConfigurationTest, NullPointerHandling) {
    // Should handle null pointers gracefully
    EXPECT_FALSE(config->AddEntry(nullptr, nullptr, "Value"));
    EXPECT_EQ(config->CreateSubSection(nullptr, nullptr), nullptr);
    EXPECT_EQ(config->GetEntry(nullptr, FALSE), nullptr);
    EXPECT_EQ(config->GetSubSection(nullptr, FALSE), nullptr);
}

TEST_F(VxConfigurationTest, EmptyStringHandling) {
    EXPECT_TRUE(config->AddEntry(nullptr, "EmptyKey", ""));

    VxConfigurationEntry *entry = config->GetEntry("EmptyKey", FALSE);
    EXPECT_NE(entry, nullptr);
    EXPECT_STREQ(entry->GetValue(), "");
}

TEST_F(VxConfigurationTest, NonExistentLookup) {
    VxConfigurationEntry *entry = config->GetEntry("NonExistent", FALSE);
    EXPECT_EQ(entry, nullptr);

    VxConfigurationSection *section = config->GetSubSection("NonExistent", FALSE);
    EXPECT_EQ(section, nullptr);
}

TEST_F(VxConfigurationTest, InvalidFileOperations) {
    int line = 0;
    XString error;

    // Try to load non-existent file
    EXPECT_FALSE(config->BuildFromFile("nonexistent.txt", line, error));
    EXPECT_FALSE(error.Empty());

    // Try to save to invalid path
    EXPECT_FALSE(config->SaveToFile("/invalid/path/file.txt"));
}

// =============================================================================
// VxConfigurationSection Tests
TEST_F(VxConfigurationSectionTest, AddAndGetEntry) {
    section->AddEntry("TestKey", "TestValue");
    EXPECT_EQ(section->GetNumberOfEntries(), 1);

    VxConfigurationEntry *entry = section->GetEntry("TestKey");
    EXPECT_NE(entry, nullptr);
    EXPECT_STREQ(entry->GetValue(), "TestValue");
}

TEST_F(VxConfigurationSectionTest, AddDifferentTypes) {
    section->AddEntry("StringKey", "StringValue");
    section->AddEntry("IntKey", 42);
    section->AddEntry("LongKey", 123456789L);
    section->AddEntry("UIntKey", 42U);
    section->AddEntry("ULongKey", 123456789UL);
    section->AddEntry("FloatKey", 3.14f);

    EXPECT_EQ(section->GetNumberOfEntries(), 6);

    // Test type conversions
    VxConfigurationEntry *intEntry = section->GetEntry("IntKey");
    int intValue;
    EXPECT_TRUE(intEntry->GetValueAsInteger(intValue));
    EXPECT_EQ(intValue, 42);

    VxConfigurationEntry *floatEntry = section->GetEntry("FloatKey");
    float floatValue;
    EXPECT_TRUE(floatEntry->GetValueAsFloat(floatValue));
    EXPECT_FLOAT_EQ(floatValue, 3.14f);
}

TEST_F(VxConfigurationSectionTest, UpdateExistingEntry) {
    section->AddEntry("Key1", "OriginalValue");

    VxConfigurationEntry *entry1 = section->GetEntry("Key1");
    EXPECT_STREQ(entry1->GetValue(), "OriginalValue");

    // Add again with same key - should update
    section->AddEntry("Key1", "UpdatedValue");
    EXPECT_EQ(section->GetNumberOfEntries(), 1); // Still only one entry

    VxConfigurationEntry *entry2 = section->GetEntry("Key1");
    EXPECT_STREQ(entry2->GetValue(), "UpdatedValue");
    EXPECT_EQ(entry1, entry2); // Should be the same entry object
}

TEST_F(VxConfigurationSectionTest, CreateSubSection) {
    VxConfigurationSection *subSection = section->CreateSubSection("SubSection");
    EXPECT_NE(subSection, nullptr);
    EXPECT_EQ(section->GetNumberOfSubSections(), 1);
    EXPECT_STREQ(subSection->GetName(), "SubSection");
    EXPECT_EQ(subSection->GetParent(), section);
}

TEST_F(VxConfigurationSectionTest, DeleteEntry) {
    section->AddEntry("TempKey", "TempValue");
    EXPECT_EQ(section->GetNumberOfEntries(), 1);

    EXPECT_TRUE(section->DeleteEntry("TempKey"));
    EXPECT_EQ(section->GetNumberOfEntries(), 0);

    VxConfigurationEntry *entry = section->GetEntry("TempKey");
    EXPECT_EQ(entry, nullptr);
}

TEST_F(VxConfigurationSectionTest, DeleteSubSection) {
    VxConfigurationSection *subSection = section->CreateSubSection("TempSection");
    subSection->AddEntry("Key1", "Value1");
    EXPECT_EQ(section->GetNumberOfSubSections(), 1);

    EXPECT_TRUE(section->DeleteSection("TempSection"));
    EXPECT_EQ(section->GetNumberOfSubSections(), 0);

    VxConfigurationSection *found = section->GetSubSection("TempSection");
    EXPECT_EQ(found, nullptr);
}

TEST_F(VxConfigurationSectionTest, IterateEntries) {
    section->AddEntry("Key1", "Value1");
    section->AddEntry("Key2", "Value2");
    section->AddEntry("Key3", "Value3");

    ConstEntryIt it = section->BeginChildEntry();
    int count = 0;
    VxConfigurationEntry *entry;

    while ((entry = section->GetNextChildEntry(it)) != nullptr) {
        count++;
        EXPECT_NE(entry, nullptr);
    }

    EXPECT_EQ(count, 3);
}

TEST_F(VxConfigurationSectionTest, IterateSubSections) {
    section->CreateSubSection("Sub1");
    section->CreateSubSection("Sub2");
    section->CreateSubSection("Sub3");

    ConstSectionIt it = section->BeginChildSection();
    int count = 0;
    VxConfigurationSection *subSection;

    while ((subSection = section->GetNextChildSection(it)) != nullptr) {
        count++;
        EXPECT_NE(subSection, nullptr);
    }

    EXPECT_EQ(count, 3);
}

TEST_F(VxConfigurationSectionTest, RecursiveCounting) {
    section->AddEntry("Key1", "Value1");

    VxConfigurationSection *sub1 = section->CreateSubSection("Sub1");
    sub1->AddEntry("SubKey1", "SubValue1");
    sub1->AddEntry("SubKey2", "SubValue2");

    VxConfigurationSection *sub2 = sub1->CreateSubSection("Sub2");
    sub2->AddEntry("DeepKey", "DeepValue");

    EXPECT_EQ(section->GetNumberOfSubSectionsRecursive(), 2); // Sub1, Sub2
    EXPECT_EQ(section->GetNumberOfEntriesRecursive(), 4);     // Key1, SubKey1, SubKey2, DeepKey
}

TEST_F(VxConfigurationSectionTest, Clear) {
    section->AddEntry("Key1", "Value1");
    section->CreateSubSection("SubSection");

    EXPECT_GT(section->GetNumberOfEntries(), 0);
    EXPECT_GT(section->GetNumberOfSubSections(), 0);

    section->Clear();

    EXPECT_EQ(section->GetNumberOfEntries(), 0);
    EXPECT_EQ(section->GetNumberOfSubSections(), 0);
}

// =============================================================================
// VxConfigurationEntry Tests
TEST_F(VxConfigurationEntryTest, StringValue) {
    VxConfigurationEntry *entry;
    section->AddEntry("StringKey", "TestString", &entry);

    EXPECT_STREQ(entry->GetName(), "StringKey");
    EXPECT_STREQ(entry->GetValue(), "TestString");
    EXPECT_EQ(entry->GetParent(), section);
}

TEST_F(VxConfigurationEntryTest, IntegerValue) {
    VxConfigurationEntry *entry;
    section->AddEntry("IntKey", 42, &entry);

    EXPECT_STREQ(entry->GetName(), "IntKey");

    int value;
    EXPECT_TRUE(entry->GetValueAsInteger(value));
    EXPECT_EQ(value, 42);

    // Should also work as string
    EXPECT_STREQ(entry->GetValue(), "42");
}

TEST_F(VxConfigurationEntryTest, LongValue) {
    VxConfigurationEntry *entry;
    long testValue = 123456789L;
    section->AddEntry("LongKey", testValue, &entry);

    long value;
    EXPECT_TRUE(entry->GetValueAsLong(value));
    EXPECT_EQ(value, testValue);
}

TEST_F(VxConfigurationEntryTest, UnsignedIntegerValue) {
    VxConfigurationEntry *entry;
    unsigned int testValue = 4294967295U;
    section->AddEntry("UIntKey", testValue, &entry);

    unsigned int value;
    EXPECT_TRUE(entry->GetValueAsUnsignedInteger(value));
    EXPECT_EQ(value, testValue);
}

TEST_F(VxConfigurationEntryTest, UnsignedLongValue) {
    VxConfigurationEntry *entry;
    unsigned long testValue = 4294967295UL;
    section->AddEntry("ULongKey", testValue, &entry);

    unsigned long value;
    EXPECT_TRUE(entry->GetValueAsUnsignedLong(value));
    EXPECT_EQ(value, testValue);
}

TEST_F(VxConfigurationEntryTest, FloatValue) {
    VxConfigurationEntry *entry;
    section->AddEntry("FloatKey", 3.14159f, &entry);

    float value;
    EXPECT_TRUE(entry->GetValueAsFloat(value));
    EXPECT_FLOAT_EQ(value, 3.14159f);
}

TEST_F(VxConfigurationEntryTest, SetValue) {
    VxConfigurationEntry *entry;
    section->AddEntry("TestKey", "InitialValue", &entry);

    // Test setting different types
    entry->SetValue("StringValue");
    EXPECT_STREQ(entry->GetValue(), "StringValue");

    entry->SetValue(100);
    int intValue;
    EXPECT_TRUE(entry->GetValueAsInteger(intValue));
    EXPECT_EQ(intValue, 100);

    entry->SetValue(2.718f);
    float floatValue;
    EXPECT_TRUE(entry->GetValueAsFloat(floatValue));
    EXPECT_FLOAT_EQ(floatValue, 2.718f);
}

TEST_F(VxConfigurationEntryTest, TypeConversionFailure) {
    VxConfigurationEntry *entry;
    section->AddEntry("TextKey", "NotANumber", &entry);

    int intValue;
    EXPECT_FALSE(entry->GetValueAsInteger(intValue));

    float floatValue;
    EXPECT_FALSE(entry->GetValueAsFloat(floatValue));

    long longValue;
    EXPECT_FALSE(entry->GetValueAsLong(longValue));
}

TEST_F(VxConfigurationEntryTest, EmptyValue) {
    VxConfigurationEntry *entry;
    section->AddEntry("EmptyKey", "", &entry);

    EXPECT_STREQ(entry->GetValue(), "");

    int intValue;
    EXPECT_FALSE(entry->GetValueAsInteger(intValue));
}

TEST_F(VxConfigurationEntryTest, NullValue) {
    VxConfigurationEntry *entry;
    section->AddEntry("NullKey", static_cast<const char *>(nullptr), &entry);

    EXPECT_STREQ(entry->GetValue(), "");
}

// =============================================================================
// Integration Tests
class VxConfigurationIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = new VxConfiguration();
    }

    void TearDown() override {
        delete config;
        config = nullptr;
        std::remove("integration_test.txt");
    }

    VxConfiguration *config;
};

TEST_F(VxConfigurationIntegrationTest, ComplexConfigurationRoundTrip) {
    // Build a complex configuration
    config->AddEntry(nullptr, "RootKey", "RootValue");
    config->AddEntry(nullptr, "RootInt", 42);

    // Add hierarchical sections
    config->AddEntry("Graphics", "Resolution", "1920x1080");
    config->AddEntry("Graphics", "FullScreen", 1);
    config->AddEntry("Graphics.Quality", "TextureQuality", "High");
    config->AddEntry("Graphics.Quality", "ShadowQuality", "Medium");

    config->AddEntry("Audio", "MasterVolume", 0.8f);
    config->AddEntry("Audio", "MusicVolume", 0.6f);
    config->AddEntry("Audio.Effects", "ReverbEnabled", 1);

    // Save to file
    EXPECT_TRUE(config->SaveToFile("integration_test.txt"));

    // Create new configuration and load
    VxConfiguration *loadedConfig = new VxConfiguration();
    int line = 0;
    XString error;
    EXPECT_TRUE(loadedConfig->BuildFromFile("integration_test.txt", line, error));

    // Verify all data was preserved
    VxConfigurationEntry *rootKey = loadedConfig->GetEntry("RootKey", FALSE);
    EXPECT_NE(rootKey, nullptr);
    EXPECT_STREQ(rootKey->GetValue(), "RootValue");

    VxConfigurationEntry *resolution = loadedConfig->GetEntry("Graphics.Resolution", TRUE);
    EXPECT_NE(resolution, nullptr);
    EXPECT_STREQ(resolution->GetValue(), "1920x1080");

    VxConfigurationEntry *textureQuality = loadedConfig->GetEntry("Graphics.Quality.TextureQuality", TRUE);
    EXPECT_NE(textureQuality, nullptr);
    EXPECT_STREQ(textureQuality->GetValue(), "High");

    VxConfigurationEntry *masterVolume = loadedConfig->GetEntry("Audio.MasterVolume", TRUE);
    EXPECT_NE(masterVolume, nullptr);
    float volume;
    EXPECT_TRUE(masterVolume->GetValueAsFloat(volume));
    EXPECT_FLOAT_EQ(volume, 0.8f);

    delete loadedConfig;
}

TEST_F(VxConfigurationIntegrationTest, DefaultConfigurationFallback) {
    // Set up default configuration
    config->AddDefaultEntry(nullptr, "DefaultKey", "DefaultValue");
    config->AddDefaultEntry("Defaults", "Setting1", 100);
    config->AddDefaultEntry("Defaults", "Setting2", "Auto");

    // Override some defaults
    config->AddEntry("Defaults", "Setting1", 200);

    // Test fallback behavior
    VxConfigurationEntry *defaultKey = config->GetEntry("DefaultKey", FALSE);
    EXPECT_NE(defaultKey, nullptr);
    EXPECT_STREQ(defaultKey->GetValue(), "DefaultValue");

    VxConfigurationEntry *setting1 = config->GetEntry("Defaults.Setting1", TRUE);
    EXPECT_NE(setting1, nullptr);
    int value;
    EXPECT_TRUE(setting1->GetValueAsInteger(value));
    EXPECT_EQ(value, 200); // Should get overridden value

    VxConfigurationEntry *setting2 = config->GetEntry("Defaults.Setting2", TRUE);
    EXPECT_NE(setting2, nullptr);
    EXPECT_STREQ(setting2->GetValue(), "Auto"); // Should get default value
}

// =============================================================================
// Performance Tests
class VxConfigurationPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = new VxConfiguration();
    }

    void TearDown() override {
        delete config;
        config = nullptr;
    }

    VxConfiguration *config;
};

TEST_F(VxConfigurationPerformanceTest, LargeNumberOfEntries) {
    const int numEntries = 10000;

    // Add many entries
    for (int i = 0; i < numEntries; i++) {
        char keyName[32];
        sprintf(keyName, "Key%d", i);
        config->AddEntry(nullptr, keyName, i);
    }

    EXPECT_EQ(config->GetNumberOfEntries(), numEntries);

    // Test lookup performance
    for (int i = 0; i < 100; i++) {
        char keyName[32];
        sprintf(keyName, "Key%d", i * 100);
        VxConfigurationEntry *entry = config->GetEntry(keyName, FALSE);
        EXPECT_NE(entry, nullptr);

        int value;
        EXPECT_TRUE(entry->GetValueAsInteger(value));
        EXPECT_EQ(value, i * 100);
    }
}

TEST_F(VxConfigurationPerformanceTest, DeepHierarchy) {
    const int depth = 100;

    // Create deep hierarchy
    char sectionPath[1000] = "";
    for (int i = 0; i < depth; i++) {
        char levelName[32];
        sprintf(levelName, "Level%d", i);

        if (i > 0) {
            strcat(sectionPath, ".");
        }
        strcat(sectionPath, levelName);

        char entryKey[32];
        sprintf(entryKey, "Key%d", i);
        config->AddEntry(sectionPath, entryKey, i);
    }

    // Test deep lookup
    char deepKey[1100];
    sprintf(deepKey, "%s.Key%d", sectionPath, depth - 1);
    VxConfigurationEntry *entry = config->GetEntry(deepKey, TRUE);
    EXPECT_NE(entry, nullptr);

    int value;
    EXPECT_TRUE(entry->GetValueAsInteger(value));
    EXPECT_EQ(value, depth - 1);
}

// =============================================================================
// XML-like Format Specific Tests
class VxConfigurationFormatTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = new VxConfiguration();
    }

    void TearDown() override {
        delete config;
        config = nullptr;
    }

    VxConfiguration *config;
};

TEST_F(VxConfigurationFormatTest, CK2_3D_Format) {
    const char *ck2Config =
        "# CK2_3D Render Engine config File\n"
        "<CK2_3D>\n"
        "  DisablePerspectiveCorrection = 0\n"
        "  ForceLinearFog = 0\n"
        "  ForceSoftware = 0\n"
        "  DisableFilter = 0\n"
        "  EnsureVertexShader = 0\n"
        "  DisableDithering = 0\n"
        "  DisableMipmap = 0\n"
        "  DisableSpecular = 0\n"
        "  Antialias = 0\n"
        "  UseIndexBuffers = 0\n"
        "  EnableScreenDump = 0\n"
        "  EnableDebugMode = 0\n"
        "  VertexCache = 16\n"
        "  SortTransparentObjects = 1\n"
        "  TextureCacheManagement = 1\n"
        "  TextureVideoFormat = _16_ARGB1555\n"
        "  SpriteVideoFormat = _16_ARGB1555\n"
        "</CK2_3D>\n";

    int line = 0;
    XString error;
    EXPECT_TRUE(config->BuildFromMemory(ck2Config, line, error));

    // Verify the section was created
    VxConfigurationSection *ck2Section = config->GetSubSection("CK2_3D", FALSE);
    EXPECT_NE(ck2Section, nullptr);
    EXPECT_EQ(ck2Section->GetNumberOfEntries(), 17);

    // Test a few specific entries
    VxConfigurationEntry *vertexCache = ck2Section->GetEntry("VertexCache");
    EXPECT_NE(vertexCache, nullptr);
    int value;
    EXPECT_TRUE(vertexCache->GetValueAsInteger(value));
    EXPECT_EQ(value, 16);

    VxConfigurationEntry *textureFormat = ck2Section->GetEntry("TextureVideoFormat");
    EXPECT_NE(textureFormat, nullptr);
    EXPECT_STREQ(textureFormat->GetValue(), "_16_ARGB1555");
}

TEST_F(VxConfigurationFormatTest, NestedXMLTags) {
    const char *nestedConfig =
        "<Parent>\n"
        "  ParentKey = ParentValue\n"
        "  <Child>\n"
        "    ChildKey = ChildValue\n"
        "    <GrandChild>\n"
        "      GrandChildKey = GrandChildValue\n"
        "    </GrandChild>\n"
        "  </Child>\n"
        "</Parent>\n";

    int line = 0;
    XString error;
    EXPECT_TRUE(config->BuildFromMemory(nestedConfig, line, error));

    VxConfigurationSection *parent = config->GetSubSection("Parent", FALSE);
    EXPECT_NE(parent, nullptr);

    VxConfigurationEntry *parentKey = parent->GetEntry("ParentKey");
    EXPECT_NE(parentKey, nullptr);
    EXPECT_STREQ(parentKey->GetValue(), "ParentValue");

    VxConfigurationSection *child = parent->GetSubSection("Child");
    EXPECT_NE(child, nullptr);

    VxConfigurationEntry *childKey = child->GetEntry("ChildKey");
    EXPECT_NE(childKey, nullptr);
    EXPECT_STREQ(childKey->GetValue(), "ChildValue");

    VxConfigurationSection *grandChild = child->GetSubSection("GrandChild");
    EXPECT_NE(grandChild, nullptr);

    VxConfigurationEntry *grandChildKey = grandChild->GetEntry("GrandChildKey");
    EXPECT_NE(grandChildKey, nullptr);
    EXPECT_STREQ(grandChildKey->GetValue(), "GrandChildValue");
}

// =============================================================================
// Main function to run all tests
