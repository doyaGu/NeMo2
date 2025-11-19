#ifndef VXCONFIGURATION_H
#define VXCONFIGURATION_H

#include "XHashTable.h"

class VxConfigurationSection;
class VxConfigurationEntry;

/// @brief A hash table mapping string keys to VxConfigurationEntry pointers.
typedef XHashTable<VxConfigurationEntry *, XString> HashOfEntry;
/// @brief An iterator for HashOfEntry.
typedef XHashTable<VxConfigurationEntry *, XString>::Iterator EntryIt;
/// @brief A const iterator for HashOfEntry.
typedef XHashTable<VxConfigurationEntry *, XString>::ConstIterator ConstEntryIt;
/// @brief A key-value pair for HashOfEntry.
typedef XHashTable<VxConfigurationEntry *, XString>::Pair EntryPair;
/// @brief A hash table mapping string keys to VxConfigurationSection pointers.
typedef XHashTable<VxConfigurationSection *, XString> HashOfSection;
/// @brief An iterator for HashOfSection.
typedef XHashTable<VxConfigurationSection *, XString>::Iterator SectionIt;
/// @brief A const iterator for HashOfSection.
typedef XHashTable<VxConfigurationSection *, XString>::ConstIterator ConstSectionIt;
/// @brief A key-value pair for HashOfSection.
typedef XHashTable<VxConfigurationSection *, XString>::Pair SectionPair;

/**
 * @brief Represents a configuration, structured as a tree of sections and entries.
 *
 * @remarks
 * A configuration can be loaded from or saved to a file. It supports a hierarchical
 * structure of sections, each containing key-value entries. A default configuration
 * can also be maintained.
 *
 * @see VxConfigurationSection, VxConfigurationEntry
 */
class VxConfiguration {
public:
    /**
     * @brief Constructs a VxConfiguration object.
     * @param indent The number of spaces to use for indentation when saving to a file. Defaults to 2.
     */
    VX_EXPORT explicit VxConfiguration(unsigned short indent = 2);

    /**
     * @brief Destructor.
     */
    VX_EXPORT ~VxConfiguration();

    /// @brief Clears the entire configuration.
    VX_EXPORT void Clear();

    /// @brief Clears the default configuration.
    VX_EXPORT void ClearDefault();

    /// @brief Gets the number of top-level subsections in the root.
    VX_EXPORT int GetNumberOfSubSections() const;

    /// @brief Gets the number of entries in the root.
    VX_EXPORT int GetNumberOfEntries() const;

    /// @brief Recursively gets the total number of subsections in the configuration.
    VX_EXPORT int GetNumberOfSubSectionsRecursive() const;

    /// @brief Recursively gets the total number of entries in the configuration.
    VX_EXPORT int GetNumberOfEntriesRecursive() const;

    /// @brief Adds a string entry to a specified section.
    VX_EXPORT XBOOL AddEntry(char *parent, char *ename, const char *evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds an integer entry to a specified section.
    VX_EXPORT XBOOL AddEntry(char *parent, char *ename, int evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds a long integer entry to a specified section.
    VX_EXPORT XBOOL AddEntry(char *parent, char *ename, long evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds an unsigned integer entry to a specified section.
    VX_EXPORT XBOOL AddEntry(char *parent, char *ename, unsigned int evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds an unsigned long integer entry to a specified section.
    VX_EXPORT XBOOL AddEntry(char *parent, char *ename, unsigned long evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds a float entry to a specified section.
    VX_EXPORT XBOOL AddEntry(char *parent, char *ename, float evalue, VxConfigurationEntry **result = NULL);

    /// @brief Creates a new subsection under a specified parent section.
    VX_EXPORT VxConfigurationSection *CreateSubSection(char *parent, char *sname);

    /// @brief Deletes an entry from a specified section.
    VX_EXPORT XBOOL DeleteEntry(char *parent, char *ename);

    /// @brief Deletes a subsection and all its content.
    VX_EXPORT XBOOL DeleteSection(char *parent, char *sname);

    /// @brief Removes an entry from its parent section without deleting it.
    VX_EXPORT VxConfigurationEntry *RemoveEntry(char *parent, char *ename);

    /// @brief Removes a subsection from its parent without deleting it.
    VX_EXPORT VxConfigurationSection *RemoveSection(char *parent, char *sname);

    /// @brief Adds a default string entry to a specified section.
    VX_EXPORT XBOOL AddDefaultEntry(char *parent, char *ename, const char *evalue);

    /// @brief Adds a default integer entry to a specified section.
    VX_EXPORT XBOOL AddDefaultEntry(char *parent, char *ename, int evalue);

    /// @brief Adds a default long integer entry to a specified section.
    VX_EXPORT XBOOL AddDefaultEntry(char *parent, char *ename, long evalue);

    /// @brief Adds a default unsigned integer entry to a specified section.
    VX_EXPORT XBOOL AddDefaultEntry(char *parent, char *ename, unsigned int evalue);

    /// @brief Adds a default unsigned long integer entry to a specified section.
    VX_EXPORT XBOOL AddDefaultEntry(char *parent, char *ename, unsigned long evalue);

    /// @brief Adds a default float entry to a specified section.
    VX_EXPORT XBOOL AddDefaultEntry(char *parent, char *ename, float evalue);

    /// @brief Creates a new default subsection under a specified parent section.
    VX_EXPORT VxConfigurationSection *CreateDefaultSubSection(char *parent, char *sname);

    /// @brief Gets a const iterator to the beginning of the root's subsections.
    VX_EXPORT ConstSectionIt BeginSections() const;

    /// @brief Gets a const iterator to the beginning of the root's entries.
    VX_EXPORT ConstEntryIt BeginEntries() const;

    /// @brief Iterates to the next subsection.
    VX_EXPORT VxConfigurationSection *GetNextSection(ConstSectionIt &it) const;

    /// @brief Iterates to the next entry.
    VX_EXPORT VxConfigurationEntry *GetNextEntry(ConstEntryIt &it) const;

    /// @brief Retrieves a subsection by its path.
    VX_EXPORT VxConfigurationSection *GetSubSection(char *sname, XBOOL usedot) const;

    /// @brief Retrieves an entry by its path.
    VX_EXPORT VxConfigurationEntry *GetEntry(char *ename, XBOOL usedot) const;

    /// @brief Populates the configuration from a data file.
    VX_EXPORT XBOOL BuildFromDataFile(const char *name, XString &error);

    /// @brief Populates the configuration from a text file.
    VX_EXPORT XBOOL BuildFromFile(const char *name, int &cline, XString &error);

    /// @brief Populates the configuration from a memory buffer.
    VX_EXPORT XBOOL BuildFromMemory(const char *buffer, int &cline, XString &error);

    /// @brief Saves the configuration to a data file.
    VX_EXPORT XBOOL SaveToDataFile(const char *name);

    /// @brief Saves the configuration to a text file.
    VX_EXPORT XBOOL SaveToFile(const char *name);

protected:
    /// @brief Creates a subsection within a given root section.
    VxConfigurationSection *CreateSubSection(VxConfigurationSection *root, char *sname, XBOOL usedot) const;

    /// @brief Retrieves a subsection from a given root section.
    VxConfigurationSection *GetSubSection(VxConfigurationSection *root, char *sname, XBOOL usedot) const;

    /// @brief Processes a line to identify and create a section.
    XBOOL ManageSection(char *line, VxConfigurationSection **current, XString &error);

    /// @brief Processes a line to identify and create an entry within a section.
    XBOOL ManageEntry(char *line, VxConfigurationSection *current, XString &error);

    VxConfigurationSection *m_Root; ///< The root section of the configuration.
    VxConfigurationSection *m_DefaultRoot; ///< The root section of the default configuration.
    unsigned short m_Indent; ///< Indentation size for saving to file.
};

/**
 * @brief Represents a section within a configuration, which can contain entries and other sections.
 */
class VxConfigurationSection {
    friend class VxConfiguration;

public:
    /**
     * @brief Destructor.
     */
    VX_EXPORT ~VxConfigurationSection();

    /// @brief Clears all entries and subsections from this section.
    VX_EXPORT void Clear();

    /// @brief Gets the number of immediate subsections.
    VX_EXPORT int GetNumberOfSubSections() const;

    /// @brief Gets the number of immediate entries.
    VX_EXPORT int GetNumberOfEntries() const;

    /// @brief Recursively gets the total number of subsections.
    VX_EXPORT int GetNumberOfSubSectionsRecursive() const;

    /// @brief Recursively gets the total number of entries.
    VX_EXPORT int GetNumberOfEntriesRecursive() const;

    /// @brief Adds a string entry to this section.
    VX_EXPORT void AddEntry(char *ename, const char *evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds an integer entry to this section.
    VX_EXPORT void AddEntry(char *ename, int evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds a long integer entry to this section.
    VX_EXPORT void AddEntry(char *ename, long evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds an unsigned integer entry to this section.
    VX_EXPORT void AddEntry(char *ename, unsigned int evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds an unsigned long integer entry to this section.
    VX_EXPORT void AddEntry(char *ename, unsigned long evalue, VxConfigurationEntry **result = NULL);

    /// @brief Adds a float entry to this section.
    VX_EXPORT void AddEntry(char *ename, float evalue, VxConfigurationEntry **result = NULL);

    /// @brief Creates a new subsection within this section.
    VX_EXPORT VxConfigurationSection *CreateSubSection(char *sname);

    /// @brief Deletes an entry from this section.
    VX_EXPORT XBOOL DeleteEntry(char *ename);

    /// @brief Deletes a subsection and all its contents.
    VX_EXPORT XBOOL DeleteSection(char *sname);

    /// @brief Removes an entry from this section without deleting it.
    VX_EXPORT VxConfigurationEntry *RemoveEntry(char *ename);

    /// @brief Removes a subsection from this section without deleting it.
    VX_EXPORT VxConfigurationSection *RemoveSection(char *sname);

    /// @brief Gets a const iterator to the beginning of this section's entries.
    VX_EXPORT ConstEntryIt BeginChildEntry() const;

    /// @brief Iterates to the next entry in this section.
    VX_EXPORT VxConfigurationEntry *GetNextChildEntry(ConstEntryIt &it) const;

    /// @brief Gets a const iterator to the beginning of this section's subsections.
    VX_EXPORT ConstSectionIt BeginChildSection() const;

    /// @brief Iterates to the next subsection in this section.
    VX_EXPORT VxConfigurationSection *GetNextChildSection(ConstSectionIt &it) const;

    /// @brief Retrieves an entry by name from this section.
    VX_EXPORT VxConfigurationEntry *GetEntry(char *ename) const;

    /// @brief Retrieves a subsection by name from this section.
    VX_EXPORT VxConfigurationSection *GetSubSection(char *sname) const;

    /// @brief Gets the name of the section.
    VX_EXPORT const char *GetName() const;

    /// @brief Gets the parent section.
    VX_EXPORT VxConfigurationSection *GetParent() const;

protected:
    /// @brief Constructs a VxConfigurationSection.
    VxConfigurationSection(char *name, VxConfigurationSection *parent);

    VxConfigurationSection *m_Parent; ///< Pointer to the parent section.
    XString m_Name; ///< The name of the section.
    XHashTable<VxConfigurationEntry *, XString> m_Entries; ///< Hash table of entries in this section.
    XHashTable<VxConfigurationSection *, XString> m_SubSections; ///< Hash table of subsections.
};

/**
 * @brief Represents a key-value pair within a configuration section.
 */
class VxConfigurationEntry {
    friend class VxConfigurationSection;

public:
    /// @brief Destructor.
    VX_EXPORT ~VxConfigurationEntry();

    /// @brief Sets the entry's value from a string.
    VX_EXPORT void SetValue(const char *value);

    /// @brief Sets the entry's value from an integer.
    VX_EXPORT void SetValue(int value);

    /// @brief Sets the entry's value from a long integer.
    VX_EXPORT void SetValue(long value);

    /// @brief Sets the entry's value from an unsigned integer.
    VX_EXPORT void SetValue(unsigned int value);

    /// @brief Sets the entry's value from an unsigned long integer.
    VX_EXPORT void SetValue(unsigned long value);

    /// @brief Sets the entry's value from a float.
    VX_EXPORT void SetValue(float value);

    /// @brief Gets the name (key) of the entry.
    VX_EXPORT const char *GetName() const;

    /// @brief Gets the parent section of the entry.
    VX_EXPORT VxConfigurationSection *GetParent() const;

    /// @brief Gets the value of the entry as a string.
    VX_EXPORT const char *GetValue() const;

    /// @brief Gets the value of the entry as an integer.
    VX_EXPORT XBOOL GetValueAsInteger(int &value) const;

    /// @brief Gets the value of the entry as a long integer.
    VX_EXPORT XBOOL GetValueAsLong(long &value) const;

    /// @brief Gets the value of the entry as an unsigned integer.
    VX_EXPORT XBOOL GetValueAsUnsignedInteger(unsigned int &value) const;

    /// @brief Gets the value of the entry as an unsigned long integer.
    VX_EXPORT XBOOL GetValueAsUnsignedLong(unsigned long &value) const;

    /// @brief Gets the value of the entry as a float.
    VX_EXPORT XBOOL GetValueAsFloat(float &value) const;

protected:
    /// @brief Constructs an entry with a string value.
    VxConfigurationEntry(VxConfigurationSection *parent, const char *name, const char *value);
    /// @brief Constructs an entry with an integer value.
    VxConfigurationEntry(VxConfigurationSection *parent, const char *name, int value);
    // @brief Constructs an entry with a long integer value.
    VxConfigurationEntry(VxConfigurationSection *parent, const char *name, long value);
    /// @brief Constructs an entry with an unsigned integer value.
    VxConfigurationEntry(VxConfigurationSection *parent, const char *name, unsigned int value);
    /// @brief Constructs an entry with a long integer value.
    VxConfigurationEntry(VxConfigurationSection *parent, const char *name, unsigned long value);
    /// @brief Constructs an entry with a float value.
    VxConfigurationEntry(VxConfigurationSection *parent, const char *name, float value);

    XString m_Name; ///< The name (key) of the entry.
    VxConfigurationSection *m_Parent; ///< Pointer to the parent section.
    XString m_Value; ///< The value of the entry, stored as a string.
};

/// @brief Removes leading and trailing whitespace from a string.
char *Shrink(char *str);

/**
 * @brief A simplified configuration class, possibly for interacting with a specific system registry or format.
 */
class VxConfig {
public:
    /// @brief Defines the access mode for opening a configuration section.
    enum Mode {
        READ  = 1, ///< Open for reading.
        WRITE = 2  ///< Open for writing.
    };

    /// @brief Default constructor.
    VX_EXPORT VxConfig();
    /// @brief Destructor.
    VX_EXPORT ~VxConfig();

    /// @brief Opens a section with a specified access mode.
    VX_EXPORT void OpenSection(char *iSection, Mode iOpeningMode);

    /// @brief Closes a currently open section.
    VX_EXPORT void CloseSection(char *iSection);

    /// @brief Writes a string value to a key in the current section.
    VX_EXPORT void WriteStringEntry(const char *iKey, const char *iValue);

    /// @brief Reads a string value from a key in the current section.
    VX_EXPORT int ReadStringEntry(char *iKey, char *oData);

private:
    void *m_VirtoolsSection; ///< A handle or key to the main "Virtools" section, likely in the system registry.
    void *m_CurrentSection;  ///< A handle or key to the currently open section.
};

#endif // VXCONFIGURATION_H
