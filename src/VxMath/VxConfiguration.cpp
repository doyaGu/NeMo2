#include "VxConfiguration.h"

#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

/// @brief Removes leading and trailing whitespace from a string.
char *Shrink(char *str);

VxConfiguration::VxConfiguration(unsigned short indent) : m_Root(NULL), m_DefaultRoot(NULL), m_Indent(indent) {
    m_Root = new VxConfigurationSection("root", NULL);
}

VxConfiguration::~VxConfiguration() {
    Clear();
    ClearDefault();
    delete m_Root;
    m_Root = NULL;
}

void VxConfiguration::Clear() { if (m_Root) m_Root->Clear(); }

void VxConfiguration::ClearDefault() {
    if (m_DefaultRoot) {
        m_DefaultRoot->Clear();
        delete m_DefaultRoot;
        m_DefaultRoot = NULL;
    }
}

size_t VxConfiguration::GetNumberOfSubSections() const {
    return m_Root ? m_Root->GetNumberOfSubSections() : 0;
}

size_t VxConfiguration::GetNumberOfEntries() const {
    return m_Root ? m_Root->GetNumberOfEntries() : 0;
}

size_t VxConfiguration::GetNumberOfSubSectionsRecursive() const {
    return m_Root ? m_Root->GetNumberOfSubSectionsRecursive() : 0;
}

size_t VxConfiguration::GetNumberOfEntriesRecursive() const {
    return m_Root ? m_Root->GetNumberOfEntriesRecursive() : 0;
}

XBOOL VxConfiguration::AddEntry(const char *parent, const char *ename, const char *evalue, VxConfigurationEntry **result) {
    if (!ename) return FALSE;
    
    if (!parent) {
        if (m_Root) {
            m_Root->AddEntry(ename, evalue, result);
            return TRUE;
        }
        return FALSE;
    }

    VxConfigurationSection *section = GetSubSection(parent, TRUE);
    if (section != NULL || (section = CreateSubSection(m_Root, parent, TRUE)) != NULL) {
        section->AddEntry(ename, evalue, result);
        return TRUE;
    }
    return FALSE;
}

XBOOL VxConfiguration::AddEntry(const char *parent, const char *ename, int evalue, VxConfigurationEntry **result) {
    if (!ename) return FALSE;
    
    if (!parent) {
        if (m_Root) {
            m_Root->AddEntry(ename, evalue, result);
            return TRUE;
        }
        return FALSE;
    }

    VxConfigurationSection *section = GetSubSection(parent, TRUE);
    if (section != NULL || (section = CreateSubSection(m_Root, parent, TRUE)) != NULL) {
        section->AddEntry(ename, evalue, result);
        return TRUE;
    }
    return FALSE;
}

XBOOL VxConfiguration::AddEntry(const char *parent, const char *ename, long evalue, VxConfigurationEntry **result) {
    if (!ename) return FALSE;
    
    if (!parent) {
        if (m_Root) {
            m_Root->AddEntry(ename, evalue, result);
            return TRUE;
        }
        return FALSE;
    }

    VxConfigurationSection *section = GetSubSection(parent, TRUE);
    if (section != NULL || (section = CreateSubSection(m_Root, parent, TRUE)) != NULL) {
        section->AddEntry(ename, evalue, result);
        return TRUE;
    }
    return FALSE;
}

XBOOL VxConfiguration::AddEntry(const char *parent, const char *ename, unsigned int evalue, VxConfigurationEntry **result) {
    if (!ename) return FALSE;
    
    if (!parent) {
        if (m_Root) {
            m_Root->AddEntry(ename, evalue, result);
            return TRUE;
        }
        return FALSE;
    }

    VxConfigurationSection *section = GetSubSection(parent, TRUE);
    if (section != NULL || (section = CreateSubSection(m_Root, parent, TRUE)) != NULL) {
        section->AddEntry(ename, evalue, result);
        return TRUE;
    }
    return FALSE;
}

XBOOL VxConfiguration::AddEntry(const char *parent, const char *ename, unsigned long evalue, VxConfigurationEntry **result) {
    if (!ename) return FALSE;
    
    if (!parent) {
        if (m_Root) {
            m_Root->AddEntry(ename, evalue, result);
            return TRUE;
        }
        return FALSE;
    }

    VxConfigurationSection *section = GetSubSection(parent, TRUE);
    if (section != NULL || (section = CreateSubSection(m_Root, parent, TRUE)) != NULL) {
        section->AddEntry(ename, evalue, result);
        return TRUE;
    }
    return FALSE;
}

XBOOL VxConfiguration::AddEntry(const char *parent, const char *ename, float evalue, VxConfigurationEntry **result) {
    if (!ename) return FALSE;
    
    if (!parent) {
        if (m_Root) {
            m_Root->AddEntry(ename, evalue, result);
            return TRUE;
        }
        return FALSE;
    }

    VxConfigurationSection *parentSection = GetSubSection(parent, TRUE);
    if (parentSection || (parentSection = CreateSubSection(m_Root, parent, TRUE)) != NULL) {
        parentSection->AddEntry(ename, evalue, result);
        return TRUE;
    }
    return FALSE;
}

VxConfigurationSection *VxConfiguration::CreateSubSection(const char *parent, const char *sname) {
    if (!sname) return NULL;
    
    if (!parent && m_Root)
        return m_Root->CreateSubSection(sname);

    if (m_Root) {
        VxConfigurationSection *parentSection = m_Root->GetSubSection(parent);
        if (parentSection)
            return parentSection->CreateSubSection(sname);
    }
    return NULL;
}

XBOOL VxConfiguration::DeleteEntry(const char *parent, const char *ename) {
    if (!parent && m_Root)
        return m_Root->DeleteEntry(ename);

    if (m_Root) {
        VxConfigurationSection *parentSection = m_Root->GetSubSection(parent);
        if (parentSection)
            return parentSection->DeleteEntry(ename);
    }
    return FALSE;
}

XBOOL VxConfiguration::DeleteSection(const char *parent, const char *sname) {
    if (!parent && m_Root)
        return m_Root->DeleteSection(sname);

    if (m_Root) {
        VxConfigurationSection *parentSection = m_Root->GetSubSection(parent);
        if (parentSection)
            return parentSection->DeleteSection(sname);
    }
    return FALSE;
}

VxConfigurationEntry *VxConfiguration::RemoveEntry(const char *parent, const char *ename) {
    if (!parent && m_Root)
        return m_Root->RemoveEntry(ename);

    if (m_Root) {
        VxConfigurationSection *parentSection = m_Root->GetSubSection(parent);
        if (parentSection)
            return parentSection->RemoveEntry(ename);
    }
    return NULL;
}

VxConfigurationSection *VxConfiguration::RemoveSection(const char *parent, const char *sname) {
    if (!parent && m_Root)
        return m_Root->RemoveSection(sname);

    if (m_Root) {
        VxConfigurationSection *parentSection = m_Root->GetSubSection(parent);
        if (parentSection)
            return parentSection->RemoveSection(sname);
    }
    return NULL;
}

XBOOL VxConfiguration::AddDefaultEntry(const char *parent, const char *ename, const char *evalue) {
    if (!m_DefaultRoot)
        m_DefaultRoot = new VxConfigurationSection("default", NULL);

    if (!parent) {
        m_DefaultRoot->AddEntry(ename, evalue);
        return TRUE;
    }

    VxConfigurationSection *parentSection = GetSubSection(m_DefaultRoot, parent, TRUE);
    if (parentSection || (parentSection = CreateSubSection(m_DefaultRoot, parent, TRUE)) != NULL) {
        parentSection->AddEntry(ename, evalue, NULL);
        return TRUE;
    }

    return FALSE;
}

XBOOL VxConfiguration::AddDefaultEntry(const char *parent, const char *ename, int evalue) {
    if (!m_DefaultRoot)
        m_DefaultRoot = new VxConfigurationSection("default", NULL);

    if (!parent) {
        m_DefaultRoot->AddEntry(ename, evalue);
        return TRUE;
    }

    VxConfigurationSection *parentSection = GetSubSection(m_DefaultRoot, parent, TRUE);
    if (parentSection || (parentSection = CreateSubSection(m_DefaultRoot, parent, TRUE)) != NULL) {
        parentSection->AddEntry(ename, evalue, NULL);
        return TRUE;
    }

    return FALSE;
}

XBOOL VxConfiguration::AddDefaultEntry(const char *parent, const char *ename, long evalue) {
    if (!m_DefaultRoot)
        m_DefaultRoot = new VxConfigurationSection("default", NULL);

    if (!parent) {
        m_DefaultRoot->AddEntry(ename, evalue);
        return TRUE;
    }

    VxConfigurationSection *parentSection = GetSubSection(m_DefaultRoot, parent, TRUE);
    if (parentSection || (parentSection = CreateSubSection(m_DefaultRoot, parent, TRUE)) != NULL) {
        parentSection->AddEntry(ename, evalue, NULL);
        return TRUE;
    }

    return FALSE;
}

XBOOL VxConfiguration::AddDefaultEntry(const char *parent, const char *ename, unsigned int evalue) {
    if (!m_DefaultRoot)
        m_DefaultRoot = new VxConfigurationSection("default", NULL);

    if (!parent) {
        m_DefaultRoot->AddEntry(ename, evalue);
        return TRUE;
    }

    VxConfigurationSection *parentSection = GetSubSection(m_DefaultRoot, parent, TRUE);
    if (parentSection || (parentSection = CreateSubSection(m_DefaultRoot, parent, TRUE)) != NULL) {
        parentSection->AddEntry(ename, evalue, NULL);
        return TRUE;
    }

    return FALSE;
}

XBOOL VxConfiguration::AddDefaultEntry(const char *parent, const char *ename, unsigned long evalue) {
    if (!m_DefaultRoot)
        m_DefaultRoot = new VxConfigurationSection("default", NULL);

    if (!parent) {
        m_DefaultRoot->AddEntry(ename, evalue);
        return TRUE;
    }

    VxConfigurationSection *parentSection = GetSubSection(m_DefaultRoot, parent, TRUE);
    if (parentSection || (parentSection = CreateSubSection(m_DefaultRoot, parent, TRUE)) != NULL) {
        parentSection->AddEntry(ename, evalue, NULL);
        return TRUE;
    }

    return FALSE;
}

XBOOL VxConfiguration::AddDefaultEntry(const char *parent, const char *ename, float evalue) {
    if (!m_DefaultRoot)
        m_DefaultRoot = new VxConfigurationSection("default", NULL);

    if (!parent) {
        m_DefaultRoot->AddEntry(ename, evalue);
        return TRUE;
    }

    VxConfigurationSection *parentSection = GetSubSection(m_DefaultRoot, parent, TRUE);
    if (parentSection || (parentSection = CreateSubSection(m_DefaultRoot, parent, TRUE)) != NULL) {
        parentSection->AddEntry(ename, evalue, NULL);
        return TRUE;
    }

    return FALSE;
}

VxConfigurationSection *VxConfiguration::CreateDefaultSubSection(const char *parent, const char *sname) {
    if (!parent) {
        if (!m_DefaultRoot)
            m_DefaultRoot = new VxConfigurationSection("default", NULL);
        return m_DefaultRoot->CreateSubSection(sname);
    }

    if (m_DefaultRoot) {
        VxConfigurationSection *parentSection = m_DefaultRoot->GetSubSection(parent);
        if (parentSection)
            return parentSection->CreateSubSection(sname);
    }

    return NULL;
}

ConstSectionIt VxConfiguration::BeginSections() const {
    return m_Root ? m_Root->BeginChildSection() : ConstSectionIt();
}

ConstEntryIt VxConfiguration::BeginEntries() const {
    return m_Root ? m_Root->BeginChildEntry() : ConstEntryIt();
}

VxConfigurationSection *VxConfiguration::GetNextSection(ConstSectionIt &it) const {
    return m_Root ? m_Root->GetNextChildSection(it) : NULL;
}

VxConfigurationEntry *VxConfiguration::GetNextEntry(ConstEntryIt &it) const {
    return m_Root ? m_Root->GetNextChildEntry(it) : NULL;
}

VxConfigurationSection *VxConfiguration::GetSubSection(const char *sname, XBOOL usedot) const {
    if (!sname) return NULL;
    
    VxConfigurationSection *section = NULL;

    if (m_Root)
        section = GetSubSection(m_Root, sname, usedot);

    if (!section && m_DefaultRoot)
        return GetSubSection(m_DefaultRoot, sname, usedot);

    return section;
}

VxConfigurationEntry *VxConfiguration::GetEntry(const char *ename, XBOOL usedot) const {
    if (!ename) return NULL;
    
    if (!usedot) {
        VxConfigurationEntry *entry = m_Root ? m_Root->GetEntry(ename) : NULL;
        if (!entry && m_DefaultRoot)
            entry = m_DefaultRoot->GetEntry(ename);
        return entry;
    }

    XString entryName(ename);
    if (entryName.Length() == 0)
        entryName = "";
    size_t dotPos = entryName.RFind('.');
    if (dotPos != XString::NOTFOUND) {
        entryName[dotPos] = '\0';
        VxConfigurationSection *section = m_Root ? GetSubSection(m_Root, entryName.Str(), TRUE) : NULL;
        entryName[dotPos] = '.';
        if (section) {
            VxConfigurationEntry *entry = section->GetEntry(&entryName[dotPos + 1]);
            if (entry) return entry;
        }

        if (m_DefaultRoot) {
            entryName[dotPos] = '\0';
            section = GetSubSection(m_DefaultRoot, entryName.Str(), TRUE);
            entryName[dotPos] = '.';
            if (section)
                return section->GetEntry(&entryName[dotPos + 1]);
        }
    }

    return NULL;
}

XBOOL VxConfiguration::BuildFromDataFile(const char *name, XString &error) {
    int line = 0;
    return BuildFromFile(name, line, error);
}

XBOOL VxConfiguration::BuildFromFile(const char *name, int &cline, XString &error) {
    FILE *file = fopen(name, "rb");
    if (!file) {
        error = "Could not open file: ";
        error += name;
        return FALSE;
    }

    // Read file contents into memory
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = new char[fileSize + 1];
    if (!buffer) {
        fclose(file);
        error = "Out of memory";
        return FALSE;
    }

    size_t read = fread(buffer, 1, fileSize, file);
    fclose(file);

    if (read != (size_t)fileSize) {
        delete[] buffer;
        error = "Failed to read entire file";
        return FALSE;
    }

    buffer[read] = '\0';

    XBOOL result = BuildFromMemory(buffer, cline, error);

    delete[] buffer;
    return result;
}

XBOOL VxConfiguration::BuildFromMemory(const char *buffer, int &cline, XString &error) {
    if (!buffer || !m_Root) return FALSE;

    Clear();

    char *bufferCopy = new char[strlen(buffer) + 1];
    strcpy(bufferCopy, buffer);

    VxConfigurationSection *currentSection = m_Root;
    char *line = strtok(bufferCopy, "\n\r");
    cline = 1;

    while (line) {
        // Skip comments and empty lines
        char *trimmed = Shrink(line);
        if (trimmed && *trimmed && *trimmed != '#' && *trimmed != ';') {
            if (*trimmed == '<') {
                // This is a section start or end tag
                if (!ManageSection(trimmed, &currentSection, error)) {
                    delete[] bufferCopy;
                    return FALSE;
                }
            } else {
                // This is an entry
                if (!ManageEntry(trimmed, currentSection, error)) {
                    delete[] bufferCopy;
                    return FALSE;
                }
            }
        }

        line = strtok(NULL, "\n\r");
        ++cline;
    }

    delete[] bufferCopy;
    return TRUE;
}

XBOOL VxConfiguration::SaveToDataFile(const char *name) {
    return SaveToFile(name);
}

XBOOL VxConfiguration::SaveToFile(const char *name) {
    if (!m_Root || !name) return FALSE;

    FILE *file = fopen(name, "wb");
    if (!file) return FALSE;

    // Helper function to build full section path
    struct SectionHelper {
        static XString GetFullPath(VxConfigurationSection *section) {
            if (!section || !section->GetParent()) {
                return XString(section ? section->GetName() : "");
            }

            XString parentPath = GetFullPath(section->GetParent());
            if (parentPath.Length() == 0 || strcmp(parentPath.CStr(), "root") == 0) {
                return XString(section->GetName());
            } else {
                return parentPath + "." + section->GetName();
            }
        }
    };

    // Lambda-like function to write indentation
    struct Indent {
        static void Write(FILE *f, int level, unsigned short indentSize) {
            for (int i = 0; i < level * indentSize; i++)
                fprintf(f, " ");
        }
    };

    // Lambda-like function to recursively write sections
    struct WriteSection {
        static bool DoWrite(FILE *f, VxConfigurationSection *section, int level, unsigned short indentSize) {
            if (!section) return true;

            // Write section header using XML-like tags
            if (level > 0) {
                // Don't write header for root
                fprintf(f, "<%s>\n", section->GetName());
            }

            // Write entries with proper indentation
            ConstEntryIt entIt = section->BeginChildEntry();
            VxConfigurationEntry *entry;
            while ((entry = section->GetNextChildEntry(entIt)) != NULL) {
                if (level > 0) Indent::Write(f, 1, indentSize);
                fprintf(f, "%s = %s\n", entry->GetName(), entry->GetValue());
            }

            // Write subsections
            ConstSectionIt secIt = section->BeginChildSection();
            VxConfigurationSection *subSection;
            while ((subSection = section->GetNextChildSection(secIt)) != NULL) {
                if (level == 0 && (section->GetNumberOfEntries() > 0 || section->GetNumberOfSubSections() > 1))
                    fprintf(f, "\n");
                if (!DoWrite(f, subSection, level + 1, indentSize))
                    return false;
            }

            // Write closing tag for non-root sections
            if (level > 0) {
                fprintf(f, "</%s>\n", section->GetName());
            }

            return true;
        }
    };

    bool success = WriteSection::DoWrite(file, m_Root, 0, m_Indent);

    fclose(file);
    return success ? TRUE : FALSE;
}

VxConfigurationSection *VxConfiguration::CreateSubSection(VxConfigurationSection *root, const char *sname, XBOOL usedot) const {
    if (!root || !sname) return NULL;

    if (!usedot) return root->CreateSubSection(sname);

    // Handle dot notation for hierarchical sections
    XString sectionPath(sname);
    size_t dotPos = sectionPath.Find('.');

    if (dotPos == XString::NOTFOUND) return root->CreateSubSection(sname);

    // Extract the first section name
    sectionPath[dotPos] = '\0';
    char *firstSection = sectionPath.Str();

    // Get or create the first section
    VxConfigurationSection *section = root->GetSubSection(firstSection);
    if (!section) section = root->CreateSubSection(firstSection);

    if (!section) return NULL;

    // Process remaining path recursively
    return CreateSubSection(section, &sectionPath[dotPos + 1], TRUE);
}

VxConfigurationSection *VxConfiguration::GetSubSection(VxConfigurationSection *root, const char *sname, XBOOL usedot) const {
    if (!root || !sname) return NULL;

    if (!usedot) return root->GetSubSection(sname);

    // Handle dot notation for hierarchical sections
    XString sectionPath(sname);
    size_t dotPos = sectionPath.Find('.');
    if (dotPos == XString::NOTFOUND) return root->GetSubSection(sname);

    // Extract the first section name
    sectionPath[dotPos] = '\0';
    const char *firstSection = sectionPath.CStr();

    // Get the first section
    VxConfigurationSection *section = root->GetSubSection(firstSection);
    if (!section) return NULL;

    // Process remaining path recursively
    return GetSubSection(section, &sectionPath[dotPos + 1], TRUE);
}

XBOOL VxConfiguration::ManageSection(const char *line, VxConfigurationSection **current, XString &error) {
    if (!line || !current || !m_Root) return FALSE;

    char *lineCopy = new char[strlen(line) + 1];
    strcpy(lineCopy, line);

    // Remove leading and trailing spaces
    lineCopy = Shrink(lineCopy);

    // Check for closing tag format </SectionName>
    size_t len = strlen(lineCopy);
    if (len > 3 && lineCopy[0] == '<' && lineCopy[1] == '/' && lineCopy[len - 1] == '>') {
        // This is a closing tag, return to parent section
        if (*current && (*current)->GetParent()) {
            *current = (*current)->GetParent();
        } else {
            *current = m_Root;
        }
        delete[] lineCopy;
        return TRUE;
    }

    // Check for opening tag format <SectionName>
    if (len < 3 || lineCopy[0] != '<' || lineCopy[len - 1] != '>') {
        error = "Invalid section format: ";
        error += lineCopy;
        delete[] lineCopy;
        return FALSE;
    }

    XString sectionNameStr(line + 1, len - 2);  // Skip brackets, copy substring
    const char *sectionName = sectionNameStr.CStr();

    // Handle dot notation in sections
    XString section(sectionName);
    size_t dotPos = section.Find('.');
    if (dotPos == XString::NOTFOUND) {
        // Simple section name
        VxConfigurationSection *existingSection = (*current)->GetSubSection(sectionName);
        if (existingSection) {
            *current = existingSection;
        } else {
            *current = (*current)->CreateSubSection(sectionName);
            if (!*current) {
                error = "Failed to create section: ";
                error += sectionName;
                delete[] lineCopy;
                return FALSE;
            }
        }
    } else {
        // Hierarchical section name
        *current = GetSubSection(sectionName, TRUE);
        if (!*current) {
            *current = CreateSubSection(m_Root, sectionName, TRUE);
            if (!*current) {
                error = "Failed to create hierarchical section: ";
                error += sectionName;
                delete[] lineCopy;
                return FALSE;
            }
        }
    }

    delete[] lineCopy;
    return TRUE;
}

XBOOL VxConfiguration::ManageEntry(const char *line, VxConfigurationSection *current, XString &error) {
    if (!line || !current) return FALSE;

    char *lineCopy = new char[strlen(line) + 1];
    strcpy(lineCopy, line);

    // Remove leading and trailing spaces
    lineCopy = Shrink(lineCopy);

    // Find key-value separator
    char *separator = strchr(lineCopy, '=');
    if (!separator) {
        error = "Invalid entry format (missing '='): ";
        error += lineCopy;
        delete[] lineCopy;
        return FALSE;
    }

    // Split line into key and value
    *separator = '\0';
    char *key = Shrink(lineCopy);
    char *value = Shrink(separator + 1);

    if (!key || !*key) {
        error = "Empty key in entry: ";
        error += lineCopy;
        delete[] lineCopy;
        return FALSE;
    }

    // Add the entry to the current section
    current->AddEntry(key, value ? value : "");
    delete[] lineCopy;
    return TRUE;
}

// VxConfigurationSection implementation

VxConfigurationSection::~VxConfigurationSection() { Clear(); }

void VxConfigurationSection::Clear() {
    // Collect all entries first, then delete them
    XArray<VxConfigurationEntry*> entriesToDelete;
    EntryIt entIt = m_Entries.Begin();
    while (entIt != m_Entries.End()) {
        VxConfigurationEntry *entry = *entIt;
        if (entry) entriesToDelete.PushBack(entry);
        ++entIt;
    }
    m_Entries.Clear();

    // Now safely delete all entries
    for (size_t i = 0; i < entriesToDelete.Size(); i++) {
        delete entriesToDelete[i];
    }

    // Same for subsections
    XArray<VxConfigurationSection*> sectionsToDelete;
    SectionIt secIt = m_SubSections.Begin();
    while (secIt != m_SubSections.End()) {
        VxConfigurationSection *section = *secIt;
        if (section) {
            section->Clear();
            sectionsToDelete.PushBack(section);
        }
        ++secIt;
    }
    m_SubSections.Clear();

    // Now safely delete all sections
    for (size_t i = 0; i < sectionsToDelete.Size(); i++) {
        delete sectionsToDelete[i];
    }
}

size_t VxConfigurationSection::GetNumberOfSubSections() const { return m_SubSections.Size(); }

size_t VxConfigurationSection::GetNumberOfEntries() const { return m_Entries.Size();}

size_t VxConfigurationSection::GetNumberOfSubSectionsRecursive() const {
    size_t count = m_SubSections.Size();

    ConstSectionIt it = BeginChildSection();
    VxConfigurationSection *section;
    while ((section = GetNextChildSection(it)) != NULL) {
        count += section->GetNumberOfSubSectionsRecursive();
    }

    return count;
}

size_t VxConfigurationSection::GetNumberOfEntriesRecursive() const {
    size_t count = m_Entries.Size();

    ConstSectionIt it = BeginChildSection();
    VxConfigurationSection *section;
    while ((section = GetNextChildSection(it)) != NULL) {
        count += section->GetNumberOfEntriesRecursive();
    }

    return count;
}

void VxConfigurationSection::AddEntry(const char *ename, const char *evalue, VxConfigurationEntry **result) {
    if (!ename) return;

    VxConfigurationEntry *entry = GetEntry(ename);
    if (entry) {
        entry->SetValue(evalue);
        if (result) *result = entry;
        return;
    }

    entry = new VxConfigurationEntry(this, ename, evalue ? evalue : "");
    m_Entries.Insert(ename, entry);

    if (result)
        *result = entry;
}

void VxConfigurationSection::AddEntry(const char *ename, int evalue, VxConfigurationEntry **result) {
    if (!ename) return;

    VxConfigurationEntry *entry = GetEntry(ename);
    if (entry) {
        entry->SetValue(evalue);
        if (result) *result = entry; return;
    }

    entry = new VxConfigurationEntry(this, ename, evalue);
    m_Entries.Insert(ename, entry);

    if (result)
        *result = entry;
}

void VxConfigurationSection::AddEntry(const char *ename, long evalue, VxConfigurationEntry **result) {
    if (!ename) return;

    VxConfigurationEntry *entry = GetEntry(ename);
    if (entry) {
        entry->SetValue(evalue);
        if (result)
            *result = entry; return;
    }

    entry = new VxConfigurationEntry(this, ename, evalue);
    m_Entries.Insert(ename, entry);

    if (result)
        *result = entry;
}

void VxConfigurationSection::AddEntry(const char *ename, unsigned int evalue, VxConfigurationEntry **result) {
    if (!ename) return;

    VxConfigurationEntry *entry = GetEntry(ename);
    if (entry) {
        entry->SetValue(evalue);
        if (result) *result = entry;
        return;
    }

    entry = new VxConfigurationEntry(this, ename, evalue);
    m_Entries.Insert(ename, entry);

    if (result)
        *result = entry;
}

void VxConfigurationSection::AddEntry(const char *ename, unsigned long evalue, VxConfigurationEntry **result) {
    if (!ename) return;

    VxConfigurationEntry *entry = GetEntry(ename);
    if (entry) {
        entry->SetValue(evalue);
        if (result) *result = entry;
        return;
    }

    entry = new VxConfigurationEntry(this, ename, evalue);
    m_Entries.Insert(ename, entry);

    if (result)
        *result = entry;
}

void VxConfigurationSection::AddEntry(const char *ename, float evalue, VxConfigurationEntry **result) {
    if (!ename) return;

    VxConfigurationEntry *entry = GetEntry(ename);
    if (entry) {
        entry->SetValue(evalue);
        if (result) *result = entry;
        return;
    }

    entry = new VxConfigurationEntry(this, ename, evalue);
    m_Entries.Insert(ename, entry);

    if (result)
        *result = entry;
}

VxConfigurationSection *VxConfigurationSection::CreateSubSection(const char *sname) {
    if (!sname) return NULL;

    VxConfigurationSection *section = GetSubSection(sname);
    if (section) return section;

    section = new VxConfigurationSection(sname, this);
    m_SubSections.Insert(sname, section);

    return section;
}

XBOOL VxConfigurationSection::DeleteEntry(const char *ename) {
    if (!ename) return FALSE;

    VxConfigurationEntry *entry = GetEntry(ename);
    if (!entry) return FALSE;

    delete entry;
    m_Entries.Remove(ename);

    return TRUE;
}

XBOOL VxConfigurationSection::DeleteSection(const char *sname) {
    if (!sname) return FALSE;

    VxConfigurationSection *section = GetSubSection(sname);
    if (!section) return FALSE;

    section->Clear();
    delete section;
    m_SubSections.Remove(sname);

    return TRUE;
}

VxConfigurationEntry *VxConfigurationSection::RemoveEntry(const char *ename) {
    if (!ename) return NULL;

    VxConfigurationEntry *entry = GetEntry(ename);
    if (!entry) return NULL;

    // Save the entry data before removing it
    XString name = entry->GetName();
    XString value = entry->GetValue();

    // Remove and delete the original entry
    delete entry;
    m_Entries.Remove(ename);

    // Create a new entry with the saved data (parent set to NULL since it's being removed)
    VxConfigurationEntry *newEntry = new VxConfigurationEntry(nullptr, name.CStr(), value.CStr());

    return newEntry;
}

// Helper function to recursively copy a section
static void CopySection(VxConfigurationSection *source, VxConfigurationSection *dest) {
    if (!source || !dest) return;

    // Copy all entries
    ConstEntryIt entIt = source->BeginChildEntry();
    VxConfigurationEntry *entry;
    while ((entry = source->GetNextChildEntry(entIt)) != nullptr) {
        dest->AddEntry(const_cast<char*>(entry->GetName()), entry->GetValue());
    }

    // Copy all subsections recursively
    ConstSectionIt secIt = source->BeginChildSection();
    VxConfigurationSection *subSection;
    while ((subSection = source->GetNextChildSection(secIt)) != nullptr) {
        VxConfigurationSection *newSubSection = dest->CreateSubSection(const_cast<char*>(subSection->GetName()));
        if (newSubSection) {
            CopySection(subSection, newSubSection);
        }
    }
}

VxConfigurationSection *VxConfigurationSection::RemoveSection(const char *sname) {
    if (!sname) return NULL;

    VxConfigurationSection *section = GetSubSection(sname);
    if (!section) return NULL;

    // Create a copy of the section (parent set to NULL since it's being removed)
    VxConfigurationSection *copy = new VxConfigurationSection(sname, nullptr);
    CopySection(section, copy);

    // Delete the original section
    section->Clear();
    delete section;
    m_SubSections.Remove(sname);

    return copy;
}

ConstEntryIt VxConfigurationSection::BeginChildEntry() const { return m_Entries.Begin(); }

VxConfigurationEntry *VxConfigurationSection::GetNextChildEntry(ConstEntryIt &it) const {
    if (it == m_Entries.End()) return NULL;

    VxConfigurationEntry *entry = *it;
    ++it;
    return entry;
}

ConstSectionIt VxConfigurationSection::BeginChildSection() const { return m_SubSections.Begin(); }

VxConfigurationSection *VxConfigurationSection::GetNextChildSection(ConstSectionIt &it) const {
    if (it == m_SubSections.End()) return NULL;

    VxConfigurationSection *section = *it;
    ++it;
    return section;
}

VxConfigurationEntry *VxConfigurationSection::GetEntry(const char *ename) const {
    if (!ename) return NULL;

    ConstEntryIt it = m_Entries.Find(ename);
    if (it == m_Entries.End()) return NULL;
    return *it;
}

VxConfigurationSection *VxConfigurationSection::GetSubSection(const char *sname) const {
    if (!sname) return NULL;

    ConstSectionIt it = m_SubSections.Find(sname);
    if (it == m_SubSections.End()) return NULL;
    return *it;
}

const char *VxConfigurationSection::GetName() const { return m_Name.CStr(); }

VxConfigurationSection *VxConfigurationSection::GetParent() const { return m_Parent; }

VxConfigurationSection::VxConfigurationSection(const char *name, VxConfigurationSection *parent)
    : m_Parent(parent), m_Name(name ? name : "") {}

// VxConfigurationEntry implementation
VxConfigurationEntry::~VxConfigurationEntry() {}

void VxConfigurationEntry::SetValue(const char *value) { m_Value = value ? value : ""; }

void VxConfigurationEntry::SetValue(int value) {
    char buffer[32];
    sprintf(buffer, "%d", value);
    m_Value = buffer;
}

void VxConfigurationEntry::SetValue(long value) {
    char buffer[32];
    sprintf(buffer, "%ld", value);
    m_Value = buffer;
}

void VxConfigurationEntry::SetValue(unsigned int value) {
    char buffer[32];
    sprintf(buffer, "%u", value);
    m_Value = buffer;
}

void VxConfigurationEntry::SetValue(unsigned long value) {
    char buffer[32];
    sprintf(buffer, "%lu", value);
    m_Value = buffer;
}

void VxConfigurationEntry::SetValue(float value) {
    char buffer[32];
    sprintf(buffer, "%f", value);
    m_Value = buffer;
}

const char *VxConfigurationEntry::GetName() const { return m_Name.CStr(); }

VxConfigurationSection *VxConfigurationEntry::GetParent() const { return m_Parent; }

const char *VxConfigurationEntry::GetValue() const { return m_Value.CStr(); }

XBOOL VxConfigurationEntry::GetValueAsInteger(int &value) const {
    return sscanf(m_Value.CStr(), "%d", &value) == 1;
}

XBOOL VxConfigurationEntry::GetValueAsLong(long &value) const {
    return sscanf(m_Value.CStr(), "%ld", &value) == 1;
}

XBOOL VxConfigurationEntry::GetValueAsUnsignedInteger(unsigned int &value) const {
    return sscanf(m_Value.CStr(), "%u", &value) == 1;
}

XBOOL VxConfigurationEntry::GetValueAsUnsignedLong(unsigned long &value) const {
    return sscanf(m_Value.CStr(), "%lu", &value) == 1;
}

XBOOL VxConfigurationEntry::GetValueAsFloat(float &value) const {
    return sscanf(m_Value.CStr(), "%f", &value) == 1;
}

VxConfigurationEntry::VxConfigurationEntry(VxConfigurationSection *parent, const char *name, const char *value)
    : m_Name(name ? name : ""), m_Parent(parent) { SetValue(value); }

VxConfigurationEntry::VxConfigurationEntry(VxConfigurationSection *parent, const char *name, int value)
    : m_Name(name ? name : ""), m_Parent(parent) { SetValue(value); }

VxConfigurationEntry::VxConfigurationEntry(VxConfigurationSection *parent, const char *name, long value)
    : m_Name(name ? name : ""), m_Parent(parent) { SetValue(value); }

VxConfigurationEntry::VxConfigurationEntry(VxConfigurationSection *parent, const char *name, unsigned int value)
    : m_Name(name ? name : ""), m_Parent(parent) { SetValue(value); }

VxConfigurationEntry::VxConfigurationEntry(VxConfigurationSection *parent, const char *name, unsigned long value)
    : m_Name(name ? name : ""), m_Parent(parent) { SetValue(value);}

VxConfigurationEntry::VxConfigurationEntry(VxConfigurationSection *parent, const char *name, float value)
    : m_Name(name ? name : ""), m_Parent(parent) { SetValue(value); }

char *Shrink(char *str) {
    if (!str) return NULL;

    // Skip leading whitespace
    while (*str && (*str == ' ' || *str == '\t')) str++;

    // Empty string
    if (!*str) return str;

    // Find end of string
    char *end = str + strlen(str) - 1;

    // Remove trailing whitespace
    while (end > str && (*end == ' ' || *end == '\t')) *end-- = '\0';

    return str;
}

// VxConfig implementation
#if defined(_WIN32)
VxConfig::VxConfig() : m_VirtoolsSection(NULL), m_CurrentSection(NULL) {
    LONG result = ::RegCreateKeyExA(HKEY_CURRENT_USER, "Software\\Virtools\\UserConfig", 0, NULL, 0, KEY_WRITE, NULL, (PHKEY) &m_VirtoolsSection, NULL);
    if (result != ERROR_SUCCESS) {
        m_VirtoolsSection = NULL;
    }
}

VxConfig::~VxConfig() {
    if (m_VirtoolsSection) ::RegCloseKey(*(PHKEY) &m_VirtoolsSection);
    if (m_CurrentSection) ::RegCloseKey(*(PHKEY) &m_CurrentSection);
}

void VxConfig::OpenSection(const char *iSection, VxConfig::Mode iOpeningMode) {
    if (m_CurrentSection) ::RegCloseKey(*(PHKEY) &m_CurrentSection);
    ::RegCreateKeyExA(*(PHKEY) &m_VirtoolsSection, iSection, 0, NULL, 0, iOpeningMode, NULL, (PHKEY) &m_CurrentSection, NULL);
}

void VxConfig::CloseSection(const char *iSection) {
    if (m_CurrentSection) {
        ::RegCloseKey(*(PHKEY) &m_CurrentSection);
        m_CurrentSection = NULL;
    }
}

void VxConfig::WriteStringEntry(const char *iKey, const char *iValue) {
    if (m_CurrentSection && iKey && iValue)
        ::RegSetValueExA(*(PHKEY) &m_CurrentSection, iKey, 0, REG_SZ, (LPBYTE) iValue, static_cast<DWORD>(strlen(iValue) + 1));
}

XDWORD VxConfig::ReadStringEntry(const char *iKey, char *oData) {
    DWORD cbData = 256;
    if (m_CurrentSection && iKey && oData) {
        if (::RegQueryValueExA(*(PHKEY) &m_CurrentSection, iKey, NULL, NULL, (LPBYTE) oData, &cbData) == ERROR_SUCCESS)
            return cbData;
    }
    return static_cast<XDWORD>(-1);
}
#else
VxConfig::VxConfig() : m_VirtoolsSection(NULL), m_CurrentSection(NULL) {}

VxConfig::~VxConfig() {}

void VxConfig::OpenSection(const char *iSection, VxConfig::Mode iOpeningMode) {
    (void)iSection;
    (void)iOpeningMode;
}

void VxConfig::CloseSection(const char *iSection) {
    (void)iSection;
}

void VxConfig::WriteStringEntry(const char *iKey, const char *iValue) {
    (void)iKey;
    (void)iValue;
}

XDWORD VxConfig::ReadStringEntry(const char *iKey, char *oData) {
    (void)iKey;
    (void)oData;
    return static_cast<XDWORD>(-1);
}
#endif
