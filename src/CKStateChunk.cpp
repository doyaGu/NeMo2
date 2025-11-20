#include "CKStateChunk.h"

#include "VxMatrix.h"
#include "VxWindowFunctions.h"
#include "VxMath.h"
#include "CKFile.h"
#include "CKPluginManager.h"

#include <miniz.h>
#include <climits>

XObjectPointerArray CKStateChunk::m_TempXOPA;
XObjectArray CKStateChunk::m_TempXOA;

CKStateChunk *CreateCKStateChunk(CK_CLASSID id, CKFile *file) {
    return new CKStateChunk(id, file);
}

CKStateChunk *CreateCKStateChunk(CKStateChunk *chunk) {
    return new CKStateChunk(*chunk);
}

void DeleteCKStateChunk(CKStateChunk *chunk) {
    delete chunk;
}

CKStateChunk::CKStateChunk(CKStateChunk *chunk) {
    m_Data = nullptr;
    m_ChunkClassID = 0;
    m_ChunkSize = 0;
    m_ChunkParser = nullptr;
    m_Ids = nullptr;
    m_Chunks = nullptr;
    m_DataVersion = 0;
    m_ChunkVersion = 0;
    m_Managers = nullptr;
    m_File = nullptr;
    m_Dynamic = TRUE;
    Clone(chunk);
}

CKStateChunk::CKStateChunk(CK_CLASSID Cid, CKFile *f) {
    m_Data = nullptr;
    m_ChunkSize = 0;
    m_ChunkParser = nullptr;
    m_Ids = nullptr;
    m_Chunks = nullptr;
    m_Managers = nullptr;
    m_DataVersion = 0;
    m_ChunkVersion = 0;
    m_ChunkClassID = Cid;
    m_File = f;
    m_Dynamic = TRUE;
}

CKStateChunk::~CKStateChunk() {
    Clear();
}

void CKStateChunk::Clear() {
    m_ChunkClassID = 0;
    m_ChunkSize = 0;

    delete[] m_Data;
    delete m_ChunkParser;
    if (m_Ids)
        delete m_Ids;
    if (m_Chunks)
        delete m_Chunks;
    if (m_Managers)
        delete m_Managers;

    m_Managers = nullptr;
    m_Data = nullptr;
    m_Chunks = nullptr;
    m_ChunkParser = nullptr;
    m_Ids = nullptr;
}

void CKStateChunk::Clone(CKStateChunk *chunk) {
    if (!chunk)
        return;

    Clear();

    m_DataVersion = chunk->m_DataVersion;
    m_ChunkVersion = chunk->m_ChunkVersion;
    m_ChunkClassID = chunk->m_ChunkClassID;
    m_ChunkSize = chunk->m_ChunkSize;

    if (chunk->m_ChunkParser)
        m_ChunkParser = new ChunkParser(*chunk->m_ChunkParser);

    if (chunk->m_Ids)
        m_Ids = new IntListStruct(*chunk->m_Ids);

    if (chunk->m_Chunks)
        m_Chunks = new IntListStruct(*chunk->m_Chunks);

    if (chunk->m_Managers)
        m_Managers = new IntListStruct(*chunk->m_Managers);

    if (chunk->m_Data) {
        m_Data = new int[m_ChunkSize];
        memcpy(m_Data, chunk->m_Data, m_ChunkSize * sizeof(int));
        if (m_ChunkParser)
            m_ChunkParser->DataSize = m_ChunkSize;
    }
    m_File = chunk->m_File;
    m_Dynamic = chunk->m_Dynamic;
}

CK_CLASSID CKStateChunk::GetChunkClassID() {
    return m_ChunkClassID;
}

short int CKStateChunk::GetChunkVersion() {
    return m_ChunkVersion;
}

short int CKStateChunk::GetDataVersion() {
    return m_DataVersion;
}

void CKStateChunk::SetDataVersion(short int version) {
    m_DataVersion = version;
}

int CKStateChunk::GetDataSize() {
    return m_ChunkSize * (int) sizeof(int);
}

void CKStateChunk::StartRead() {
    if (m_ChunkParser) {
        m_ChunkParser->CurrentPos = 0;
        m_ChunkParser->PrevIdentifierPos = 0;
    } else {
        m_ChunkParser = new ChunkParser;
    }
    m_ChunkParser->DataSize = m_ChunkSize;
}

void CKStateChunk::StartWrite() {
    delete[] m_Data;
    m_Data = nullptr;
    m_ChunkVersion = CHUNK_VERSION4;
    if (m_ChunkParser) {
        m_ChunkParser->CurrentPos = 0;
        m_ChunkParser->PrevIdentifierPos = 0;
        m_ChunkParser->DataSize = 0;
    } else {
        m_ChunkParser = new ChunkParser;
    }
}

void CKStateChunk::CheckSize(int size) {
    if (!m_ChunkParser)
        return;

    int sz = size / sizeof(int);
    if (sz + m_ChunkParser->CurrentPos > m_ChunkParser->DataSize) {
        if (sz < 500)
            sz = 500;
        int *data = new int[m_ChunkParser->DataSize + sz];
        if (m_Data) {
            memcpy(data, m_Data, m_ChunkParser->CurrentPos * sizeof(int));
            delete[] m_Data;
        }
        m_Data = data;
        m_ChunkParser->DataSize = m_ChunkParser->DataSize + sz;
    }
}

void CKStateChunk::CloseChunk() {
    if (!m_ChunkParser)
        return;

    if (m_ChunkParser->CurrentPos > m_ChunkSize)
        m_ChunkSize = m_ChunkParser->CurrentPos;

    if (m_ChunkSize > 0) {
        if (m_ChunkSize + 10 < m_ChunkParser->DataSize) {
            int *data = new int[m_ChunkSize];
            if (m_Data) {
                memcpy(data, m_Data, m_ChunkSize * sizeof(int));
                delete[] m_Data;
            }
            m_Data = data;
        }
    } else {
        delete[] m_Data;
        m_Data = nullptr;
    }

    delete m_ChunkParser;
    m_ChunkParser = nullptr;

    if (m_Ids)
        m_Ids->Compact();
    if (m_Chunks)
        m_Chunks->Compact();
    if (m_Managers)
        m_Managers->Compact();
}

void CKStateChunk::UpdateDataSize() {
    if (m_ChunkParser) {
        if (m_ChunkParser->CurrentPos > m_ChunkSize)
            m_ChunkSize = m_ChunkParser->CurrentPos;
    }
}

void CKStateChunk::WriteIdentifier(CKDWORD id) {
    if (!m_ChunkParser)
        StartWrite();

    CheckSize(8);
    if (m_ChunkParser->PrevIdentifierPos < m_ChunkParser->CurrentPos)
        m_Data[m_ChunkParser->PrevIdentifierPos + 1] = m_ChunkParser->CurrentPos;
    m_ChunkParser->PrevIdentifierPos = m_ChunkParser->CurrentPos;
    m_Data[m_ChunkParser->CurrentPos++] = (int) id;
    m_Data[m_ChunkParser->CurrentPos++] = 0;
}

CKDWORD CKStateChunk::ReadIdentifier() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize)
        return 0;
    m_ChunkParser->PrevIdentifierPos = m_ChunkParser->CurrentPos;
    m_ChunkParser->CurrentPos += 2;
    return m_Data[m_ChunkParser->CurrentPos - 2];
}

CKBOOL CKStateChunk::SeekIdentifier(CKDWORD identifier) {
    if (!m_Data || m_ChunkSize == 0)
        return FALSE;

    if (!m_ChunkParser) {
        m_ChunkParser = new ChunkParser;
        m_ChunkParser->DataSize = m_ChunkSize;
    }

    int startPos = m_Data[m_ChunkParser->PrevIdentifierPos + 1];
    int currentPos = startPos;

    // Phase 1: Search from startPos to the end of the list
    if (currentPos != 0) {
        while (m_Data[currentPos] != identifier) {
            currentPos = m_Data[currentPos + 1];
            if (currentPos == 0)
                break;
        }

        if (currentPos != 0) {
            m_ChunkParser->PrevIdentifierPos = currentPos;
            m_ChunkParser->CurrentPos = currentPos + 2;
            return TRUE;
        }
    }

    // Phase 2: Search from the beginning of the list to startPos
    currentPos = 0;
    while (m_Data[currentPos] != identifier) {
        currentPos = m_Data[currentPos + 1];
        if (currentPos == startPos)
            return FALSE;
    }

    m_ChunkParser->PrevIdentifierPos = currentPos;
    m_ChunkParser->CurrentPos = currentPos + 2;

    return TRUE;
}

int CKStateChunk::SeekIdentifierAndReturnSize(CKDWORD identifier) {
    if (!m_Data || m_ChunkSize == 0)
        return -1;

    if (!m_ChunkParser) {
        m_ChunkParser = new ChunkParser;
        m_ChunkParser->DataSize = m_ChunkSize;
    }

    int i = 0;
    while (m_Data[i] != identifier) {
        i = m_Data[i + 1];
        if (i == 0)
            return -1;
    }
    m_ChunkParser->PrevIdentifierPos = i;
    int j = m_Data[i + 1];
    if (j == 0)
        j = m_ChunkSize;
    m_ChunkParser->CurrentPos = i + 2;
    return (j - i - 2) * sizeof(int);
}

void CKStateChunk::Skip(int DwordCount) {
    CheckSize(DwordCount * sizeof(int));
    if (m_ChunkParser)
        m_ChunkParser->CurrentPos += DwordCount;
}

void CKStateChunk::Goto(int DwordOffset) {
    if (m_ChunkParser)
        m_ChunkParser->CurrentPos = DwordOffset;
}

int CKStateChunk::GetCurrentPos() {
    if (m_ChunkParser)
        return m_ChunkParser->CurrentPos;
    else
        return -1;
}

void *CKStateChunk::LockWriteBuffer(int DwordCount) {
    if (!m_ChunkParser)
        return nullptr;
    CheckSize(DwordCount * sizeof(int));
    return &m_Data[m_ChunkParser->CurrentPos];
}

void *CKStateChunk::LockReadBuffer() {
    if (!m_ChunkParser)
        return nullptr;
    return &m_Data[m_ChunkParser->CurrentPos];
}

void CKStateChunk::AddChunk(CKStateChunk *chunk) {
    if (!m_ChunkParser || !chunk || !chunk->m_Data || chunk->m_ChunkSize == 0)
        return;

    CheckSize(chunk->m_ChunkSize * sizeof(int));
    memcpy(&m_Data[m_ChunkParser->CurrentPos], chunk->m_Data, chunk->m_ChunkSize * sizeof(int));
    if (m_ChunkParser->PrevIdentifierPos != m_ChunkParser->CurrentPos)
        m_Data[m_ChunkParser->PrevIdentifierPos + 1] = m_ChunkParser->CurrentPos;

    int j = m_ChunkParser->CurrentPos;
    if (m_Data[m_ChunkParser->CurrentPos + 1] != 0) {
        do {
            int *p = &m_Data[m_ChunkParser->CurrentPos + 1];
            j = *p + m_ChunkParser->CurrentPos;
            *p = j;
        } while (m_Data[j + 1] != 0);
    }

    if (chunk->m_Ids) {
        if (!m_Ids)
            m_Ids = new IntListStruct;
        m_Ids->Append(chunk->m_Ids, m_ChunkParser->CurrentPos);
    }

    if (chunk->m_Chunks) {
        if (!m_Chunks)
            m_Chunks = new IntListStruct;
        m_Chunks->Append(chunk->m_Chunks, m_ChunkParser->CurrentPos);
    }

    if (chunk->m_Managers) {
        if (!m_Managers)
            m_Managers = new IntListStruct;
        m_Managers->Append(chunk->m_Managers, m_ChunkParser->CurrentPos);
    }

    m_ChunkParser->PrevIdentifierPos = j;
    m_ChunkParser->CurrentPos += chunk->m_ChunkSize;;
    m_DataVersion = chunk->m_DataVersion;
    m_ChunkVersion = chunk->m_ChunkVersion;
    m_File = chunk->m_File;
}

void CKStateChunk::AddChunkAndDelete(CKStateChunk *chunk) {
    if (!m_ChunkParser || !chunk || !chunk->m_Data)
        return;

    m_Dynamic = chunk->m_Dynamic;
    if (m_ChunkParser->DataSize != 0
        || m_ChunkParser->CurrentPos != 0
        || m_ChunkParser->PrevIdentifierPos != 0
        || m_Ids
        || m_Chunks
        || m_Managers) {
        AddChunk(chunk);
    } else {
        if (m_ChunkParser) {
            m_ChunkParser = chunk->m_ChunkParser;
            chunk->m_ChunkParser->Clear();
        }
        m_ChunkSize = chunk->m_ChunkSize;
        m_Data = chunk->m_Data;
        m_Ids = chunk->m_Ids;
        m_Chunks = chunk->m_Chunks;
        m_Managers = chunk->m_Managers;
        m_File = chunk->m_File;
        m_DataVersion = chunk->m_DataVersion;
        m_ChunkVersion = chunk->m_ChunkVersion;
        chunk->m_ChunkSize = 0;
        chunk->m_Data = nullptr;
        chunk->m_Managers = nullptr;
        chunk->m_Chunks = nullptr;
        chunk->m_Ids = nullptr;
    }

    delete chunk;
}

void CKStateChunk::StartManagerSequence(CKGUID man, int count) {
    if (!m_Managers)
        m_Managers = new IntListStruct;

    m_Managers->AddEntries(m_ChunkParser->CurrentPos);
    CheckSize((count + 3) + sizeof(int));
    m_Data[m_ChunkParser->CurrentPos++] = count;
    m_Data[m_ChunkParser->CurrentPos++] = man.d1;
    m_Data[m_ChunkParser->CurrentPos++] = man.d2;
}

void CKStateChunk::WriteManagerSequence(int val) {
    m_Data[m_ChunkParser->CurrentPos++] = val;
}

void CKStateChunk::WriteManagerInt(CKGUID Manager, int val) {
    if (!m_Managers)
        m_Managers = new IntListStruct;

    m_Managers->AddEntry(m_ChunkParser->CurrentPos);
    m_Data[m_ChunkParser->CurrentPos++] = Manager.d1;
    m_Data[m_ChunkParser->CurrentPos++] = Manager.d2;
    m_Data[m_ChunkParser->CurrentPos++] = val;
}

void CKStateChunk::WriteArray_LEndian(int elementCount, int elementSize, void *srcData) {
    if (srcData && elementCount > 0 && elementSize > 0) {
        // Check for integer overflow in total bytes calculation
        if (elementCount > INT_MAX / elementSize) {
            // Handle overflow
            CheckSize(8);
            m_Data[m_ChunkParser->CurrentPos++] = 0;
            m_Data[m_ChunkParser->CurrentPos++] = 0;
            return;
        }

        // Calculate total bytes and required dword count (rounding up)
        const size_t totalBytes = static_cast<size_t>(elementSize) * static_cast<size_t>(elementCount);
        if (totalBytes > static_cast<size_t>(INT_MAX)) {
            CheckSize(8);
            m_Data[m_ChunkParser->CurrentPos++] = 0;
            m_Data[m_ChunkParser->CurrentPos++] = 0;
            return;
        }
        const size_t dwordCount = (totalBytes + sizeof(int) - 1) / sizeof(int); // Convert bytes to 4-byte dwords

        // Ensure buffer has space for: [totalBytes (int)] + [count (int)] + data
        const size_t requiredBytes = sizeof(int) * 2 + dwordCount * sizeof(int);
        if (requiredBytes > static_cast<size_t>(INT_MAX)) {
            CheckSize(8);
            m_Data[m_ChunkParser->CurrentPos++] = 0;
            m_Data[m_ChunkParser->CurrentPos++] = 0;
            return;
        }
        CheckSize(static_cast<int>(requiredBytes));

        // Write array header
        m_Data[m_ChunkParser->CurrentPos++] = static_cast<int>(totalBytes);
        m_Data[m_ChunkParser->CurrentPos++] = elementCount;

        // Copy array data
        memcpy(&m_Data[m_ChunkParser->CurrentPos], srcData, totalBytes);
        m_ChunkParser->CurrentPos += static_cast<int>(dwordCount);
    } else {
        // Write empty array marker
        CheckSize(8); // Space for two zero ints
        m_Data[m_ChunkParser->CurrentPos++] = 0;
        m_Data[m_ChunkParser->CurrentPos++] = 0;
    }
}

void CKStateChunk::WriteArray_LEndian16(int elementCount, int elementSize, void *srcData) {
    WriteArray_LEndian(elementCount, elementSize, srcData);
}

int CKStateChunk::StartManagerReadSequence(CKGUID *guid) {
    int i = m_Data[m_ChunkParser->CurrentPos++];
    if (guid)
        *guid = *(CKGUID *) &m_Data[m_ChunkParser->CurrentPos];
    m_ChunkParser->CurrentPos += 2;
    return i;
}

int CKStateChunk::ReadManagerInt(CKGUID *guid) {
    if (guid)
        *guid = *(CKGUID *) &m_Data[m_ChunkParser->CurrentPos];
    m_ChunkParser->CurrentPos += 2;
    return m_Data[m_ChunkParser->CurrentPos++];
}

int CKStateChunk::ReadArray_LEndian(void **array) {
    ChunkParser *parser = m_ChunkParser;
    if (parser && parser->CurrentPos < m_ChunkSize) {
        // Read array metadata
        const int dataSizeBytes = m_Data[parser->CurrentPos++];
        const int elementCount = m_Data[parser->CurrentPos++];

        if (dataSizeBytes > 0 && elementCount > 0) {
            // Calculate needed dwords (4-byte chunks)
            const int dwordCount = (dataSizeBytes + 3) / 4;

            // Bounds check before allocation
            if (parser->CurrentPos + dwordCount > m_ChunkSize) {
                *array = nullptr;
                return 0;
            }

            // Allocate and check allocation success
            void *arrayData = new CKBYTE[dataSizeBytes];
            if (!arrayData) {
                *array = nullptr;
                return 0;
            }

            memcpy(arrayData, &m_Data[parser->CurrentPos], dataSizeBytes);
            // Update parser position
            parser->CurrentPos += dwordCount;
            *array = arrayData;
            return elementCount;
        }

        // Invalid size parameters
        *array = nullptr;
    } else if (m_File) {
        // Report read error through context
        m_File->m_Context->OutputToConsole("Chunk Read Error", true);
    }

    return 0;
}

int CKStateChunk::ReadArray_LEndian16(void **array) {
    return ReadArray_LEndian(array);
}

int CKStateChunk::ReadManagerIntSequence() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize)
        return 0;
    return m_Data[m_ChunkParser->CurrentPos++];
}

void CKStateChunk::WriteObjectID(CK_ID obj) {
    CheckSize(sizeof(int));
    if (m_File) {
        m_Data[m_ChunkParser->CurrentPos] = m_File->SaveFindObjectIndex(obj);
    } else if (obj) {
        if (!m_Ids)
            m_Ids = new IntListStruct;
        m_Ids->AddEntry(m_ChunkParser->CurrentPos);
        m_Data[m_ChunkParser->CurrentPos] = (int) obj;
    } else {
        m_Data[m_ChunkParser->CurrentPos] = 0;
    }
    ++m_ChunkParser->CurrentPos;
}

void CKStateChunk::WriteObject(CKObject *obj) {
    if (m_ChunkClassID == -1 || obj && (m_Dynamic || !obj->IsDynamic()))
        WriteObjectID((obj) ? obj->GetID() : 0);
    else
        WriteObjectID(0);
}

void CKStateChunk::WriteObjectIDSequence(CK_ID id) {
    CheckSize(sizeof(int));
    if (m_File)
        m_Data[m_ChunkParser->CurrentPos] = m_File->SaveFindObjectIndex(id);
    else
        m_Data[m_ChunkParser->CurrentPos] = (int) id;
    ++m_ChunkParser->CurrentPos;
}

void CKStateChunk::WriteObjectSequence(CKObject *obj) {
    if (m_ChunkClassID == -1 || obj && (m_Dynamic || !obj->IsDynamic()))
        WriteObjectIDSequence((obj) ? obj->GetID() : 0);
    else
        WriteObjectIDSequence(0);
}

void CKStateChunk::StartObjectIDSequence(int count) {
    CheckSize(sizeof(int));
    if (count != 0 && !m_File) {
        if (!m_Ids)
            m_Ids = new IntListStruct;
        m_Ids->AddEntries(m_ChunkParser->CurrentPos);
    }
    m_Data[m_ChunkParser->CurrentPos++] = count;
}

CK_ID CKStateChunk::ReadObjectID() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return 0;
    }

    int id = m_Data[m_ChunkParser->CurrentPos];
    if (m_ChunkVersion >= CHUNK_VERSION1) {
        ++m_ChunkParser->CurrentPos;
        if (!m_File)
            return id;
        if (id >= 0)
            return m_File->LoadFindObject(id);
    } else {
        ++m_ChunkParser->CurrentPos;
        if (id != 0) {
            if (m_ChunkParser->CurrentPos + 3 > m_ChunkSize) {
                if (m_File)
                    m_File->m_Context->OutputToConsole("Chunk Read error (legacy format)");
                return 0;
            }
            m_ChunkParser->CurrentPos += 2;
            return m_Data[m_ChunkParser->CurrentPos++];
        }
    }

    return 0;
}

CKObject *CKStateChunk::ReadObject(CKContext *context) {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return nullptr;
    }
    return context->GetObject(ReadObjectID());
}

void CKStateChunk::WriteObjectArray(CKObjectArray *array, CKContext *context) {
    int objectCount = (array) ? array->GetCount() : 0;
    CheckSize(objectCount * sizeof(int));
    StartObjectIDSequence(objectCount);
    if (array) {
        array->Reset();
        while (!array->EndOfList()) {
            CK_ID id = 0;
            if (context) {
                CKObject *object = array->GetData(context);
                if (object && (m_ChunkClassID == -1 || m_Dynamic || object->IsDynamic()))
                    id = object->GetID();
            } else {
                id = array->GetDataId();
            }
            if (m_File) {
                int index = -1;
                XFileObjectsTableIt it = m_File->m_ObjectsHashTable.Find(id);
                if (it != m_File->m_ObjectsHashTable.End())
                    index = *it;
                m_Data[m_ChunkParser->CurrentPos++] = index;
            } else {
                m_Data[m_ChunkParser->CurrentPos++] = id;
            }
            array->Next();
        }
    }
}

void CKStateChunk::ReadObjectArray(CKObjectArray *array) {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return;
    }

    int count = m_Data[m_ChunkParser->CurrentPos++];
    if (count == 0) {
        if (array)
            array->Clear();
        return;
    }

    if (m_ChunkVersion < CHUNK_VERSION1) {
        m_ChunkParser->CurrentPos += 4;
        count = m_Data[m_ChunkParser->CurrentPos++];
    }

    if (!array)
        m_ChunkParser->CurrentPos += count;

    array->Clear();
    if (count + m_ChunkParser->CurrentPos <= m_ChunkSize) {
        if (m_File) {
            for (int i = 0; i < count; ++i) {
                int index = m_Data[m_ChunkParser->CurrentPos++];
                CK_ID id = (index >= 0) ? m_File->m_FileObjects[index].CreatedObject : 0;
                array->InsertRear(id);
            }
        } else {
            for (int i = 0; i < count; ++i) {
                CK_ID id = m_Data[m_ChunkParser->CurrentPos++];
                array->InsertRear(id);
            }
        }
    }
}

const XObjectArray &CKStateChunk::ReadXObjectArray() {
    m_TempXOPA.Resize(0);

    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return m_TempXOA;
    }

    int count = StartReadSequence();
    if (count != 0) {
        if (m_ChunkVersion < CHUNK_VERSION1) {
            m_ChunkParser->CurrentPos += 4;
            count = ReadInt();
            m_TempXOA.Resize(count);
            for (int i = 0; i < count; ++i)
                m_TempXOA[i] = m_Data[m_ChunkParser->CurrentPos++];
        } else {
            m_TempXOA.Resize(count);
            if (count + m_ChunkParser->CurrentPos <= m_ChunkSize) {
                if (m_File) {
                    for (int i = 0; i < count; ++i) {
                        int index = m_Data[m_ChunkParser->CurrentPos++];
                        CK_ID id = (index >= 0) ? m_File->m_FileObjects[index].CreatedObject : 0;
                        m_TempXOA[i] = id;
                    }
                } else {
                    for (int i = 0; i < count; ++i)
                        m_TempXOA[i] = m_Data[m_ChunkParser->CurrentPos++];
                }
            }
        }
    }

    return m_TempXOA;
}

const XObjectPointerArray &CKStateChunk::ReadXObjectArray(CKContext *context) {
    m_TempXOPA.Resize(0);
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return m_TempXOPA;
    }

    int count = StartReadSequence();
    if (count != 0) {
        if (m_ChunkVersion < CHUNK_VERSION1) {
            m_ChunkParser->CurrentPos += 4;
            count = ReadInt();
            m_TempXOPA.Reserve(count);
            for (int i = 0; i < count; ++i) {
                CK_ID id = m_Data[m_ChunkParser->CurrentPos++];
                CKObject *obj = context->GetObject(id);
                if (obj)
                    m_TempXOPA.PushBack(obj);
            }
        } else {
            m_TempXOPA.Reserve(count);
            if (count + m_ChunkParser->CurrentPos <= m_ChunkSize) {
                if (m_File) {
                    for (int i = 0; i < count; ++i) {
                        int index = m_Data[m_ChunkParser->CurrentPos++];
                        CK_ID id = (index >= 0) ? m_File->m_FileObjects[index].CreatedObject : 0;
                        CKObject *obj = context->GetObject(id);
                        if (obj)
                            m_TempXOPA.PushBack(obj);
                    }
                } else {
                    for (int i = 0; i < count; ++i) {
                        CK_ID id = m_Data[m_ChunkParser->CurrentPos++];
                        CKObject *obj = context->GetObject(id);
                        if (obj)
                            m_TempXOPA.PushBack(obj);
                    }
                }
            }
        }
    }

    return m_TempXOPA;
}

CKObjectArray *CKStateChunk::ReadObjectArray() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return nullptr;
    }

    int count = m_Data[m_ChunkParser->CurrentPos++];
    if (count == 0)
        return nullptr;

    if (m_ChunkVersion < CHUNK_VERSION1) {
        m_ChunkParser->CurrentPos += 4;
        count = m_Data[m_ChunkParser->CurrentPos++];
        if (count == 0)
            return nullptr;
    }

    CKObjectArray *array = nullptr;
    if (count + m_ChunkParser->CurrentPos <= m_ChunkSize) {
        array = new CKObjectArray;
        if (!array) {
            m_ChunkParser->CurrentPos += count;
            return nullptr;
        }

        if (m_File) {
            for (int i = 0; i < count; ++i) {
                int index = m_Data[m_ChunkParser->CurrentPos++];
                CK_ID id = (index >= 0) ? m_File->m_FileObjects[index].CreatedObject : 0;
                array->InsertRear(id);
            }
        } else {
            for (int i = 0; i < count; ++i) {
                CK_ID id = m_Data[m_ChunkParser->CurrentPos++];
                array->InsertRear(id);
            }
        }
    }

    return array;
}

void CKStateChunk::WriteSubChunk(CKStateChunk *sub) {
    int size = 4;
    if (sub) {
        size = 8 * sizeof(int);
        if (sub->m_ChunkSize != 0)
            size += sub->m_ChunkSize * sizeof(int);
        if (sub->m_Ids)
            size += sub->m_Ids->Size * sizeof(int);
        if (sub->m_Chunks)
            size += sub->m_Chunks->Size * sizeof(int);
        if (sub->m_Managers)
            size += sub->m_Managers->Size * sizeof(int);
    }
    CheckSize(size);
    m_Data[m_ChunkParser->CurrentPos++] = size / sizeof(int) - 1;
    if (sub) {
        if (!m_Chunks)
            m_Chunks = new IntListStruct;
        m_Chunks->AddEntry(m_ChunkParser->CurrentPos - 1);
        m_Data[m_ChunkParser->CurrentPos++] = sub->m_ChunkClassID;
        m_Data[m_ChunkParser->CurrentPos++] = sub->m_DataVersion | sub->m_ChunkVersion << 16;
        m_Data[m_ChunkParser->CurrentPos++] = sub->m_ChunkSize;
        m_Data[m_ChunkParser->CurrentPos++] = sub->m_File != nullptr;
        m_Data[m_ChunkParser->CurrentPos++] = (sub->m_Ids) ? sub->m_Ids->Size : 0;
        m_Data[m_ChunkParser->CurrentPos++] = (sub->m_Chunks) ? sub->m_Chunks->Size : 0;
        m_Data[m_ChunkParser->CurrentPos++] = (sub->m_Managers) ? sub->m_Managers->Size : 0;
        if (sub->m_ChunkSize != 0)
            WriteBufferNoSize_LEndian(sub->m_ChunkSize * sizeof(int), sub->m_Data);
        if (sub->m_Ids && sub->m_Ids->Size > 0)
            WriteBufferNoSize_LEndian(sub->m_Ids->Size * sizeof(int), sub->m_Ids->Data);
        if (sub->m_Chunks && sub->m_Chunks->Size > 0)
            WriteBufferNoSize_LEndian(sub->m_Chunks->Size * sizeof(int), sub->m_Chunks->Data);
        if (sub->m_Managers && sub->m_Managers->Size > 0)
            WriteBufferNoSize_LEndian(sub->m_Managers->Size * sizeof(int), sub->m_Managers->Data);
    }
}

void CKStateChunk::StartSubChunkSequence(int count) {
    if (!m_Chunks)
        m_Chunks = new IntListStruct;
    m_Chunks->AddEntries(m_ChunkParser->CurrentPos - 1);
    CheckSize(sizeof(int));
    m_Data[m_ChunkParser->CurrentPos++] = count;
}

int CKStateChunk::StartReadSequence() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize)
        return 0;

    return m_Data[m_ChunkParser->CurrentPos++];
}

void CKStateChunk::WriteSubChunkSequence(CKStateChunk *sub) {
    int size = 4;
    if (sub) {
        size = 8 * sizeof(int);
        if (sub->m_ChunkSize != 0)
            size += sub->m_ChunkSize * sizeof(int);
        if (sub->m_Ids)
            size += sub->m_Ids->Size * sizeof(int);
        if (sub->m_Chunks)
            size += sub->m_Chunks->Size * sizeof(int);
        if (sub->m_Managers)
            size += sub->m_Managers->Size * sizeof(int);
    }
    CheckSize(size);
    m_Data[m_ChunkParser->CurrentPos++] = size / sizeof(int) - 1;
    if (sub) {
        m_Data[m_ChunkParser->CurrentPos++] = sub->m_ChunkClassID;
        m_Data[m_ChunkParser->CurrentPos++] = sub->m_DataVersion | sub->m_ChunkVersion << 16;
        m_Data[m_ChunkParser->CurrentPos++] = sub->m_ChunkSize;
        m_Data[m_ChunkParser->CurrentPos++] = sub->m_File != nullptr;
        m_Data[m_ChunkParser->CurrentPos++] = (sub->m_Ids) ? sub->m_Ids->Size : 0;
        m_Data[m_ChunkParser->CurrentPos++] = (sub->m_Chunks) ? sub->m_Chunks->Size : 0;
        m_Data[m_ChunkParser->CurrentPos++] = (sub->m_Managers) ? sub->m_Managers->Size : 0;
        if (sub->m_ChunkSize != 0)
            WriteBufferNoSize_LEndian(sub->m_ChunkSize * sizeof(int), sub->m_Data);
        if (sub->m_Ids && sub->m_Ids->Size > 0)
            WriteBufferNoSize_LEndian(sub->m_Ids->Size * sizeof(int), sub->m_Ids->Data);
        if (sub->m_Chunks && sub->m_Chunks->Size > 0)
            WriteBufferNoSize_LEndian(sub->m_Chunks->Size * sizeof(int), sub->m_Chunks->Data);
        if (sub->m_Managers && sub->m_Managers->Size > 0)
            WriteBufferNoSize_LEndian(sub->m_Managers->Size * sizeof(int), sub->m_Managers->Data);
    }
}

CKStateChunk *CKStateChunk::ReadSubChunk() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return nullptr;
    }

    int size = m_Data[m_ChunkParser->CurrentPos++];
    if (size <= 0 || size + m_ChunkParser->CurrentPos > m_ChunkSize)
        return nullptr;

    int classID = m_Data[m_ChunkParser->CurrentPos++];
    CKStateChunk *sub = CreateCKStateChunk(classID, nullptr);

    if (m_ChunkVersion >= CHUNK_VERSION1) {
        int version = m_Data[m_ChunkParser->CurrentPos++];
        sub->m_DataVersion = (short int) (version & 0x0000FFFF);
        sub->m_ChunkVersion = (short int) ((version & 0xFFFF0000) >> 16);
        sub->m_ChunkSize = m_Data[m_ChunkParser->CurrentPos++];
        CKBOOL hasFile = m_Data[m_ChunkParser->CurrentPos++];
        int idCount = m_Data[m_ChunkParser->CurrentPos++];
        int chunkCount = m_Data[m_ChunkParser->CurrentPos++];
        int managerCount = 0;
        if (m_ChunkVersion > 4)
            managerCount = m_Data[m_ChunkParser->CurrentPos++];
        if (sub->m_ChunkSize != 0) {
            sub->m_Data = new int[sub->m_ChunkSize];
            if (!sub->m_Data) {
                delete sub;
                return nullptr;
            }
            ReadAndFillBuffer(sub->m_ChunkSize * sizeof(int), sub->m_Data);
        }

        if (idCount > 0) {
            sub->m_Ids = new IntListStruct;
            sub->m_Ids->AllocatedSize = idCount;
            sub->m_Ids->Size = idCount;
            sub->m_Ids->Data = new int[idCount];
            if (!sub->m_Ids->Data) {
                delete sub;
                return nullptr;
            }
            ReadAndFillBuffer(idCount * sizeof(int), sub->m_Ids->Data);
        }

        if (chunkCount > 0) {
            sub->m_Chunks = new IntListStruct;
            sub->m_Chunks->AllocatedSize = chunkCount;
            sub->m_Chunks->Size = chunkCount;
            sub->m_Chunks->Data = new int[chunkCount];
            if (!sub->m_Chunks->Data) {
                delete sub;
                return nullptr;
            }
            ReadAndFillBuffer(chunkCount * sizeof(int), sub->m_Chunks->Data);
        }

        if (managerCount > 0) {
            sub->m_Managers = new IntListStruct;
            sub->m_Managers->AllocatedSize = managerCount;
            sub->m_Managers->Size = managerCount;
            sub->m_Managers->Data = new int[managerCount];
            if (!sub->m_Managers->Data) {
                delete sub;
                return nullptr;
            }
            ReadAndFillBuffer(managerCount * sizeof(int), sub->m_Managers->Data);
        }

        if (hasFile && m_File)
            sub->m_File = m_File;

        return sub;
    } else {
        sub->m_ChunkSize = m_Data[m_ChunkParser->CurrentPos++];
        ++m_ChunkParser->CurrentPos;
        if (sub->m_ChunkSize == 0)
            return sub;
        sub->m_Data = new int[sub->m_ChunkSize];
        if (sub->m_Data) {
            memcpy(sub->m_Data, &m_Data[m_ChunkParser->CurrentPos], sub->m_ChunkSize * sizeof(int));
            m_ChunkParser->CurrentPos += sub->m_ChunkSize;
            return sub;
        }
    }

    delete sub;
    return nullptr;
}

void CKStateChunk::WriteByte(CKCHAR byte) {
    CheckSize(sizeof(int));
    m_Data[m_ChunkParser->CurrentPos++] = byte;
}

CKBYTE CKStateChunk::ReadByte() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize)
        return 0;
    return m_Data[m_ChunkParser->CurrentPos++];
}

void CKStateChunk::WriteWord(CKWORD data) {
    CheckSize(sizeof(int));
    m_Data[m_ChunkParser->CurrentPos++] = data;
}

CKWORD CKStateChunk::ReadWord() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize)
        return 0;
    return m_Data[m_ChunkParser->CurrentPos++];
}

void CKStateChunk::WriteInt(int data) {
    CheckSize(sizeof(int));
    m_Data[m_ChunkParser->CurrentPos++] = data;
}

int CKStateChunk::ReadInt() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return 0;
    }
    return m_Data[m_ChunkParser->CurrentPos++];
}

void CKStateChunk::WriteDword(CKDWORD data) {
    WriteInt(data);
}

CKDWORD CKStateChunk::ReadDword() {
    return ReadInt();
}

void CKStateChunk::WriteDwordAsWords(CKDWORD data) {
    WriteInt(data);
}

CKDWORD CKStateChunk::ReadDwordAsWords() {
    return ReadInt();
}

void CKStateChunk::WriteGuid(CKGUID data) {
    CheckSize(sizeof(CKGUID));
    m_Data[m_ChunkParser->CurrentPos++] = data.d1;
    m_Data[m_ChunkParser->CurrentPos++] = data.d2;
}

CKGUID CKStateChunk::ReadGuid() {
    CKGUID guid;
    if (m_ChunkParser->CurrentPos + 2 > m_ChunkSize)
        return guid;
    guid.d1 = m_Data[m_ChunkParser->CurrentPos++];
    guid.d2 = m_Data[m_ChunkParser->CurrentPos++];
    return guid;
}

void CKStateChunk::WriteFloat(float data) {
    WriteInt(*(int *) &data);
}

float CKStateChunk::ReadFloat() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return 0;
    }
    return *(float *) &m_Data[m_ChunkParser->CurrentPos++];
}

void CKStateChunk::WriteBuffer(int size, void *buf) {
    WriteBuffer_LEndian(size, buf);
}

void CKStateChunk::WriteBuffer_LEndian(int size, void *buf) {
    if (!buf) {
        CheckSize(sizeof(CKGUID));
        m_Data[m_ChunkParser->CurrentPos++] = 0;
    } else {
        int sz = (size + 3) / sizeof(int);
        CheckSize((sz + 1) * sizeof(int));
        m_Data[m_ChunkParser->CurrentPos++] = size;
        memcpy(&m_Data[m_ChunkParser->CurrentPos], buf, size);
        m_ChunkParser->CurrentPos += sz;
    }
}

void CKStateChunk::WriteBuffer_LEndian16(int size, void *buf) {
    WriteBuffer_LEndian(size, buf);
}

void CKStateChunk::WriteBufferNoSize(int size, void *buf) {
    WriteBufferNoSize_LEndian(size, buf);
}

void CKStateChunk::WriteBufferNoSize_LEndian(int size, void *buf) {
    if (buf) {
        int sz = (size + 3) / sizeof(int);
        CheckSize(sz * sizeof(int));
        memcpy(&m_Data[m_ChunkParser->CurrentPos], buf, size);
        m_ChunkParser->CurrentPos += sz;
    }
}

void CKStateChunk::WriteBufferNoSize_LEndian16(int size, void *buf) {
    WriteBufferNoSize_LEndian(size, buf);
}

int CKStateChunk::ReadBuffer(void **buffer) {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return 0;
    }

    CKBYTE *buf = nullptr;
    int size = m_Data[m_ChunkParser->CurrentPos];
    int sz = (size + 3) / sizeof(int);
    ++m_ChunkParser->CurrentPos;
    if (sz + m_ChunkParser->CurrentPos <= m_ChunkSize) {
        if (size > 0) {
            buf = new CKBYTE[size];
            if (!buf) {
                *buffer = nullptr;
                return 0;
            }
            memcpy(buf, &m_Data[m_ChunkParser->CurrentPos], size);
            m_ChunkParser->CurrentPos += sz;
        }
        *buffer = buf;
        return size;
    }

    *buffer = nullptr;
    return 0;
}

void CKStateChunk::ReadAndFillBuffer(void *buffer) {
    ReadAndFillBuffer_LEndian(buffer);
}

void CKStateChunk::ReadAndFillBuffer_LEndian(void *buffer) {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return;
    }

    int size = m_Data[m_ChunkParser->CurrentPos];
    ++m_ChunkParser->CurrentPos;
    if (size > 0) {
        memcpy(buffer, &m_Data[m_ChunkParser->CurrentPos], size);
        m_ChunkParser->CurrentPos += (size + 3) / sizeof(int);
    }
}

void CKStateChunk::ReadAndFillBuffer_LEndian16(void *buffer) {
    ReadAndFillBuffer_LEndian(buffer);
}

void CKStateChunk::ReadAndFillBuffer(int size, void *buffer) {
    ReadAndFillBuffer_LEndian(size, buffer);
}

void CKStateChunk::ReadAndFillBuffer_LEndian(int size, void *buffer) {
    if (buffer)
        memcpy(buffer, &m_Data[m_ChunkParser->CurrentPos], size);
    m_ChunkParser->CurrentPos += (size + 3) / sizeof(int);
}

void CKStateChunk::ReadAndFillBuffer_LEndian16(int size, void *buffer) {
    ReadAndFillBuffer_LEndian(size, buffer);
}

void CKStateChunk::WriteString(char *str) {
    if (!str) {
        CheckSize(sizeof(int));
        m_Data[m_ChunkParser->CurrentPos++] = 0;
    } else {
        int len = strlen(str);
        int size = len + 1;
        int dwords = (size + sizeof(int) - 1) / sizeof(int);
        CheckSize((dwords + 1) * sizeof(int));
        m_Data[m_ChunkParser->CurrentPos++] = size;
        if (size > 0) {
            memcpy(&m_Data[m_ChunkParser->CurrentPos], str, size);
            m_ChunkParser->CurrentPos += dwords;
        }
    }
}

int CKStateChunk::ReadString(CKSTRING *str) {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return 0;
    }

    CKSTRING s = nullptr;
    int size = m_Data[m_ChunkParser->CurrentPos];
    int sz = (size + 3) / sizeof(int);
    ++m_ChunkParser->CurrentPos;
    if (sz + m_ChunkParser->CurrentPos <= m_ChunkSize) {
        if (size > 0) {
            s = new char[size];
            if (!s) {
                *str = nullptr;
                return 0;
            }
            memcpy(s, &m_Data[m_ChunkParser->CurrentPos], size);
            m_ChunkParser->CurrentPos += sz;
        }
        *str = s;
        return size;
    }

    *str = nullptr;
    return 0;
}

void CKStateChunk::ReadString(XString &str) {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize) {
        str = "";
        return;
    }

    CKDWORD length = m_Data[m_ChunkParser->CurrentPos++];

    CKDWORD dwordCount = (length + 3) >> 2; // (length+3)/4

    if ((m_ChunkParser->CurrentPos + dwordCount) > m_ChunkSize) {
        str = "";
        return;
    }

    if (length == 0) {
        str = "";
        return;
    }

    str = (char *) &m_Data[m_ChunkParser->CurrentPos];
    m_ChunkParser->CurrentPos += dwordCount;
}

void CKStateChunk::WriteVector(const VxVector &v) {
    CheckSize(sizeof(VxVector));
    memcpy(&m_Data[m_ChunkParser->CurrentPos], &v, sizeof(VxVector));
    m_ChunkParser->CurrentPos += sizeof(VxVector) / sizeof(int);
}

void CKStateChunk::WriteVector(const VxVector *v) {
    if (v)
        WriteVector(*v);
}

void CKStateChunk::ReadVector(VxVector &v) {
    const int vectorInts = sizeof(VxVector) / sizeof(int);
    if (!m_ChunkParser || m_ChunkParser->CurrentPos + vectorInts > m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return;
    }

    memcpy(&v, &m_Data[m_ChunkParser->CurrentPos], sizeof(VxVector));
    m_ChunkParser->CurrentPos += vectorInts;
}

void CKStateChunk::ReadVector(VxVector *v) {
    if (v)
        ReadVector(*v);
}

void CKStateChunk::WriteMatrix(const VxMatrix &mat) {
    CheckSize(sizeof(VxMatrix));
    memcpy(&m_Data[m_ChunkParser->CurrentPos], &mat, sizeof(VxMatrix));
    m_ChunkParser->CurrentPos += sizeof(VxMatrix) / sizeof(int);
}

void CKStateChunk::ReadMatrix(VxMatrix &mat) {
    const int matrixInts = sizeof(VxMatrix) / sizeof(int);  // 16 ints
    if (!m_ChunkParser || m_ChunkParser->CurrentPos + matrixInts > m_ChunkSize) {
        if (m_File)
            m_File->m_Context->OutputToConsole("Chunk Read error");
        return;
    }

    memcpy(&mat, &m_Data[m_ChunkParser->CurrentPos], sizeof(VxMatrix));
    m_ChunkParser->CurrentPos += matrixInts;
}

int CKStateChunk::ConvertToBuffer(void *buffer) {
    int size = 2 * sizeof(int) + m_ChunkSize * sizeof(int);
    CKBYTE chunkOptions = 0;
    if (m_Ids) {
        size += m_Ids->Size * sizeof(int);
        chunkOptions |= CHNK_OPTION_IDS;
    }
    if (m_Chunks) {
        size += m_Chunks->Size * sizeof(int);
        chunkOptions |= CHNK_OPTION_CHN;
    }
    if (m_Managers) {
        size += m_Managers->Size * sizeof(int);
        chunkOptions |= CHNK_OPTION_MAN;
    }
    if (m_File) {
        chunkOptions |= CHNK_OPTION_FILE;
    }

    if (buffer) {
        int pos = 0;
        int *buf = (int *) buffer;
        CKWORD chunkVersion = m_ChunkVersion | chunkOptions;
        CKWORD dataVersion = m_DataVersion | m_ChunkClassID << 8;
        buf[pos++] = dataVersion | chunkVersion << 16;
        buf[pos++] = m_ChunkSize;
        memcpy(&buf[pos], m_Data, m_ChunkSize * sizeof(int));
        pos += m_ChunkSize;
        if (m_Ids) {
            memcpy(&buf[pos], m_Ids->Data, m_Ids->Size * sizeof(int));
            pos += m_Ids->Size;
        }
        if (m_Chunks) {
            memcpy(&buf[pos], m_Chunks->Data, m_Chunks->Size * sizeof(int));
            pos += m_Chunks->Size;
        }
        if (m_Managers) {
            memcpy(&buf[pos], m_Managers->Data, m_Managers->Size * sizeof(int));
            pos += m_Managers->Size;
        }
    }

    return size;
}

CKBOOL CKStateChunk::ConvertFromBuffer(void *buffer) {
    if (!buffer)
        return FALSE;

    int pos = 0;
    int *buf = (int *) buffer;

    Clear();
    int val = buf[pos++];

    // Extract raw version data - do NOT mask off high bytes yet
    // as VERSION3/4 formats encode options and classID in the high bytes
    int rawDataVersion = (val & 0x0000FFFF);
    int rawChunkVersion = ((val & 0xFFFF0000) >> 16);

    // For older formats, only the low byte is the actual version
    m_DataVersion = (short) (rawDataVersion & 0x00FF);
    m_ChunkVersion = (short) (rawChunkVersion & 0x00FF);

    if (m_ChunkVersion < CHUNK_VERSION2) {
        m_ChunkClassID = buf[pos++];
        m_ChunkSize = buf[pos++];
        ++pos;
        int idCount = buf[pos++];
        int chunkCount = buf[pos++];

        if (m_ChunkSize != 0) {
            m_Data = new int[m_ChunkSize];
            if (m_Data) {
                memcpy(m_Data, &buf[pos], m_ChunkSize * sizeof(int));
                pos += m_ChunkSize;
            }
        }

        if (idCount > 0) {
            m_Ids = new IntListStruct;
            m_Ids->AllocatedSize = idCount;
            m_Ids->Size = idCount;
            m_Ids->Data = new int[idCount];
            if (m_Ids->Data) {
                memcpy(m_Ids->Data, &buf[pos], idCount * sizeof(int));
                pos += idCount;
            }
        }

        if (chunkCount > 0) {
            m_Chunks = new IntListStruct;
            m_Chunks->AllocatedSize = chunkCount;
            m_Chunks->Size = chunkCount;
            m_Chunks->Data = new int[chunkCount];
            if (m_Chunks->Data) {
                memcpy(m_Chunks->Data, &buf[pos], chunkCount * sizeof(int));
                pos += chunkCount;
            }
        }

        m_File = nullptr;
        return TRUE;
    } else if (m_ChunkVersion == CHUNK_VERSION2) {
        m_ChunkClassID = buf[pos++];
        m_ChunkSize = buf[pos++];
        ++pos;
        int idCount = buf[pos++];
        int chunkCount = buf[pos++];
        int managerCount = buf[pos++];

        if (m_ChunkSize != 0) {
            m_Data = new int[m_ChunkSize];
            if (m_Data) {
                memcpy(m_Data, &buf[pos], m_ChunkSize * sizeof(int));
                pos += m_ChunkSize;
            }
        }

        if (idCount > 0) {
            m_Ids = new IntListStruct;
            m_Ids->AllocatedSize = idCount;
            m_Ids->Size = idCount;
            m_Ids->Data = new int[idCount];
            if (m_Ids->Data) {
                memcpy(m_Ids->Data, &buf[pos], idCount * sizeof(int));
                pos += idCount;
            }
        }

        if (chunkCount > 0) {
            m_Chunks = new IntListStruct;
            m_Chunks->AllocatedSize = chunkCount;
            m_Chunks->Size = chunkCount;
            m_Chunks->Data = new int[chunkCount];
            if (m_Chunks->Data) {
                memcpy(m_Chunks->Data, &buf[pos], chunkCount * sizeof(int));
                pos += chunkCount;
            }
        }

        if (managerCount > 0) {
            m_Managers = new IntListStruct;
            m_Managers->AllocatedSize = managerCount;
            m_Managers->Size = managerCount;
            m_Managers->Data = new int[managerCount];
            if (m_Managers->Data) {
                memcpy(m_Managers->Data, &buf[pos], managerCount * sizeof(int));
                pos += managerCount;
            }
        }

        m_File = nullptr;
        return TRUE;
    } else if (m_ChunkVersion <= CHUNK_VERSION4) {
        // VERSION3/4 format packs data as:
        // [ChunkVersion(high8:options, low8:version)][DataVersion(high8:classID, low8:version)]

        // Extract options and classID from the high bytes BEFORE they were masked
        CKBYTE chunkOptions = (rawChunkVersion & 0xFF00) >> 8;
        m_ChunkClassID = (rawDataVersion & 0xFF00) >> 8;

        m_ChunkSize = buf[pos++];
        if (m_ChunkSize != 0) {
            m_Data = new int[m_ChunkSize];
            if (m_Data) {
                memcpy(m_Data, &buf[pos], m_ChunkSize * sizeof(int));
                pos += m_ChunkSize;
            }
        }

        if ((chunkOptions & CHNK_OPTION_FILE) == 0)
            m_File = nullptr;

        if ((chunkOptions & CHNK_OPTION_IDS) != 0) {
            int idCount = buf[pos++];
            if (idCount > 0) {
                m_Ids = new IntListStruct;
                m_Ids->AllocatedSize = idCount;
                m_Ids->Size = idCount;
                m_Ids->Data = new int[idCount];
                if (m_Ids->Data) {
                    memcpy(m_Ids->Data, &buf[pos], idCount * sizeof(int));
                    pos += idCount;
                }
            }
        }

        if ((chunkOptions & CHNK_OPTION_CHN) != 0) {
            int chunkCount = buf[pos++];
            if (chunkCount > 0) {
                m_Chunks = new IntListStruct;
                m_Chunks->AllocatedSize = chunkCount;
                m_Chunks->Size = chunkCount;
                m_Chunks->Data = new int[chunkCount];
                if (m_Chunks->Data) {
                    memcpy(m_Chunks->Data, &buf[pos], chunkCount * sizeof(int));
                    pos += chunkCount;
                }
            }
        }

        if ((chunkOptions & CHNK_OPTION_MAN) != 0) {
            int managerCount = buf[pos++];
            if (managerCount > 0) {
                m_Managers = new IntListStruct;
                m_Managers->AllocatedSize = managerCount;
                m_Managers->Size = managerCount;
                m_Managers->Data = new int[managerCount];
                if (m_Managers->Data) {
                    memcpy(m_Managers->Data, &buf[pos], managerCount * sizeof(int));
                    pos += managerCount;
                }
            }
        }

        return TRUE;
    }

    return FALSE;
}

int CKStateChunk::RemapObject(CK_ID old_id, CK_ID new_id) {
    if (old_id == 0 || new_id == 0)
        return 0;

    CKDependenciesContext depContext(nullptr);
    depContext.m_MapID.Insert(old_id, new_id);
    return RemapObjects(nullptr, &depContext);
}

int CKStateChunk::RemapParameterInt(CKGUID ParameterType, int *ConversionTable, int NbEntries) {
    if (!ConversionTable)
        return 0;
    ChunkIteratorData data;
    data.ConversionTable = ConversionTable;
    data.NbEntries = NbEntries;
    data.Guid = ParameterType;
    data.Flag = TRUE;
    data.ChunkVersion = m_ChunkVersion;
    data.Data = m_Data;
    data.ChunkSize = m_ChunkSize;
    if (m_Chunks) {
        data.Chunks = m_Chunks->Data;
        data.ChunkCount = m_Chunks->Size;
    }
    if (m_Ids) {
        data.Ids = m_Ids->Data;
        data.IdCount = m_Ids->Size;
    }
    return IterateAndDo(ParameterRemapper, &data);
}

int CKStateChunk::RemapManagerInt(CKGUID Manager, int *ConversionTable, int NbEntries) {
    if (!ConversionTable)
        return 0;
    ChunkIteratorData data;
    data.ConversionTable = ConversionTable;
    data.NbEntries = NbEntries;
    data.Guid = Manager;
    data.Flag = TRUE;
    data.ChunkVersion = m_ChunkVersion;
    data.Data = m_Data;
    data.ChunkSize = m_ChunkSize;
    if (m_Chunks) {
        data.Chunks = m_Chunks->Data;
        data.ChunkCount = m_Chunks->Size;
    }
    if (m_Managers) {
        data.Managers = m_Managers->Data;
        data.ManagerCount = m_Managers->Size;
    }
    return IterateAndDo(ManagerRemapper, &data);
}

int CKStateChunk::RemapObjects(CKContext *context, CKDependenciesContext *Depcontext) {
    ChunkIteratorData data;
    data.Flag = TRUE;
    data.Context = context;
    data.DepContext = Depcontext;
    data.ChunkVersion = m_ChunkVersion;
    data.Data = m_Data;
    data.ChunkSize = m_ChunkSize;
    if (m_Chunks) {
        data.Chunks = m_Chunks->Data;
        data.ChunkCount = m_Chunks->Size;
    }
    if (m_Ids) {
        data.Ids = m_Ids->Data;
        data.IdCount = m_Ids->Size;
    }
    return IterateAndDo(ObjectRemapper, &data);
}

static CKDWORD g_RedShift_LSB;
static CKDWORD g_RedQuantShift_MSB; // 8 - RedBitCount
static CKDWORD g_GreenShift_LSB;
static CKDWORD g_GreenQuantShift_MSB;
static CKDWORD g_BlueShift_LSB;
static CKDWORD g_BlueQuantShift_MSB;
static CKDWORD g_AlphaShift_LSB;
static CKDWORD g_AlphaQuantShift_MSB;

static CKBOOL CalculateShiftsForComponent(CKDWORD mask, CKDWORD &outLsbShift, CKDWORD &outQuantShift) {
    if (mask == 0) {
        outLsbShift = 0;
        outQuantShift = 8; // Effectively no quantization or full range if 0 bits
        return TRUE;
    }

    CKDWORD lsbShift = 0;
    CKDWORD tempMask = 1;
    while ((tempMask & mask) == 0) {
        tempMask <<= 1;
        lsbShift++;
        if (lsbShift >= 32) {
            // Should not happen for valid masks
            outLsbShift = 0;
            outQuantShift = 8;
            return FALSE;
        }
    }

    CKDWORD bitCount = 0;
    // tempMask now aligns with the LSB of the component mask
    while ((tempMask & mask) != 0) {
        tempMask <<= 1;
        bitCount++;
        if (bitCount > 8) return FALSE; // Component uses more than 8 bits
        if (bitCount >= 32) break;      // Safety break
    }
    if (bitCount == 0) {
        // Mask was present but no contiguous bits from LSB? Invalid mask.
        outLsbShift = 0;
        outQuantShift = 8;
        return FALSE;
    }

    outLsbShift = lsbShift;
    outQuantShift = 8 - bitCount;
    return TRUE;
}

static CKBOOL CalculateMaskShifts(CKDWORD RedMask, CKDWORD GreenMask, CKDWORD BlueMask, CKDWORD AlphaMask) {
    if (!CalculateShiftsForComponent(RedMask, g_RedShift_LSB, g_RedQuantShift_MSB)) return FALSE;
    if (!CalculateShiftsForComponent(GreenMask, g_GreenShift_LSB, g_GreenQuantShift_MSB)) return FALSE;
    if (!CalculateShiftsForComponent(BlueMask, g_BlueShift_LSB, g_BlueQuantShift_MSB)) return FALSE;
    if (!CalculateShiftsForComponent(AlphaMask, g_AlphaShift_LSB, g_AlphaQuantShift_MSB)) return FALSE;
    return TRUE;
}

// TODO: Need fix
void CKStateChunk::WriteReaderBitmap(const VxImageDescEx &desc, CKBitmapReader *reader, CKBitmapProperties *bp) {
    if (!m_ChunkParser)
        return;

    if (!desc.Image || !reader || !bp) {
        WriteDword(0);
        return;
    }

    CKBOOL alphaIsSavedByReader = reader->IsAlphaSaved(bp);
    void *memoryToSave = nullptr; // Pointer to the data buffer that reader->SaveMemory will produce
    int savedMemorySize = 0; // Size of memoryToSave
    void *temp32BitBuffer = nullptr; // Buffer for on-the-fly conversion

    // 1. Prepare the image data in bp->m_Data and bp->m_Format
    //    If desc is not 32-bit, convert it to a temporary 32-bit RGBA buffer.
    if (desc.BitsPerPixel == 32 &&
        desc.AlphaMask == A_MASK &&
        desc.RedMask == R_MASK &&
        desc.GreenMask == G_MASK &&
        desc.BlueMask == B_MASK) {
        // Already in desired format for most readers (or readers expect 32-bit input for conversion)
        bp->m_Format = desc; // Copy descriptor
        bp->m_Data = desc.Image; // Use original image data directly
    } else {
        // Need to convert to a temporary 32-bit RGBA buffer
        VxImageDescEx temp32BitDesc;
        temp32BitDesc.Width = desc.Width;
        temp32BitDesc.Height = desc.Height;
        temp32BitDesc.BitsPerPixel = 32;
        temp32BitDesc.BytesPerLine = desc.Width * 4;
        // Standard ARGB32 masks
        temp32BitDesc.RedMask = R_MASK;
        temp32BitDesc.GreenMask = G_MASK;
        temp32BitDesc.BlueMask = B_MASK;
        temp32BitDesc.AlphaMask = A_MASK;

        size_t bufferSize = temp32BitDesc.Height * temp32BitDesc.BytesPerLine;
        if (bufferSize == 0) {
            // Avoid 0-byte allocation
            WriteDword(0);
            return;
        }
        temp32BitBuffer = new CKBYTE[bufferSize];
        if (!temp32BitBuffer) {
            WriteDword(0);
            return; // Out of memory
        }
        temp32BitDesc.Image = (CKBYTE *) temp32BitBuffer;

        VxDoBlit(desc, temp32BitDesc); // Convert original 'desc' to 'temp32BitDesc'

        bp->m_Format = temp32BitDesc; // Reader will save from this 32-bit format
        bp->m_Data = temp32BitBuffer;
    }

    // 2. Use the reader to save the (potentially temporary 32-bit) image to a memory buffer
    savedMemorySize = reader->SaveMemory(&memoryToSave, bp);

    // 3. Clean up temporary 32-bit buffer if it was used
    delete[] (CKBYTE *) temp32BitBuffer; // Safe if temp32BitBuffer is nullptr
    bp->m_Data = nullptr; // Crucial: bp->m_Data should not point to freed memory or original desc.Image after this

    // 4. Write the saved data to the chunk
    if (savedMemorySize > 0 && memoryToSave != nullptr) {
        // Path 1: Alpha is saved by reader OR there's no alpha mask in original
        if (alphaIsSavedByReader || !desc.AlphaMask) {
            WriteDword(1); // Type 1: Reader handles alpha or no alpha
            WriteBuffer(sizeof(CKFileExtension), bp->m_Ext.m_Data);
            WriteGuid(bp->m_ReaderGuid);
            WriteBuffer(savedMemorySize, memoryToSave);
        }
        // Path 2: Alpha NOT saved by reader AND original had an alpha mask
        // AND original was 32bpp (this condition seems implied by palette gen logic)
        else if (desc.BitsPerPixel == 32) {
            // Only try to save separate alpha if original was 32bpp
            WriteDword(2); // Type 2: Separate alpha palette/mask
            WriteBuffer(sizeof(CKFileExtension), bp->m_Ext.m_Data);
            WriteGuid(bp->m_ReaderGuid);
            WriteBuffer(savedMemorySize, memoryToSave);

            // Calculate and write alpha palette/mask
            CKDWORD alphaPalette[256];
            memset(alphaPalette, 0, sizeof(alphaPalette));

            // Calculate shifts for the original descriptor's alpha mask
            if (!CalculateMaskShifts(0, 0, 0, desc.AlphaMask)) {
                // Should not happen if desc.AlphaMask is valid
                // Write empty palette info as a fallback
                WriteDword(0); // 0 distinct alpha values
            } else {
                CKBYTE *pixelPtr = desc.Image;
                int pixels = desc.Width * desc.Height;
                int bytesPerPixel = desc.BitsPerPixel >> 3; // Bytes per pixel of original image

                for (int p = 0; p < pixels; ++p) {
                    CKDWORD pixelValue = 0;
                    // Safely read pixel based on BPP
                    if (bytesPerPixel == 1) {
                        pixelValue = *pixelPtr;
                    } else if (bytesPerPixel == 2) {
                        pixelValue = *(CKWORD *) pixelPtr;
                    } else if (bytesPerPixel == 3) {
                        // 24bpp, needs careful read
                        pixelValue = pixelPtr[0] | pixelPtr[1] << 8 | pixelPtr[2] << 16;
                    } else if (bytesPerPixel == 4) {
                        pixelValue = *(CKDWORD *) pixelPtr;
                    }

                    // Extract alpha using calculated shifts
                    CKBYTE alphaByte = (CKBYTE) ((pixelValue & desc.AlphaMask) >> g_AlphaShift_LSB);
                    alphaPalette[alphaByte] = 1; // Mark this alpha value as used

                    pixelPtr += bytesPerPixel;
                }

                int distinctAlphaCount = 0;
                for (int i = 0; i < 256; ++i) {
                    if (alphaPalette[i]) {
                        distinctAlphaCount++;
                    }
                }
                WriteDword(distinctAlphaCount);

                if (distinctAlphaCount == 1) {
                    for (int i = 0; i < 256; ++i) {
                        if (alphaPalette[i]) {
                            WriteDword(i); // Write the single alpha value
                            break;
                        }
                    }
                } else if (distinctAlphaCount > 1 && distinctAlphaCount < 256) {
                    // IDA's code also writes full mask if > 1
                    // Create a compact alpha mask (1 bit per pixel)
                    CKBYTE *alphaPlane = new CKBYTE[pixels];
                    if (alphaPlane) {
                        pixelPtr = desc.Image;
                        for (int p = 0; p < pixels; ++p) {
                            CKDWORD pixelValue = 0;
                            if (bytesPerPixel == 1) {
                                pixelValue = *pixelPtr;
                            } else if (bytesPerPixel == 2) {
                                pixelValue = *(CKWORD *) pixelPtr;
                            } else if (bytesPerPixel == 3) {
                                pixelValue = pixelPtr[0] | pixelPtr[1] << 8 | pixelPtr[2] << 16;
                            } else if (bytesPerPixel == 4) {
                                pixelValue = *(CKDWORD *) pixelPtr;
                            }

                            alphaPlane[p] = (CKBYTE) ((pixelValue & desc.AlphaMask) >> g_AlphaShift_LSB);
                            pixelPtr += bytesPerPixel;
                        }
                        WriteBuffer(pixels, alphaPlane);
                        delete[] alphaPlane;
                    } else {
                        // Allocation failed for alphaPlane
                        // This case is not handled well by IDA (would likely corrupt data)
                        // Fallback: Indicate 0 distinct alphas (or error)
                        WriteDword(0); // Or handle error more gracefully
                    }
                }
                // If distinctAlphaCount is 0 or 256, nothing more is written for alpha.
            }
        } else {
            // Fallback for non-32bpp images that need alpha but reader doesn't save it
            // This case wasn't explicitly handled by IDA's alpha palette logic (which assumed 32bpp source)
            // So, act like Type 1.
            WriteDword(1);
            WriteBuffer(sizeof(CKFileExtension), bp->m_Ext.m_Data);
            WriteGuid(bp->m_ReaderGuid);
            WriteBuffer(savedMemorySize, memoryToSave);
        }
        reader->ReleaseMemory(memoryToSave);
    } else {
        WriteDword(0);
    }
}

CKBOOL CKStateChunk::ReadReaderBitmap(const VxImageDescEx &desc) {
    CKDWORD formatType = ReadDword();
    if (formatType == 0)
        return FALSE;
    if (!desc.Image)
        return FALSE;

    CKFileExtension ext;
    CKGUID readerGuid;
    CKBitmapProperties *props = nullptr;

    const int cursor = m_ChunkParser->CurrentPos;
    if (cursor >= m_ChunkSize)
        return FALSE;

    int storedSize = m_Data[cursor];
    if (storedSize >= 0 && storedSize <= (int) sizeof(CKFileExtension)) {
        void *extBuffer = nullptr;
        int bytes = ReadBuffer(&extBuffer);
        if (bytes == (int) sizeof(CKFileExtension) && extBuffer) {
            memcpy(ext.m_Data, extBuffer, sizeof(CKFileExtension));
        } else {
            memset(ext.m_Data, 0, sizeof(CKFileExtension));
        }
        delete[] (CKBYTE *) extBuffer;
    } else {
        if (cursor + ((int) sizeof(CKFileExtension) + 3) / 4 > m_ChunkSize)
            return FALSE;
        ReadAndFillBuffer(sizeof(CKFileExtension), ext.m_Data);
    }
    readerGuid = ReadGuid();

    CKBitmapReader *reader = CKGetPluginManager()->GetBitmapReader(ext, &readerGuid);
    if (!reader)
        return FALSE;

    int size = ReadInt();
    int dwords = (size + 3) / sizeof(CKDWORD);
    if (m_ChunkParser->CurrentPos + dwords <= m_ChunkSize) {
        if (size > 0) {
            if (reader->ReadMemory(&m_Data[m_ChunkParser->CurrentPos], size, &props) != 0)
                return FALSE;

            props->m_Format.Image = (CKBYTE *) props->m_Data;
            VxDoBlit(props->m_Format, desc);

            reader->ReleaseMemory(props->m_Data);
            props->m_Data = nullptr;
            props->m_Format.ColorMap = nullptr;
        }
        Skip(dwords);
        if (formatType == 2) {
            int distinctAlphaCount = ReadInt();
            if (distinctAlphaCount == 1) {
                VxDoAlphaBlit(desc, (CKBYTE) ReadDword());
            } else {
                void *buffer = nullptr;
                int bufferSize = ReadBuffer(&buffer);
                if (buffer && bufferSize > 0) {
                    VxDoAlphaBlit(desc, (CKBYTE *) buffer);
                    delete[] (CKBYTE *) buffer;
                }
            }
        }
    }

    reader->Release();
    return TRUE;
}

// TODO: Need fix
void CKStateChunk::WriteRawBitmap(const VxImageDescEx &desc) {
    if (!m_ChunkParser)
        return;

    if (!desc.Image || desc.Width <= 0 || desc.Height <= 0 || desc.BitsPerPixel == 0) {
        WriteInt(0); // Write 0 for BitsPerPixel to indicate no/invalid image data
        return;
    }

    const int width = desc.Width;
    const int height = desc.Height;
    if (height > 0 && width > INT_MAX / height) {
        WriteInt(0);
        return;
    }

    // Write image metadata
    WriteInt(desc.BitsPerPixel);
    WriteInt(desc.Width);
    WriteInt(desc.Height);
    WriteDword(desc.AlphaMask);
    WriteDword(desc.RedMask);
    WriteDword(desc.GreenMask);
    WriteDword(desc.BlueMask);
    WriteDword(0); // Placeholder for compression type (0 = raw, 1 = JPEG)

    int bytesPerPixel = desc.BitsPerPixel >> 3;
    if (bytesPerPixel == 0 && desc.BitsPerPixel > 0) bytesPerPixel = 1; // For < 8 bpp, treat as 1 BPP for iteration
    if (bytesPerPixel == 0) {
        // Should not happen if BitsPerPixel > 0
        // Write empty buffers if BPP is invalid
        WriteBuffer(0, nullptr); // R
        WriteBuffer(0, nullptr); // G
        WriteBuffer(0, nullptr); // B
        WriteBuffer(0, nullptr); // A
        return;
    }

    const size_t planeSize = static_cast<size_t>(width) * static_cast<size_t>(height); // Number of pixels
    if (planeSize == 0 || planeSize > static_cast<size_t>(INT_MAX)) {
        WriteBuffer(0, nullptr);
        WriteBuffer(0, nullptr);
        WriteBuffer(0, nullptr);
        WriteBuffer(0, nullptr);
        return;
    }

    CKBYTE *rPlane = new CKBYTE[planeSize];
    CKBYTE *gPlane = new CKBYTE[planeSize];
    CKBYTE *bPlane = new CKBYTE[planeSize];
    CKBYTE *aPlane = (desc.AlphaMask != 0) ? new CKBYTE[planeSize] : nullptr;

    if (!rPlane || !gPlane || !bPlane || (desc.AlphaMask && !aPlane)) {
        // Memory allocation failure
        delete[] rPlane;
        delete[] gPlane;
        delete[] bPlane;
        delete[] aPlane;
        // We've already written metadata, so we must write empty buffers to maintain chunk structure
        WriteBuffer(0, nullptr); // R
        WriteBuffer(0, nullptr); // G
        WriteBuffer(0, nullptr); // B
        WriteBuffer(0, nullptr); // A (even if AlphaMask was 0, write empty alpha buffer)
        return;
    }

    // Process image from bottom-up (last scanline first)
    // planar_idx is the current index into the R,G,B,A planes
    for (int y = 0; y < desc.Height; ++y) {
        // Source scanline: Starts from last line and goes up.
        // desc.Image points to the top-left.
        // Scanline y from top is desc.Image + y * desc.BytesPerLine
        // Upside-down processing for channel separation:
        const CKBYTE *srcScanline = desc.Image + (desc.Height - 1 - y) * desc.BytesPerLine;
        CKBYTE *rPlanePtr = rPlane + static_cast<size_t>(y) * width;
        CKBYTE *gPlanePtr = gPlane + static_cast<size_t>(y) * width;
        CKBYTE *bPlanePtr = bPlane + static_cast<size_t>(y) * width;
        CKBYTE *aPlanePtr = aPlane ? (aPlane + static_cast<size_t>(y) * width) : nullptr;

        for (int x = 0; x < desc.Width; ++x) {
            const CKBYTE *srcPixel = srcScanline + x * bytesPerPixel;
            CKDWORD pixelValue = 0;

            // Read pixel value based on BPP
            if (bytesPerPixel == 1) {
                // 8-bit (usually palettized or grayscale)
                pixelValue = *srcPixel;
            } else if (bytesPerPixel == 2) {
                // 16-bit
                pixelValue = *(CKWORD *) srcPixel;
            } else if (bytesPerPixel == 3) {
                // 24-bit
                pixelValue = srcPixel[0] | (srcPixel[1] << 8) | (srcPixel[2] << 16);
            } else if (bytesPerPixel == 4) {
                // 32-bit
                pixelValue = *(CKDWORD *) srcPixel;
            }

            if (desc.BitsPerPixel == 32) {
                // Direct channel extraction for 32-bit (ARGB, RGBA etc.)
                bPlanePtr[x] = srcPixel[0];
                gPlanePtr[x] = srcPixel[1];
                rPlanePtr[x] = srcPixel[2];
                if (aPlanePtr) {
                    aPlanePtr[x] = (bytesPerPixel == 4) ? srcPixel[3] : 0xFF;
                }
            } else if (desc.BitsPerPixel < 24) {
                // Quantize for < 24 bpp
                static VxImageDescEx lastMaskDesc;
                if (memcmp(&lastMaskDesc, &desc, sizeof(VxImageDescEx)) != 0) {
                    CalculateMaskShifts(desc.RedMask, desc.GreenMask, desc.BlueMask, desc.AlphaMask);
                    lastMaskDesc = desc;
                }

                rPlanePtr[x] = (CKBYTE) (((pixelValue & desc.RedMask) >> g_RedShift_LSB) << g_RedQuantShift_MSB);
                gPlanePtr[x] = (CKBYTE) (((pixelValue & desc.GreenMask) >> g_GreenShift_LSB) << g_GreenQuantShift_MSB);
                bPlanePtr[x] = (CKBYTE) (((pixelValue & desc.BlueMask) >> g_BlueShift_LSB) << g_BlueQuantShift_MSB);
                if (aPlanePtr && desc.AlphaMask) {
                    aPlanePtr[x] = (CKBYTE) (((pixelValue & desc.AlphaMask) >> g_AlphaShift_LSB) << g_AlphaQuantShift_MSB);
                } else if (aPlanePtr) {
                    aPlanePtr[x] = 0xFF; // Default to opaque if no alpha mask but alpha plane exists
                }
            } else {
                // 24-bit (BGR usually)
                bPlanePtr[x] = srcPixel[0];
                gPlanePtr[x] = srcPixel[1];
                rPlanePtr[x] = srcPixel[2];
                if (aPlanePtr) {
                    aPlanePtr[x] = 0xFF; // 24-bit has no alpha, so opaque
                }
            }
        }
    }

    // Write de-interleaved planes
    WriteBuffer(static_cast<int>(planeSize), rPlane);
    WriteBuffer(static_cast<int>(planeSize), gPlane);
    WriteBuffer(static_cast<int>(planeSize), bPlane);
    if (aPlane) {
        WriteBuffer(static_cast<int>(planeSize), aPlane);
    } else {
        WriteBuffer(0, nullptr); // Write empty buffer if no alpha
    }

    // Cleanup
    delete[] rPlane;
    delete[] gPlane;
    delete[] bPlane;
    delete[] aPlane; // Safe if aPlane is nullptr
}

CKBYTE *CKStateChunk::ReadRawBitmap(VxImageDescEx &desc) {
    if (!m_ChunkParser)
        return nullptr;

    int originalBpp = ReadInt();
    if (originalBpp == 0)
        return nullptr;

    // Read metadata
    desc.Width = ReadInt();
    desc.Height = ReadInt();
    desc.AlphaMask = ReadDword();
    desc.RedMask = ReadDword();
    desc.GreenMask = ReadDword();
    desc.BlueMask = ReadDword();

    if (desc.Width <= 0 || desc.Height <= 0)
        return nullptr;

    const int width = desc.Width;
    const int height = desc.Height;
    if (height > 0 && width > INT_MAX / height)
        return nullptr;

    const size_t planeSize = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (planeSize == 0 || planeSize > static_cast<size_t>(INT_MAX))
        return nullptr;

    const size_t bytesPerLine = static_cast<size_t>(width) * sizeof(CKDWORD);
    if (bytesPerLine > static_cast<size_t>(INT_MAX))
        return nullptr;

    desc.BytesPerLine = static_cast<int>(bytesPerLine);
    desc.BitsPerPixel = 32;

    CKDWORD compressionType = ReadDword() & 0xF; // 0 for raw, 1 for JPEG
    if (compressionType != 0) {
        // Unsupported compression type
        void *compressedData = nullptr;
        ReadBuffer(&compressedData);
        ReadBuffer(&compressedData);
        ReadBuffer(&compressedData);
        return nullptr;
    }

    CKBYTE *bPlane = nullptr;
    CKBYTE *gPlane = nullptr;
    CKBYTE *rPlane = nullptr;
    CKBYTE *aPlane = nullptr;

    int bSize = ReadBuffer((void **) &bPlane);
    int gSize = ReadBuffer((void **) &gPlane);
    int rSize = ReadBuffer((void **) &rPlane);
    int aSize = ReadBuffer((void **) &aPlane);

    if (bSize != static_cast<int>(planeSize) || gSize != static_cast<int>(planeSize) ||
        rSize != static_cast<int>(planeSize)) {
        delete[] bPlane;
        delete[] gPlane;
        delete[] rPlane;
        delete[] aPlane;
        return nullptr;
    }

    CKBYTE *outputBitmap = new CKBYTE[desc.BytesPerLine * desc.Height];
    if (!outputBitmap) {
        delete[] bPlane;
        delete[] gPlane;
        delete[] rPlane;
        delete[] aPlane;
        return nullptr;
    }

    for (int y = 0; y < height; ++y) {
        CKDWORD *rowPtr = reinterpret_cast<CKDWORD *>(outputBitmap + y * desc.BytesPerLine);
        const size_t planeOffset = static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            const size_t index = planeOffset + x;
            CKBYTE r = rPlane[index];
            CKBYTE g = gPlane[index];
            CKBYTE b = bPlane[index];
            CKBYTE a = (aPlane && aSize == static_cast<int>(planeSize)) ? aPlane[index] : 0xFF;

            rowPtr[x] = (CKDWORD) a << A_SHIFT | (CKDWORD) r << R_SHIFT | (CKDWORD) g << G_SHIFT | b;
        }
    }

    desc.AlphaMask = A_MASK;

    delete[] rPlane;
    delete[] gPlane;
    delete[] bPlane;
    delete[] aPlane;

    return outputBitmap;
}

void CKStateChunk::WriteBitmap(BITMAP_HANDLE bitmap, CKSTRING ext) {
    if (!m_ChunkParser)
        return;

    CKFileExtension fileExt(ext);
    CKBitmapReader *reader = CKGetPluginManager()->GetBitmapReader(fileExt, nullptr);
    if (!reader) {
        // Fallback to BMP if the provided extension's reader is not found
        CKFileExtension bmpExt("bmp");
        reader = CKGetPluginManager()->GetBitmapReader(bmpExt, nullptr);
        if (!reader) {
            // If still no reader, write empty data
            WriteInt(0);
            WriteBuffer(0, nullptr);
            return;
        }
        fileExt = bmpExt;
    }

    if (!bitmap) {
        reader->Release();
        WriteInt(0);
        WriteBuffer(0, nullptr);
        return;
    }

    // Prepare the "CKxxx" signature (5 bytes + null terminator)
    char signature[6];
    strcpy(signature, "CK");
    strncat(signature, fileExt.m_Data, 3);
    for (int i = 0; signature[i] && i < 5; ++i) {
        signature[i] = (char) toupper(signature[i]);
    }
    signature[5] = '\0';

    VxImageDescEx desc;
    CKBYTE *imageData = VxConvertBitmap(bitmap, desc);
    if (!imageData) {
        reader->Release();
        WriteInt(0);
        WriteBuffer(0, nullptr);
        return;
    }

    CKBitmapProperties props;
    props.m_Data = imageData;
    props.m_Format = desc;

    void *buffer = nullptr;
    int bufferSize = reader->SaveMemory(&buffer, &props);

    delete[] imageData;

    if (bufferSize > 0 && buffer != nullptr) {
        // Total size to write: signature (5 bytes) + savedMemorySize
        int size = 5 + bufferSize;
        WriteInt(size);
        WriteInt(size);

        // Lock a buffer in the chunk big enough for signature + saved data
        // Size is in DWORDS for LockWriteBuffer
        int dwords = (size + sizeof(CKDWORD) - 1) / sizeof(CKDWORD);
        CKDWORD *chunkBuffer = (CKDWORD *) LockWriteBuffer(dwords);
        if (chunkBuffer) {
            memcpy(chunkBuffer, signature, 5);
            memcpy((char *) chunkBuffer + 5, buffer, bufferSize);
            m_ChunkParser->CurrentPos += dwords;
        } else {
            // LockWriteBuffer failed, major issue.
            // Chunk is likely corrupted from this point.
        }

        reader->ReleaseMemory(buffer);
    } else {
        // SaveMemory failed
        WriteInt(0);
        WriteBuffer(0, nullptr);
    }

    reader->Release();
}

BITMAP_HANDLE CKStateChunk::ReadBitmap() {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize)
        return nullptr;

    void *buffer = nullptr;

    // Read bitmap header information
    ReadInt();
    int bufferSize = ReadBuffer(&buffer);
    if (!buffer || bufferSize < 5) {
        delete[] (CKBYTE *) buffer;
        return nullptr;
    }

    // Detect file format from header
    char signature[6] = {0};
    memcpy(signature, buffer, 5);

    const char *fileExtension = nullptr;
    bool hasCustomHeader = true;

    if (!stricmp(signature, "CKTGA")) {
        fileExtension = "TGA";
    } else if (!stricmp(signature, "CKJPG")) {
        fileExtension = "JPG";
    } else if (!stricmp(signature, "CKDIB") || !stricmp(signature, "CKBMP")) {
        fileExtension = "BMP";
    } else if (!stricmp(signature, "CKDIB")) {
        fileExtension = "TIF";
    } else if (!stricmp(signature, "CKGIF")) {
        fileExtension = "GIF";
    } else if (!stricmp(signature, "CKPCX")) {
        fileExtension = "PCX";
    } else {
        fileExtension = "TGA";
        hasCustomHeader = false;
    }

    // Get appropriate bitmap reader
    CKFileExtension ext(fileExtension);
    CKBitmapReader *reader = CKGetPluginManager()->GetBitmapReader(ext);
    if (!reader) {
        delete[] (CKBYTE *) buffer;
        return nullptr;
    }

    char *dataStart = static_cast<char *>(buffer);
    int dataSize = bufferSize;

    // Skip custom header if present
    if (hasCustomHeader) {
        dataStart += 5;
        dataSize -= 5;
    }

    BITMAP_HANDLE bitmapHandle = nullptr;

    // Read bitmap from memory
    CKBitmapProperties *bitmapProps = nullptr;
    if (reader->ReadMemory(dataStart, dataSize, &bitmapProps) == 0 && bitmapProps) {
        // Create bitmap from decoded data
        bitmapProps->m_Format.Image = (XBYTE *) bitmapProps->m_Data;
        bitmapHandle = VxCreateBitmap(bitmapProps->m_Format);

        // Cleanup reader allocations
        reader->ReleaseMemory(bitmapProps->m_Data);
        reader->ReleaseMemory(bitmapProps->m_Format.ColorMap);
        bitmapProps->m_Data = nullptr;
        bitmapProps->m_Format.ColorMap = nullptr;
    }

    reader->Release();

    delete[] (CKBYTE *) buffer;
    return bitmapHandle;
}

CKBYTE *CKStateChunk::ReadBitmap2(VxImageDescEx &desc) {
    if (!m_ChunkParser || m_ChunkParser->CurrentPos >= m_ChunkSize)
        return nullptr;

    CKBYTE *bitmapData = nullptr;
    void *buffer = nullptr;

    // Read bitmap header information
    ReadInt();
    int bufferSize = ReadBuffer(&buffer);
    if (!buffer || bufferSize < 5) {
        delete[] (CKBYTE *) buffer;
        return nullptr;
    }

    // Detect file format from header
    char signature[6] = {0};
    memcpy(signature, buffer, 5);

    const char *fileExtension = nullptr;
    bool hasCustomHeader = true;

    if (!stricmp(signature, "CKTGA")) {
        fileExtension = "TGA";
    } else if (!stricmp(signature, "CKJPG")) {
        fileExtension = "JPG";
    } else if (!stricmp(signature, "CKDIB") || !stricmp(signature, "CKBMP")) {
        fileExtension = "BMP";
    } else if (!stricmp(signature, "CKDIB")) {
        fileExtension = "TIF";
    } else if (!stricmp(signature, "CKGIF")) {
        fileExtension = "GIF";
    } else if (!stricmp(signature, "CKPCX")) {
        fileExtension = "PCX";
    } else {
        fileExtension = "TGA";
        hasCustomHeader = false;
    }

    // Get appropriate bitmap reader
    CKFileExtension ext(fileExtension);
    CKBitmapReader *reader = CKGetPluginManager()->GetBitmapReader(ext);
    if (!reader) {
        delete[] (CKBYTE*) buffer;
        return nullptr;
    }

    char *dataStart = static_cast<char *>(buffer);
    int dataSize = bufferSize;

    // Skip custom header if present
    if (hasCustomHeader) {
        dataStart += 5;
        dataSize -= 5;
    }

    // Read bitmap from memory
    CKBitmapProperties *bitmapProps = nullptr;
    if (reader->ReadMemory(dataStart, dataSize, &bitmapProps) == 0 && bitmapProps) {
        desc = bitmapProps->m_Format;
        bitmapData = (CKBYTE *) bitmapProps->m_Data;
        bitmapProps->m_Data = nullptr;
    }

    reader->Release();

    delete[] (CKBYTE *) buffer;
    return bitmapData;
}

CKDWORD CKStateChunk::ComputeCRC(CKDWORD adler) {
    return adler32(adler, (unsigned char *) m_Data, m_ChunkSize * sizeof(int));
}

void CKStateChunk::Pack(int CompressionLevel) {
    uLongf srcSize = m_ChunkSize * sizeof(int);
    uLongf destSize = srcSize;
    Bytef *buf = new Bytef[destSize];
    if (!buf) return;

    if (compress2(buf, &destSize, (const Bytef *) m_Data, srcSize, CompressionLevel) == Z_OK) {
        // Allocate just enough memory for the compressed data
        Bytef *data = new Bytef[destSize];
        if (!data) {
            delete[] buf;
            return;
        }
        // Copy from buf (compressed data), not m_Data
        memcpy(data, buf, destSize);
        delete[] m_Data;
        m_Data = (int *) data;
        // Size should be in ints, not bytes
        m_ChunkSize = (destSize + sizeof(int) - 1) / sizeof(int);
    }
    delete[] buf;
}

CKBOOL CKStateChunk::UnPack(int DestSize) {
    if (DestSize <= 0) return FALSE;

    uLongf size = DestSize;
    Bytef *buf = new Bytef[DestSize];
    if (!buf) return FALSE;

    int err = uncompress(buf, &size, (const Bytef *) m_Data, m_ChunkSize * sizeof(int));
    if (err == Z_OK) {
        int sz = (DestSize + sizeof(int) - 1) / sizeof(int); // Ceiling division
        int *data = new int[sz];
        if (!data) {
            delete[] buf;
            return FALSE;
        }

        memcpy(data, buf, DestSize);
        delete[] m_Data;
        m_Data = data;
        m_ChunkSize = sz;
    }
    delete[] buf;
    return err == Z_OK;
}

void CKStateChunk::AttributePatch(CKBOOL preserveData, int *ConversionTable, int NbEntries) {
    // Skip processing if managers array already exists
    if (m_Managers)
        return;

    // Read sequence length and store current position
    int sequenceLength = StartReadSequence();
    int initialPosition = GetCurrentPos();

    // Skip over the sequence data
    Skip(sequenceLength);

    if (!preserveData) {
        // Clean up subchunks if not preserving data
        StartReadSequence();
        while (sequenceLength-- > 0) {
            CKStateChunk *subChunk = ReadSubChunk();
            if (subChunk) {
                delete subChunk;
            }
        }
    }

    // Initialize managers list
    m_Managers = new IntListStruct();
    m_Managers->AddEntries(initialPosition);

    // Prepare for conversion
    Skip(sizeof(CKDWORD) * 3); // Skip header data

    if (ConversionTable && NbEntries > 0 && sequenceLength > 0) {
        do {
            int *currentPosPtr = &m_Data[m_ChunkParser->CurrentPos];
            int originalValue = *currentPosPtr;

            // Check if value needs conversion
            if (!(originalValue & 0x80000000) && originalValue < NbEntries) {
                *currentPosPtr = ConversionTable[originalValue];
            }

            m_ChunkParser->CurrentPos++;
        } while (--sequenceLength);
    }

    // Reset chunk position to attribute section
    SeekIdentifier(CK_STATESAVE_NEWATTRIBUTES);
}

int CKStateChunk::IterateAndDo(ChunkIterateFct fct, ChunkIteratorData *it) {
    if (!it) return 0;

    // 1. Apply the function to the current chunk level first.
    int totalResult = fct(it);

    // 2. If there are no sub-chunks to process, we are done with this level.
    if (!it->Chunks || it->ChunkCount <= 0) {
        return totalResult;
    }

    // 3. Iterate through the list of sub-chunks.
    // The m_Chunks->Data array contains offsets into the parent's m_Data buffer.
    int chunkListIdx = 0;
    while (chunkListIdx < it->ChunkCount) {
        ChunkIteratorData localIt;
        localIt.CopyFctData(it); // Inherit context (mappers, etc.) from parent.

        int currentChunkInfoOffset = it->Chunks[chunkListIdx];

        // LOGIC: A negative offset (-1) marks the start of a "packed sequence" of sub-chunks.
        // A positive offset points to a single, standalone sub-chunk.

        if (currentChunkInfoOffset >= 0) {
            // ---===[ PATH A: Parsing a single, standalone sub-chunk ]===---

            // Boundary check for the sub-chunk header itself.
            // A header needs at least 8 dwords.
            if (currentChunkInfoOffset + 8 > it->ChunkSize) {
                chunkListIdx++;
                continue;
            }

            int *subChunkHeaderPtr = it->Data + currentChunkInfoOffset;

            // The total size of this sub-chunk block, in dwords.
            const int subChunkBlockTotalDwords = subChunkHeaderPtr[0] + 1;

            // This condition from the decompilation (Data[v6 + 2] + 4 != v18) is likely a flawed or
            // misinterpreted validity check. A robust boundary check is a safer replacement.
            if (currentChunkInfoOffset + subChunkBlockTotalDwords > it->ChunkSize) {
                // The chunk claims to be larger than the remaining parent buffer.
                chunkListIdx++;
                continue;
            }

            // Extract sub-chunk metadata from its header.
            // Field [0] is total size, [1] is ClassID, [2] is versions, etc.
            localIt.ChunkVersion = static_cast<short>(subChunkHeaderPtr[2] >> 16);
            localIt.ChunkSize = subChunkHeaderPtr[3];
            localIt.IdCount = subChunkHeaderPtr[5];
            localIt.ChunkCount = subChunkHeaderPtr[6];

            int dataPtrOffsetFromHeaderStart;
            if (it->ChunkVersion > CHUNK_VERSION2) {
                // CHUNK_VERSION2 is 5
                localIt.ManagerCount = subChunkHeaderPtr[7];
                dataPtrOffsetFromHeaderStart = 8;
            } else {
                localIt.ManagerCount = 0;
                dataPtrOffsetFromHeaderStart = 7;
            }

            // Set up pointers to the data slices within the parent buffer.
            localIt.Data = subChunkHeaderPtr + dataPtrOffsetFromHeaderStart;
            int *currentDataSectionPtr = localIt.Data + localIt.ChunkSize;

            localIt.Ids = (localIt.IdCount > 0) ? currentDataSectionPtr : nullptr;
            if (localIt.Ids) currentDataSectionPtr += localIt.IdCount;

            localIt.Chunks = (localIt.ChunkCount > 0) ? currentDataSectionPtr : nullptr;
            if (localIt.Chunks) currentDataSectionPtr += localIt.ChunkCount;

            localIt.Managers = (localIt.ManagerCount > 0) ? currentDataSectionPtr : nullptr;

            // Recursively call this function for the sub-chunk.
            localIt.Flag = 0; // Mark that this is a sub-level call.
            totalResult += IterateAndDo(fct, &localIt);

            chunkListIdx++;
        } else {
            // ---===[ PATH B: Parsing a packed sequence of sub-chunks ]===---

            // The -1 is a marker. The next entry in m_Chunks has the real offset.
            chunkListIdx++;
            if (chunkListIdx >= it->ChunkCount) break;

            int sequenceDataActualOffset = it->Chunks[chunkListIdx];
            if (sequenceDataActualOffset < 0 || sequenceDataActualOffset >= it->ChunkSize) {
                chunkListIdx++;
                continue;
            }

            // This cursor points to the start of the sequence data.
            int *packedCursor = it->Data + sequenceDataActualOffset;

            // The first dword of the sequence is the number of sub-chunks in it.
            const int numSubChunksInPack = *packedCursor++;

            for (int i = 0; i < numSubChunksInPack; ++i) {
                // Boundary check before reading next chunk's size.
                if ((packedCursor - it->Data) >= it->ChunkSize) break;

                const int currentPackedSubBlockDwords = *packedCursor++;
                if (currentPackedSubBlockDwords <= 0) continue;

                // Boundary check for the entire sub-chunk block.
                if ((packedCursor - it->Data) + currentPackedSubBlockDwords > it->ChunkSize) break;

                // The logic for parsing the header is identical to the standalone path.
                int *packedSubChunkHeaderPtr = packedCursor;

                // Advance cursor past this sub-chunk's data to be ready for the next iteration.
                packedCursor += currentPackedSubBlockDwords;

                // Same validity check and parsing as in PATH A
                int requiredPackedHeaderDwords = (it->ChunkVersion > CHUNK_VERSION2) ? 7 : 6;
                if (currentPackedSubBlockDwords < requiredPackedHeaderDwords) continue;

                localIt.ChunkVersion = static_cast<short>(packedSubChunkHeaderPtr[1] >> 16);
                localIt.ChunkSize = packedSubChunkHeaderPtr[2];
                localIt.IdCount = packedSubChunkHeaderPtr[4];
                localIt.ChunkCount = packedSubChunkHeaderPtr[5];

                int dataPtrOffsetFromPackedHeaderStart;
                if (it->ChunkVersion > CHUNK_VERSION2) {
                    localIt.ManagerCount = packedSubChunkHeaderPtr[6];
                    dataPtrOffsetFromPackedHeaderStart = 7;
                } else {
                    localIt.ManagerCount = 0;
                    dataPtrOffsetFromPackedHeaderStart = 6;
                }

                // Setup pointers to data slices inside this packed sub-chunk
                localIt.Data = packedSubChunkHeaderPtr + dataPtrOffsetFromPackedHeaderStart;
                int *currentPackedDataSectionPtr = localIt.Data + localIt.ChunkSize;

                localIt.Ids = (localIt.IdCount > 0) ? currentPackedDataSectionPtr : nullptr;
                if (localIt.Ids) currentPackedDataSectionPtr += localIt.IdCount;

                localIt.Chunks = (localIt.ChunkCount > 0) ? currentPackedDataSectionPtr : nullptr;
                if (localIt.Chunks) currentPackedDataSectionPtr += localIt.ChunkCount;

                localIt.Managers = (localIt.ManagerCount > 0) ? currentPackedDataSectionPtr : nullptr;

                // Recursively call for this sub-chunk within the sequence.
                localIt.Flag = 0;
                totalResult += IterateAndDo(fct, &localIt);
            }
            chunkListIdx++;
        }
    }
    return totalResult;
}

int CKStateChunk::ObjectRemapper(ChunkIteratorData *it) {
    if (!it || !it->Data) return 0;
    int remappedCount = 0;

    // ---===[ PATH A: Modern Chunk Format (Version >= 4) ]===---
    // This version uses the m_Ids list which stores offsets to all object IDs.
    if (it->ChunkVersion >= CHUNK_VERSION1) {
        if (!it->Ids || it->IdCount <= 0)
            return 0;

        for (int i = 0; i < it->IdCount;) {
            int idOffsetInData = it->Ids[i];
            if (idOffsetInData >= 0) {
                // -- Single Object ID --
                if (idOffsetInData < it->ChunkSize) {
                    CK_ID &oldIdRef = reinterpret_cast<CK_ID &>(it->Data[idOffsetInData]);
                    CK_ID oldIdVal = oldIdRef;
                    if (oldIdVal != 0) {
                        CK_ID newId = 0;
                        if (it->DepContext) {
                            XHashID::Iterator mapIt = it->DepContext->m_MapID.Find(oldIdVal);
                            if (mapIt != it->DepContext->m_MapID.End())
                                newId = *mapIt;
                        } else if (it->Context && it->Context->m_ObjectManager) {
                            newId = it->Context->m_ObjectManager->RealId(oldIdVal);
                        }
                        if (newId != 0 && newId != oldIdVal) {
                            oldIdRef = newId;
                            remappedCount++;
                        }
                    }
                }
                i++;
            } else {
                // -- Sequence of Object IDs --
                i++; // Move to the offset of the sequence count
                if (i >= it->IdCount)
                    break;

                int sequenceHeaderOffset = it->Ids[i];
                if (sequenceHeaderOffset >= 0 && sequenceHeaderOffset < it->ChunkSize) {
                    int count = it->Data[sequenceHeaderOffset];
                    int sequenceStartOffset = sequenceHeaderOffset + 1;

                    if (count > 0 && sequenceStartOffset + count <= it->ChunkSize) {
                        for (int k = 0; k < count; ++k) {
                            CK_ID &oldIdRef = reinterpret_cast<CK_ID &>(it->Data[sequenceHeaderOffset + 1 + k]);
                            CK_ID oldIdVal = oldIdRef;
                            if (oldIdVal != 0) {
                                CK_ID newId = 0;
                                if (it->DepContext) {
                                    XHashID::Iterator mapIt = it->DepContext->m_MapID.Find(oldIdVal);
                                    if (mapIt != it->DepContext->m_MapID.End()) newId = *mapIt;
                                } else if (it->Context && it->Context->m_ObjectManager) {
                                    newId = it->Context->m_ObjectManager->RealId(oldIdVal);
                                }
                                if (newId != 0 && newId != oldIdVal) {
                                    oldIdRef = newId;
                                    remappedCount++;
                                }
                            }
                        }
                    }
                }
                i++; // Move past the sequence offset
            }
        }
    } else {
        // ---===[ PATH B: Legacy Chunk Format (Version < 4) ]===---
        // This version requires scanning the entire data buffer for magic number markers.
        const CKDWORD OBJID_MARKER[] = {0xE32BC4C9, 0x134212E3, 0xFCBAE9DC};
        const CKDWORD SEQ_MARKER[] = {0xE192BD47, 0x13246628, 0x13EAB3CE, 0x7891AEFC, 0x13984562};

        for (int pos = 0; pos < it->ChunkSize; ++pos) {
            // Check for single object ID marker
            if (pos + 3 < it->ChunkSize && memcmp(&it->Data[pos], OBJID_MARKER, sizeof(OBJID_MARKER)) == 0) {
                CK_ID &oldIdRef = reinterpret_cast<CK_ID &>(it->Data[pos + 3]);
                CK_ID oldIdVal = oldIdRef;
                if (oldIdVal != 0) {
                    CK_ID newId = 0;
                    if (it->DepContext) {
                        XHashID::Iterator mapIt = it->DepContext->m_MapID.Find(oldIdVal);
                        if (mapIt != it->DepContext->m_MapID.End()) newId = *mapIt;
                    } else if (it->Context && it->Context->m_ObjectManager) {
                        newId = it->Context->m_ObjectManager->RealId(oldIdVal);
                    }
                    if (newId != 0 && newId != oldIdVal) {
                        oldIdRef = newId;
                        remappedCount++;
                    }
                }
                pos += 3; // Advance past marker and ID
                continue;
            }

            // Check for object ID sequence marker
            if (pos + 5 < it->ChunkSize && memcmp(&it->Data[pos], SEQ_MARKER, sizeof(SEQ_MARKER)) == 0) {
                int count = it->Data[pos + 5];
                int sequenceStartOffset = pos + 6;

                if (count > 0 && sequenceStartOffset + count <= it->ChunkSize) {
                    for (int k = 0; k < count; ++k) {
                        CK_ID &oldIdRef = reinterpret_cast<CK_ID &>(it->Data[pos + 6 + k]);
                        CK_ID oldIdVal = oldIdRef;
                        if (oldIdVal != 0) {
                            CK_ID newId = 0;
                            if (it->DepContext) {
                                XHashID::Iterator mapIt = it->DepContext->m_MapID.Find(oldIdVal);
                                if (mapIt != it->DepContext->m_MapID.End()) newId = *mapIt;
                            } else if (it->Context && it->Context->m_ObjectManager) {
                                newId = it->Context->m_ObjectManager->RealId(oldIdVal);
                            }
                            if (newId != 0 && newId != oldIdVal) {
                                oldIdRef = newId;
                                remappedCount++;
                            }
                        }
                    }
                    pos = sequenceStartOffset + count - 1; // Position 'pos' for next loop iteration
                } else {
                    pos += 5; // Skip marker
                }
            }
        }
    }

    return remappedCount;
}

int CKStateChunk::ManagerRemapper(ChunkIteratorData *it) {
    if (!it || !it->Managers || it->ManagerCount <= 0 || !it->Data || !it->ConversionTable || it->NbEntries <= 0)
        return 0;

    int remappedCount = 0;
    for (int i = 0; i < it->ManagerCount;) {
        int managerDataOffset = it->Managers[i];
        if (managerDataOffset >= 0) {
            // -- Single Manager Integer --
            // Header is [GUID.d1, GUID.d2, Value]
            if (managerDataOffset + 2 < it->ChunkSize) {
                CKGUID storedGuid(it->Data[managerDataOffset], it->Data[managerDataOffset + 1]);
                if (it->Guid == storedGuid) {
                    int &valRef = it->Data[managerDataOffset + 2];
                    if (valRef >= 0 && valRef < it->NbEntries) {
                        int newVal = it->ConversionTable[valRef];
                        if (newVal != valRef) {
                            valRef = newVal;
                            remappedCount++;
                        }
                    }
                }
            }
            i++;
        } else {
            // -- Sequence of Manager Integers --
            i++; // Move to the offset of the sequence header
            if (i >= it->ManagerCount)
                break;

            int sequenceHeaderOffset = it->Managers[i];
            // Header is [Count, GUID.d1, GUID.d2], followed by integers
            if (sequenceHeaderOffset >= 0 && sequenceHeaderOffset + 2 < it->ChunkSize) {
                int count = it->Data[sequenceHeaderOffset];
                int sequenceStartOffset = sequenceHeaderOffset + 3;

                if (count > 0 && sequenceStartOffset + count <= it->ChunkSize) {
                    CKGUID storedGuid(it->Data[sequenceHeaderOffset + 1], it->Data[sequenceHeaderOffset + 2]);
                    if (it->Guid == storedGuid) {
                        for (int k = 0; k < count; ++k) {
                            int &valRef = it->Data[sequenceStartOffset + k];
                            if (valRef >= 0 && valRef < it->NbEntries) {
                                int newVal = it->ConversionTable[valRef];
                                if (newVal != valRef) {
                                    valRef = newVal;
                                    remappedCount++;
                                }
                            }
                        }
                    }
                }
            }
            i++; // Move past the sequence offset
        }
    }
    return remappedCount;
}

int CKStateChunk::ParameterRemapper(ChunkIteratorData *it) {
    if (!it || !it->Data || it->ChunkSize == 0 || !it->ConversionTable)
        return 0;

    int currentPos = 0;
    while (currentPos + 1 < it->ChunkSize) {
        // Search for the start of a parameter data block (Identifier 0x40)
        if (it->Data[currentPos] == 0x40) {
            int headerStart = currentPos + 2;
            // A parameter header has [GUID.d1, GUID.d2, State, ...]
            if (headerStart + 3 < it->ChunkSize) {
                CKGUID storedGuid(it->Data[headerStart], it->Data[headerStart + 1]);
                int paramState = it->Data[headerStart + 2];

                // Check if it's the target parameter type and if it holds raw data (State 1)
                if (it->Guid == storedGuid && paramState == 1) {
                    int &valRef = it->Data[headerStart + 3];
                    if (valRef >= 0 && valRef < it->NbEntries) {
                        int newVal = it->ConversionTable[valRef];
                        if (newVal != valRef) {
                            valRef = newVal;
                            return 1; // Remapped one value
                        }
                    }
                }
            }
        }

        // Move to the next data block within the chunk
        int nextPos = it->Data[currentPos + 1];
        if (nextPos == 0 || nextPos <= currentPos || nextPos >= it->ChunkSize) {
            // Reached end of linked list of blocks or invalid offset
            break;
        }
        currentPos = nextPos;
    }

    return 0; // Nothing remapped
}
