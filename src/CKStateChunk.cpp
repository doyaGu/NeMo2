#include "CKStateChunk.h"

#include "VxMatrix.h"
#include "VxWindowFunctions.h"
#include "VxMath.h"
#include "CKFile.h"
#include "CKPluginManager.h"
#include "CKJpegDecoder.h"

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
    // IDA: m_Dynamic is NOT copied in Clone
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

    // IDA: chunka starts as CurrentPos, then follows the linked list
    int j = m_ChunkParser->CurrentPos;
    if (m_Data[j + 1] != 0) {
        do {
            int *p = &m_Data[j + 1];
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
        // IDA: Copy ChunkParser contents (not pointer), then zero source
        if (chunk->m_ChunkParser) {
            memcpy(m_ChunkParser, chunk->m_ChunkParser, sizeof(ChunkParser));
            memset(chunk->m_ChunkParser, 0, sizeof(ChunkParser));
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
    // IDA: CheckSize(4 * count + 12) - space for count + guid(2) + data
    CheckSize(4 * count + 12);
    m_Data[m_ChunkParser->CurrentPos++] = count;
    m_Data[m_ChunkParser->CurrentPos++] = man.d1;
    m_Data[m_ChunkParser->CurrentPos++] = man.d2;
}

void CKStateChunk::WriteManagerSequence(int val) {
    m_Data[m_ChunkParser->CurrentPos++] = val;
}

void CKStateChunk::WriteManagerInt(CKGUID Manager, int val) {
    // IDA: CheckSize called BEFORE checking m_Managers
    CheckSize(12);  // guid(2) + value = 3 ints = 12 bytes
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
    // IDA: AddEntries uses CurrentPos (not CurrentPos-1)
    m_Chunks->AddEntries(m_ChunkParser->CurrentPos);
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
        // IDA: CheckSize(4) for null buffer
        CheckSize(sizeof(int));
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
    int sz = (size + 3) / sizeof(int);
    ++m_ChunkParser->CurrentPos;
    if (size > 0) {
        // IDA: only copy if buffer is not null
        if (buffer)
            memcpy(buffer, &m_Data[m_ChunkParser->CurrentPos], size);
    }
    // IDA: Always advance CurrentPos even if size <= 0
    m_ChunkParser->CurrentPos += sz;
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
    // Base size: header (2 DWORDs) + data
    int size = 2 * sizeof(int) + m_ChunkSize * sizeof(int);
    CKBYTE chunkOptions = 0;

    // Each list adds: size count (1 DWORD) + list data
    if (m_Ids) {
        size += sizeof(int) + m_Ids->Size * sizeof(int);  // +4 for count
        chunkOptions |= CHNK_OPTION_IDS;
    }
    if (m_Chunks) {
        size += sizeof(int) + m_Chunks->Size * sizeof(int);  // +4 for count
        chunkOptions |= CHNK_OPTION_CHN;
    }
    if (m_Managers) {
        size += sizeof(int) + m_Managers->Size * sizeof(int);  // +4 for count
        chunkOptions |= CHNK_OPTION_MAN;
    }
    if (m_File) {
        chunkOptions |= CHNK_OPTION_FILE;
    }

    if (buffer) {
        int *buf = (int *) buffer;

        // Temporarily OR options/classID into version fields for serialization
        short savedChunkVersion = m_ChunkVersion;
        short savedDataVersion = m_DataVersion;
        m_ChunkVersion |= (chunkOptions << 8);
        m_DataVersion |= (m_ChunkClassID << 8);

        // Write header: [DataVersion | ChunkVersion<<16]
        *(short *)buf = m_DataVersion;
        *((short *)buf + 1) = m_ChunkVersion;
        buf[1] = m_ChunkSize;

        // Copy main data
        memcpy(&buf[2], m_Data, m_ChunkSize * sizeof(int));
        int *pos = &buf[m_ChunkSize + 2];

        // Write lists with size prefix
        if (m_Ids) {
            *pos++ = m_Ids->Size;
            memcpy(pos, m_Ids->Data, m_Ids->Size * sizeof(int));
            pos += m_Ids->Size;
        }
        if (m_Chunks) {
            *pos++ = m_Chunks->Size;
            memcpy(pos, m_Chunks->Data, m_Chunks->Size * sizeof(int));
            pos += m_Chunks->Size;
        }
        if (m_Managers) {
            *pos++ = m_Managers->Size;
            memcpy(pos, m_Managers->Data, m_Managers->Size * sizeof(int));
            pos += m_Managers->Size;
        }

        // Restore original version fields
        m_ChunkVersion = savedChunkVersion;
        m_DataVersion = savedDataVersion;
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
    // IDA: early return if no conversion table OR no managers/chunks to process
    if (!ConversionTable || (!m_Managers && !m_Chunks))
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

void CKStateChunk::WriteReaderBitmap(const VxImageDescEx &desc, CKBitmapReader *reader, CKBitmapProperties *bp) {
    if (!m_ChunkParser)
        return;

    if (!desc.Image || !reader || !bp) {
        WriteDword(0);
        return;
    }

    CKBOOL alphaIsSaved = reader->IsAlphaSaved(bp);
    void *savedBuffer = nullptr;
    CKBYTE *converted32Bits = nullptr;

    if (desc.BitsPerPixel == 32) {
        bp->m_Format = desc;
        bp->m_Data = desc.Image;
    } else {
        VxImageDescEx tempDesc = {};
        tempDesc.Width = desc.Width;
        tempDesc.Height = desc.Height;
        tempDesc.BitsPerPixel = 32;
        tempDesc.BytesPerLine = desc.Width * 4;
        tempDesc.RedMask = R_MASK;
        tempDesc.GreenMask = G_MASK;
        tempDesc.BlueMask = B_MASK;
        tempDesc.AlphaMask = A_MASK;

        const size_t convertedSize = static_cast<size_t>(tempDesc.Height) * tempDesc.BytesPerLine;
        converted32Bits = (convertedSize > 0) ? new CKBYTE[convertedSize] : nullptr;
        if (!converted32Bits) {
            WriteDword(0);
            return;
        }
        tempDesc.Image = converted32Bits;
        VxDoBlit(desc, tempDesc);
        bp->m_Format = tempDesc;
        bp->m_Data = converted32Bits;
    }

    const size_t savedSize = reader->SaveMemory(&savedBuffer, bp);
    delete[] converted32Bits;

    if (!savedBuffer || savedSize == 0) {
        WriteDword(0);
        if (savedBuffer)
            reader->ReleaseMemory(savedBuffer);
        return;
    }

    if (alphaIsSaved || !desc.AlphaMask) {
        WriteDword(1);
        WriteBufferNoSize_LEndian(sizeof(CKFileExtension), bp->m_Ext.m_Data);
        WriteGuid(bp->m_ReaderGuid);
        WriteBuffer_LEndian(static_cast<int>(savedSize), savedBuffer);
        reader->ReleaseMemory(savedBuffer);
        return;
    }

    WriteDword(2);
    CKDWORD extensionValue = 0;
    memcpy(&extensionValue, bp->m_Ext.m_Data, sizeof(extensionValue));
    WriteDword(extensionValue);
    WriteGuid(bp->m_ReaderGuid);
    WriteBuffer_LEndian(static_cast<int>(savedSize), savedBuffer);

    CKBYTE alphaHistogram[256];
    memset(alphaHistogram, 0, sizeof(alphaHistogram));

    CalculateMaskShifts(desc.RedMask, desc.GreenMask, desc.BlueMask, desc.AlphaMask);
    const CKDWORD alphaMask = desc.AlphaMask;
    const int bytesPerLine = desc.BytesPerLine;
    int bytesPerPixel = desc.BitsPerPixel >> 3;
    if (bytesPerPixel <= 0)
        bytesPerPixel = 1;

    int totalDwords = (desc.Height * bytesPerLine) >> 2;
    CKBYTE *pixelCursor = desc.Image;
    const int lutShift = static_cast<int>(g_AlphaShift_LSB) - static_cast<int>(g_AlphaQuantShift_MSB);
    while (totalDwords-- > 0) {
        CKDWORD packedPixel = *reinterpret_cast<const CKDWORD *>(pixelCursor);
        CKBYTE histogramIndex = static_cast<CKBYTE>((packedPixel & alphaMask) >> lutShift);
        alphaHistogram[histogramIndex] = 1;
        pixelCursor += bytesPerPixel;
    }

    int distinctAlpha = 0;
    for (int i = 0; i < 256; ++i)
        distinctAlpha += alphaHistogram[i];
    WriteInt(distinctAlpha);

    if (distinctAlpha == 1) {
        for (int i = 0; i < 256; ++i) {
            if (alphaHistogram[i]) {
                WriteInt(i);
                break;
            }
        }
        reader->ReleaseMemory(savedBuffer);
        return;
    }

    const size_t planeSize = static_cast<size_t>(desc.Width) * desc.Height;
    CKBYTE *alphaPlane = (planeSize > 0) ? new CKBYTE[planeSize] : nullptr;
    if (alphaPlane) {
        if (desc.BitsPerPixel == 32) {
            CKBYTE *srcRow = desc.Image;
            for (int y = 0; y < desc.Height; ++y) {
                CKBYTE *dst = alphaPlane + static_cast<size_t>(y) * desc.Width;
                CKBYTE *src = srcRow + 3;
                for (int x = 0; x < desc.Width; ++x) {
                    dst[x] = *src;
                    src += bytesPerPixel;
                }
                srcRow += desc.BytesPerLine;
            }
        } else {
            CKBYTE *srcRow = desc.Image;
            for (int y = 0; y < desc.Height; ++y) {
                CKBYTE *dst = alphaPlane + static_cast<size_t>(y) * desc.Width;
                CKBYTE *src = srcRow;
                for (int x = 0; x < desc.Width; ++x) {
                    CKDWORD packedPixel = 0;
                    memcpy(&packedPixel, src, bytesPerPixel);
                    dst[x] = static_cast<CKBYTE>(((packedPixel & alphaMask) >> g_AlphaShift_LSB) << g_AlphaQuantShift_MSB);
                    src += bytesPerPixel;
                }
                srcRow -= desc.BytesPerLine;
            }
        }
        WriteBuffer_LEndian(static_cast<int>(planeSize), alphaPlane);
        delete[] alphaPlane;
    } else {
        WriteBuffer_LEndian(0, nullptr);
    }

    reader->ReleaseMemory(savedBuffer);
}

CKBOOL CKStateChunk::ReadReaderBitmap(const VxImageDescEx &desc) {
    CKDWORD formatType = ReadDword();
    if (formatType == 0 || !desc.Image)
        return FALSE;

    CKFileExtension ext;
    memset(ext.m_Data, 0, sizeof(ext.m_Data));
    ReadAndFillBuffer_LEndian(sizeof(CKFileExtension), ext.m_Data);
    CKGUID readerGuid = ReadGuid();

    CKBitmapReader *reader = CKGetPluginManager()->GetBitmapReader(ext, &readerGuid);
    if (!reader)
        return FALSE;

    CKBitmapProperties *props = nullptr;
    int size = ReadInt();
    int dwordCount = (size + 3) / sizeof(int);
    if (!m_ChunkParser || m_ChunkParser->CurrentPos + dwordCount > m_ChunkSize) {
        reader->Release();
        return FALSE;
    }

    if (size > 0) {
        if (reader->ReadMemory(&m_Data[m_ChunkParser->CurrentPos], size, &props) != 0) {
            reader->Release();
            return FALSE;
        }

        if (props) {
            props->m_Format.Image = (CKBYTE *) props->m_Data;
            VxDoBlit(props->m_Format, desc);

            reader->ReleaseMemory(props->m_Data);
            props->m_Data = nullptr;
            delete[] (CKBYTE *) props->m_Format.ColorMap;
            props->m_Format.ColorMap = nullptr;
        }
    }
    Skip(dwordCount);

    if (formatType == 2) {
        int distinctAlphaCount = ReadInt();
        if (distinctAlphaCount == 1) {
            CKBYTE alpha = static_cast<CKBYTE>(ReadDword());
            VxDoAlphaBlit(desc, alpha);
        } else {
            void *buffer = nullptr;
            if (ReadBuffer(&buffer) > 0 && buffer) {
                VxDoAlphaBlit(desc, (CKBYTE *) buffer);
                delete[] (CKBYTE *) buffer;
            }
        }
    }

    reader->Release();
    return TRUE;
}

void CKStateChunk::WriteRawBitmap(const VxImageDescEx &desc) {
    if (!m_ChunkParser)
        return;

    if (!desc.Image) {
        WriteInt(0);
        return;
    }

    const int bitsPerPixel = desc.BitsPerPixel;
    const int width = desc.Width;
    const int height = desc.Height;

    WriteInt(bitsPerPixel);
    WriteInt(width);
    WriteInt(height);
    WriteDword(desc.AlphaMask);
    WriteDword(desc.RedMask);
    WriteDword(desc.GreenMask);
    WriteDword(desc.BlueMask);
    const int compressionFlagPos = m_ChunkParser->CurrentPos;
    WriteDword(0); // Will be patched to 1 if JPEG compression succeeds

    const size_t planeSize = static_cast<size_t>(width) * static_cast<size_t>(height);
    CKBYTE *bluePlane = new CKBYTE[planeSize];
    CKBYTE *greenPlane = new CKBYTE[planeSize];
    CKBYTE *redPlane = new CKBYTE[planeSize];
    CKBYTE *alphaPlane = (desc.AlphaMask != 0) ? new CKBYTE[planeSize] : nullptr;

    if (!bluePlane || !greenPlane || !redPlane || (desc.AlphaMask && !alphaPlane)) {
        delete[] bluePlane;
        delete[] greenPlane;
        delete[] redPlane;
        delete[] alphaPlane;
        WriteBuffer_LEndian(0, nullptr);
        WriteBuffer_LEndian(0, nullptr);
        WriteBuffer_LEndian(0, nullptr);
        WriteBuffer_LEndian(0, nullptr);
        return;
    }

    const int bytesPerPixel = bitsPerPixel >> 3;
    const CKBYTE *scanline = desc.Image + desc.BytesPerLine * (height - 1);
    CKBYTE *blueOut = bluePlane;
    CKBYTE *greenOut = greenPlane;
    CKBYTE *redOut = redPlane;
    CKBYTE *alphaOut = alphaPlane;

    if (bitsPerPixel == 32) {
        if (alphaPlane) {
            for (int remaining = height; remaining > 0; --remaining) {
                const CKBYTE *pixel = scanline;
                for (int x = 0; x < width; ++x) {
                    *blueOut++ = pixel[0];
                    *greenOut++ = pixel[1];
                    *redOut++ = pixel[2];
                    *alphaOut++ = pixel[3];
                    pixel += bytesPerPixel;
                }
                scanline -= desc.BytesPerLine;
            }
        } else {
            for (int remaining = height; remaining > 0; --remaining) {
                const CKBYTE *pixel = scanline;
                for (int x = 0; x < width; ++x) {
                    *blueOut++ = pixel[0];
                    *greenOut++ = pixel[1];
                    *redOut++ = pixel[2];
                    pixel += bytesPerPixel;
                }
                scanline -= desc.BytesPerLine;
            }
        }
    } else if (bitsPerPixel < 24) {
        CalculateMaskShifts(desc.RedMask, desc.GreenMask, desc.BlueMask, desc.AlphaMask);
        const CKDWORD redLsb = g_RedShift_LSB;
        const CKDWORD redQuant = g_RedQuantShift_MSB;
        const CKDWORD greenLsb = g_GreenShift_LSB;
        const CKDWORD greenQuant = g_GreenQuantShift_MSB;
        const CKDWORD blueLsb = g_BlueShift_LSB;
        const CKDWORD blueQuant = g_BlueQuantShift_MSB;
        const CKDWORD alphaLsb = g_AlphaShift_LSB;
        const CKDWORD alphaQuant = g_AlphaQuantShift_MSB;

        for (int remaining = height; remaining > 0; --remaining) {
            const CKBYTE *pixel = scanline;
            for (int x = 0; x < width; ++x) {
                CKDWORD sample = 0;
                memcpy(&sample, pixel, bytesPerPixel);
                *blueOut++ = static_cast<CKBYTE>(((sample & desc.BlueMask) >> blueLsb) << blueQuant);
                *greenOut++ = static_cast<CKBYTE>(((sample & desc.GreenMask) >> greenLsb) << greenQuant);
                *redOut++ = static_cast<CKBYTE>(((sample & desc.RedMask) >> redLsb) << redQuant);
                if (alphaPlane)
                    *alphaOut++ = static_cast<CKBYTE>(((sample & desc.AlphaMask) >> alphaLsb) << alphaQuant);
                pixel += bytesPerPixel;
            }
            scanline -= desc.BytesPerLine;
        }
    } else {
        for (int remaining = height; remaining > 0; --remaining) {
            const CKBYTE *pixel = scanline;
            for (int x = 0; x < width; ++x) {
                *blueOut++ = pixel[0];
                *greenOut++ = pixel[1];
                *redOut++ = pixel[2];
                pixel += bytesPerPixel;
            }
            scanline -= desc.BytesPerLine;
        }
    }

    CKBYTE *encodedBlue = nullptr;
    CKBYTE *encodedGreen = nullptr;
    CKBYTE *encodedRed = nullptr;
    int encodedBlueSize = 0;
    int encodedGreenSize = 0;
    int encodedRedSize = 0;
    bool compressionUsed = false;

    const bool canAttemptCompression = (planeSize > 0) &&
                                       (planeSize <= static_cast<size_t>(INT_MAX)) &&
                                       (width > 0 && height > 0);
    if (canAttemptCompression) {
        const int jpegQuality = 85;
        bool encodeOk = CKJpegDecoder::EncodeGrayscalePlane(bluePlane, width, height, jpegQuality, &encodedBlue, encodedBlueSize);
        if (encodeOk)
            encodeOk = CKJpegDecoder::EncodeGrayscalePlane(greenPlane, width, height, jpegQuality, &encodedGreen, encodedGreenSize);
        if (encodeOk)
            encodeOk = CKJpegDecoder::EncodeGrayscalePlane(redPlane, width, height, jpegQuality, &encodedRed, encodedRedSize);

        if (encodeOk) {
            compressionUsed = true;
        } else {
            delete[] encodedBlue;
            delete[] encodedGreen;
            delete[] encodedRed;
            encodedBlue = encodedGreen = encodedRed = nullptr;
            encodedBlueSize = encodedGreenSize = encodedRedSize = 0;
        }
    }

    if (compressionUsed) {
        m_Data[compressionFlagPos] = 1;
        WriteBuffer_LEndian(encodedBlueSize, encodedBlue);
        WriteBuffer_LEndian(encodedGreenSize, encodedGreen);
        WriteBuffer_LEndian(encodedRedSize, encodedRed);
    } else {
        WriteBuffer_LEndian(static_cast<int>(planeSize), bluePlane);
        WriteBuffer_LEndian(static_cast<int>(planeSize), greenPlane);
        WriteBuffer_LEndian(static_cast<int>(planeSize), redPlane);
    }

    if (alphaPlane) {
        WriteBuffer_LEndian(static_cast<int>(planeSize), alphaPlane);
    } else {
        WriteBuffer_LEndian(0, nullptr);
    }

    delete[] encodedBlue;
    delete[] encodedGreen;
    delete[] encodedRed;

    delete[] bluePlane;
    delete[] greenPlane;
    delete[] redPlane;
    delete[] alphaPlane;
}

CKBYTE *CKStateChunk::ReadRawBitmap(VxImageDescEx &desc) {
    if (!m_ChunkParser)
        return nullptr;

    int bitsPerPixel = ReadInt();
    if (bitsPerPixel == 0)
        return nullptr;

    desc.Width = ReadInt();
    desc.Height = ReadInt();
    desc.AlphaMask = ReadDword();
    desc.RedMask = ReadDword();
    desc.GreenMask = ReadDword();
    desc.BlueMask = ReadDword();
    desc.BytesPerLine = desc.Width * 4;
    desc.BitsPerPixel = 32;

    void *bluePlane = nullptr;
    void *greenPlane = nullptr;
    void *redPlane = nullptr;
    void *alphaPlane = nullptr;
    bool alphaAlreadyRead = false;

    int compression = ReadDword() & 0xF;
    if (compression == 0) {
        ReadBuffer(&bluePlane);
        ReadBuffer(&greenPlane);
        ReadBuffer(&redPlane);
    } else if (compression == 1) {
        void *encodedBlue = nullptr;
        void *encodedGreen = nullptr;
        void *encodedRed = nullptr;

        const int blueEncodedSize = ReadBuffer(&encodedBlue);
        const int greenEncodedSize = ReadBuffer(&encodedGreen);
        const int redEncodedSize = ReadBuffer(&encodedRed);

        CKBYTE *decodedBlue = nullptr;
        CKBYTE *decodedGreen = nullptr;
        CKBYTE *decodedRed = nullptr;

        bool decodeOk = blueEncodedSize > 0 && greenEncodedSize > 0 && redEncodedSize > 0;
        if (decodeOk)
            decodeOk = CKJpegDecoder::DecodeGrayscalePlane(static_cast<CKBYTE *>(encodedBlue), blueEncodedSize, desc.Width, desc.Height, &decodedBlue);
        if (decodeOk)
            decodeOk = CKJpegDecoder::DecodeGrayscalePlane(static_cast<CKBYTE *>(encodedGreen), greenEncodedSize, desc.Width, desc.Height, &decodedGreen);
        if (decodeOk)
            decodeOk = CKJpegDecoder::DecodeGrayscalePlane(static_cast<CKBYTE *>(encodedRed), redEncodedSize, desc.Width, desc.Height, &decodedRed);

        delete[] static_cast<CKBYTE *>(encodedBlue);
        delete[] static_cast<CKBYTE *>(encodedGreen);
        delete[] static_cast<CKBYTE *>(encodedRed);

        if (!decodeOk) {
            delete[] decodedBlue;
            delete[] decodedGreen;
            delete[] decodedRed;
            void *alphaDiscard = nullptr;
            ReadBuffer(&alphaDiscard);
            delete[] static_cast<CKBYTE *>(alphaDiscard);
            return nullptr;
        }

        bluePlane = decodedBlue;
        greenPlane = decodedGreen;
        redPlane = decodedRed;

        ReadBuffer(&alphaPlane);
        alphaAlreadyRead = true;
    } else {
        for (int plane = 0; plane < 3; ++plane) {
            void *discard = nullptr;
            ReadBuffer(&discard);
            delete[] static_cast<CKBYTE *>(discard);
        }
        void *alphaDiscard = nullptr;
        ReadBuffer(&alphaDiscard);
        delete[] static_cast<CKBYTE *>(alphaDiscard);
        return nullptr;
    }

    if (!alphaAlreadyRead)
        ReadBuffer(&alphaPlane);

    CKBYTE *output = new CKBYTE[desc.BytesPerLine * desc.Height];
    if (!output) {
        delete[] static_cast<CKBYTE *>(bluePlane);
        delete[] static_cast<CKBYTE *>(greenPlane);
        delete[] static_cast<CKBYTE *>(redPlane);
        delete[] static_cast<CKBYTE *>(alphaPlane);
        return nullptr;
    }

    CKBYTE *blue = static_cast<CKBYTE *>(bluePlane);
    CKBYTE *green = static_cast<CKBYTE *>(greenPlane);
    CKBYTE *red = static_cast<CKBYTE *>(redPlane);
    CKBYTE *alpha = static_cast<CKBYTE *>(alphaPlane);

    int rows = desc.Height;
    int columns = desc.Width;
    CKBYTE *dst = output;

    if (rows > 0 && columns > 0 && blue && green && red) {
        if (alpha) {
            for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < columns; ++x) {
                    *dst++ = *blue++;
                    *dst++ = *green++;
                    *dst++ = *red++;
                    *dst++ = *alpha++;
                }
            }
        } else {
            for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < columns; ++x) {
                    *dst++ = *blue++;
                    *dst++ = *green++;
                    *dst++ = *red++;
                    *dst++ = 0xFF;
                }
            }
        }
    }

    desc.AlphaMask = A_MASK;

    delete[] static_cast<CKBYTE *>(bluePlane);
    delete[] static_cast<CKBYTE *>(greenPlane);
    delete[] static_cast<CKBYTE *>(redPlane);
    delete[] static_cast<CKBYTE *>(alphaPlane);

    return output;
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
        delete[] m_Data;
        // Allocate exact size for compressed data
        Bytef *data = new Bytef[destSize];
        if (data) {
            memcpy(data, buf, destSize);
            m_Data = (int *) data;
            // IDA stores raw byte size, not DWORD count
            m_ChunkSize = destSize;
        }
    }
    delete[] buf;
}

CKBOOL CKStateChunk::UnPack(int DestSize) {
    if (DestSize <= 0) return FALSE;

    uLongf size = DestSize;
    Bytef *buf = new Bytef[DestSize];
    if (!buf) return FALSE;

    // Pass raw m_ChunkSize as compressed byte count (from Pack)
    int err = uncompress(buf, &size, (const Bytef *) m_Data, m_ChunkSize);
    if (err == Z_OK) {
        delete[] m_Data;
        // Store as DWORD count after decompression
        m_ChunkSize = DestSize >> 2;
        int *data = new int[m_ChunkSize];
        if (data) {
            memcpy(data, buf, DestSize);
            m_Data = data;
        }
    }
    delete[] buf;
    return err == Z_OK;
}

void CKStateChunk::AttributePatch(CKBOOL preserveData, int *ConversionTable, int NbEntries) {
    if (m_Managers)
        return;

    const int sequenceCount = StartReadSequence();
    const int attributeSectionStart = GetCurrentPos();
    Skip(sequenceCount);

    if (!preserveData && sequenceCount > 0) {
        StartReadSequence();
        int remaining = sequenceCount;
        while (remaining-- > 0) {
            CKStateChunk *subChunk = ReadSubChunk();
            if (subChunk)
                delete subChunk;
        }
    }

    m_Managers = new IntListStruct();
    m_Managers->AddEntries(attributeSectionStart);

    Skip(3); // Skip attribute header (GUID + count)

    if (ConversionTable && NbEntries > 0 && sequenceCount > 0) {
        int remaining = sequenceCount;
        while (remaining-- > 0) {
            int *value = &m_Data[m_ChunkParser->CurrentPos];
            if (*value >= 0 && *value < NbEntries)
                *value = ConversionTable[*value];
            m_ChunkParser->CurrentPos++;
        }
    }

    SeekIdentifier(CK_STATESAVE_NEWATTRIBUTES);
}

int CKStateChunk::IterateAndDo(ChunkIterateFct fct, ChunkIteratorData *it) {
    if (!it)
        return 0;

    int totalResult = fct(it);
    if (!it->Chunks || it->ChunkCount <= 0 || !it->Data)
        return totalResult;

    const bool parentUsesLegacyLayout = (it->ChunkVersion <= CHUNK_VERSION1);
    int chunkIndex = 0;

    while (chunkIndex < it->ChunkCount) {
        ChunkIteratorData child;
        child.CopyFctData(it);

        const int entry = it->Chunks[chunkIndex];
        if (entry >= 0) {
            int *data = it->Data;
            const int blockSize = data[entry];
            if (blockSize > 0 && data[entry + 2] + 4 != blockSize) {
                child.ChunkVersion = static_cast<short>(data[entry + 2] >> 16);
                child.ChunkSize = data[entry + 3];
                child.IdCount = data[entry + 5];
                child.ChunkCount = data[entry + 6];

                int payloadOffset = entry + 7;
                int managerCount = data[payloadOffset];
                if (parentUsesLegacyLayout) {
                    managerCount = 0;
                } else {
                    payloadOffset++;
                }
                child.ManagerCount = managerCount;

                child.Data = &data[payloadOffset];
                int *cursor = child.Data + child.ChunkSize;

                child.Ids = (child.IdCount > 0) ? cursor : nullptr;
                cursor += child.IdCount;

                child.Chunks = (child.ChunkCount > 0) ? cursor : nullptr;
                cursor += child.ChunkCount;

                child.Managers = (child.ManagerCount > 0) ? cursor : nullptr;
                child.Flag = 0;

                totalResult += IterateAndDo(fct, &child);
            }

            ++chunkIndex;
            continue;
        }

        // Packed sequence marker (-1 followed by offset to packed payload)
        ++chunkIndex;
        if (chunkIndex >= it->ChunkCount)
            break;

        const int sequenceOffset = it->Chunks[chunkIndex];
        if (sequenceOffset < 0 || sequenceOffset >= it->ChunkSize) {
            ++chunkIndex;
            continue;
        }

        int *data = it->Data;
        int packedCount = data[sequenceOffset];
        int cursor = sequenceOffset + 1;

        while (packedCount-- > 0) {
            if (cursor >= it->ChunkSize)
                break;

            const int blockSize = data[cursor++];
            if (blockSize <= 0)
                continue;

            if (cursor + 2 >= it->ChunkSize)
                break;

            const int chunkSizeToken = data[cursor + 1];
            if (chunkSizeToken + 4 == blockSize) {
                cursor += chunkSizeToken + 4;
                continue;
            }

            child.ChunkVersion = static_cast<short>(data[cursor + 1] >> 16);
            child.ChunkSize = data[cursor + 2];
            child.IdCount = data[cursor + 4];
            child.ChunkCount = data[cursor + 5];

            int payloadOffset = cursor + 6;
            int managerCount = data[cursor + 6];
            if (parentUsesLegacyLayout) {
                managerCount = 0;
            } else {
                payloadOffset = cursor + 7;
            }
            child.ManagerCount = managerCount;

            child.Data = &data[payloadOffset];
            int *packedCursor = child.Data + child.ChunkSize;

            child.Ids = (child.IdCount > 0) ? packedCursor : nullptr;
            packedCursor += child.IdCount;

            child.Chunks = (child.ChunkCount > 0) ? packedCursor : nullptr;
            packedCursor += child.ChunkCount;

            child.Managers = (child.ManagerCount > 0) ? packedCursor : nullptr;
            child.Flag = 0;

            totalResult += IterateAndDo(fct, &child);

            cursor = payloadOffset + child.ChunkSize + child.IdCount + child.ChunkCount + child.ManagerCount;
        }

        ++chunkIndex;
    }

    return totalResult;
}

int CKStateChunk::ObjectRemapper(ChunkIteratorData *it) {
    if (!it || !it->Data)
        return 0;

    auto remapId = [&](CK_ID &value) {
        if (it->DepContext) {
            XHashID::Iterator mapIt = it->DepContext->m_MapID.Find(value);
            if (mapIt != it->DepContext->m_MapID.End()) {
                CK_ID mapped = *mapIt;
                if (mapped)
                    value = mapped;
            }
        } else if (it->Context && it->Context->m_ObjectManager) {
            value = it->Context->m_ObjectManager->RealId(value);
        }
    };

    int remappedCount = 0;

    if (it->ChunkVersion >= CHUNK_VERSION1) {
        if (!it->Ids || it->IdCount <= 0)
            return 0;

        for (int idx = 0; idx < it->IdCount;) {
            int dataOffset = it->Ids[idx];
            if (dataOffset >= 0) {
                if (dataOffset >= 0 && dataOffset < it->ChunkSize) {
                    CK_ID &value = reinterpret_cast<CK_ID &>(it->Data[dataOffset]);
                    remapId(value);
                    ++remappedCount;
                }
                ++idx;
                continue;
            }

            ++idx;
            if (idx >= it->IdCount)
                break;

            const int sequenceOffset = it->Ids[idx];
            if (sequenceOffset < 0 || sequenceOffset >= it->ChunkSize) {
                ++idx;
                continue;
            }

            const int count = it->Data[sequenceOffset];
            const int firstEntry = sequenceOffset + 1;
            const int lastEntry = firstEntry + count;
            if (count > 0 && lastEntry <= it->ChunkSize) {
                for (int cursor = firstEntry; cursor < lastEntry; ++cursor) {
                    CK_ID &value = reinterpret_cast<CK_ID &>(it->Data[cursor]);
                    remapId(value);
                }
                remappedCount += count;
            }

            ++idx;
        }
        return remappedCount;
    }

    static constexpr CKDWORD OBJID_MARKER[3] = {0xE32BC4C9, 0x134212E3, 0xFCBAE9DC};
    static constexpr CKDWORD SEQ_MARKER[5] = {0xE192BD47, 0x13246628, 0x13EAB3CE, 0x7891AEFC, 0x13984562};

    for (int pos = 3; pos < it->ChunkSize; ++pos) {
        if (it->Data[pos - 3] == OBJID_MARKER[0] &&
            it->Data[pos - 2] == OBJID_MARKER[1] &&
            it->Data[pos - 1] == OBJID_MARKER[2]) {
            CK_ID &value = reinterpret_cast<CK_ID &>(it->Data[pos]);
            remapId(value);
            ++remappedCount;
        }
    }

    if (it->ChunkSize > 5) {
        for (int pos = 2; pos < it->ChunkSize - 2; ++pos) {
            if (it->Data[pos - 2] == SEQ_MARKER[0] &&
                it->Data[pos - 1] == SEQ_MARKER[1] &&
                it->Data[pos] == SEQ_MARKER[2] &&
                it->Data[pos + 1] == SEQ_MARKER[3] &&
                it->Data[pos + 2] == SEQ_MARKER[4]) {
                const int count = it->Data[pos + 3];
                if (count > 0) {
                    const int firstEntry = pos + 4;
                    const int lastEntry = firstEntry + count;
                    if (lastEntry <= it->ChunkSize) {
                        for (int cursor = firstEntry; cursor < lastEntry; ++cursor) {
                            CK_ID &value = reinterpret_cast<CK_ID &>(it->Data[cursor]);
                            remapId(value);
                        }
                    }
                    remappedCount += count;
                }
            }
        }
    }

    return remappedCount;
}

int CKStateChunk::ManagerRemapper(ChunkIteratorData *it) {
    if (!it || !it->Managers || it->ManagerCount <= 0 || !it->Data || !it->ConversionTable || it->NbEntries <= 0)
        return 0;

    int total = 0;
    for (int idx = 0; idx < it->ManagerCount;) {
        int offset = it->Managers[idx];
        if (offset >= 0) {
            if (offset + 2 < it->ChunkSize) {
                CKGUID guid(it->Data[offset], it->Data[offset + 1]);
                if (guid == it->Guid) {
                    int &value = it->Data[offset + 2];
                    if (value >= 0 && value < it->NbEntries) {
                        value = it->ConversionTable[value];
                        ++total;
                    }
                }
            }
            ++idx;
            continue;
        }

        ++idx;
        if (idx >= it->ManagerCount)
            break;

        const int sequenceOffset = it->Managers[idx];
        if (sequenceOffset < 0 || sequenceOffset + 2 >= it->ChunkSize) {
            ++idx;
            continue;
        }

        const int count = it->Data[sequenceOffset];
        if (count <= 0) {
            ++idx;
            continue;
        }

        CKGUID guid(it->Data[sequenceOffset + 1], it->Data[sequenceOffset + 2]);
        if (guid == it->Guid) {
            const int firstValue = sequenceOffset + 3;
            const int endValue = firstValue + count;
            if (endValue <= it->ChunkSize) {
                for (int cursor = firstValue; cursor < endValue; ++cursor) {
                    int &value = it->Data[cursor];
                    if (value >= 0 && value < it->NbEntries)
                        value = it->ConversionTable[value];
                }
            }
            total += count;
        }

        ++idx;
    }

    return total;
}

int CKStateChunk::ParameterRemapper(ChunkIteratorData *it) {
    if (!it || !it->Data || it->ChunkSize == 0 || !it->ConversionTable)
        return 0;

    int currentPos = 0;
    while (currentPos + 1 < it->ChunkSize) {
        if (it->Data[currentPos] == 0x40) {
            const int headerStart = currentPos + 2;
            if (headerStart + 3 < it->ChunkSize) {
                CKGUID storedGuid(it->Data[headerStart], it->Data[headerStart + 1]);
                if (storedGuid == it->Guid && it->Data[headerStart + 2] == 1) {
                    int &value = it->Data[headerStart + 3];
                    if (value >= 0 && value < it->NbEntries)
                        value = it->ConversionTable[value];
                    else
                        value = 0;
                    return 1;
                }
            }
        }

        const int nextPos = it->Data[currentPos + 1];
        if (nextPos == 0 || nextPos <= currentPos || nextPos >= it->ChunkSize)
            break;
        currentPos = nextPos;
    }

    return 0;
}
