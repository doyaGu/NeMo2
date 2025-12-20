#include "CKFile.h"

#include <stdio.h>

#include "CKPluginManager.h"
#include "CKPathManager.h"
#include "CKParameterManager.h"
#include "CKObjectManager.h"
#include "CKAttributeManager.h"
#include "CKBehavior.h"
#include "CKBeObject.h"
#include "CKInterfaceObjectManager.h"

struct CKFileHeaderPart0 {
    char Signature[8];
    CKDWORD Crc;
    CKDWORD CKVersion;
    CKDWORD FileVersion;
    CKDWORD FileVersion2;
    CKDWORD FileWriteMode;
    CKDWORD Hdr1PackSize;
};

struct CKFileHeaderPart1 {
    CKDWORD DataPackSize;
    CKDWORD DataUnPackSize;
    CKDWORD ManagerCount;
    CKDWORD ObjectCount;
    CKDWORD MaxIDSaved;
    CKDWORD ProductVersion;
    CKDWORD ProductBuild;
    CKDWORD Hdr1UnPackSize;
};

struct CKFileHeader {
    CKFileHeaderPart0 Part0;
    CKFileHeaderPart1 Part1;
};

extern XClassInfoArray g_CKClassInfo;

static CKDWORD g_MaxClassID = CKCID_MAXCLASSID - 1;
static CKBOOL WarningForOlderVersion = FALSE;
static CKDWORD CurrentFileVersion = 0;
static CKDWORD CurrentFileWriteMode = CKFILE_UNCOMPRESSED;

CKSTRING CKJustFile(CKSTRING path) {
    static char buffer[256];

    if (!path)
        return nullptr;

    CKPathSplitter splitter(path);

    const char *name = splitter.GetName();
    const char *ext = splitter.GetExtension();

    if (name && ext) {
        snprintf(buffer, sizeof(buffer), "%s%s", name, ext);
    } else if (name) {
        snprintf(buffer, sizeof(buffer), "%s", name);
    } else if (ext) {
        snprintf(buffer, sizeof(buffer), "%s", ext);
    } else {
        buffer[0] = '\0';
    }

    return buffer;
}

CKBufferParser::CKBufferParser(void *Buffer, int Size)
    : m_Valid(FALSE),
      m_CursorPos(0),
      m_Buffer((char *) Buffer),
      m_Size(Size) {}

CKBufferParser::~CKBufferParser() {
    if (m_Valid)
        delete[] m_Buffer;
}

CKBOOL CKBufferParser::Write(void *x, int size) {
    if (size < 0 || m_CursorPos + size > m_Size)
        return FALSE;

    memcpy(&m_Buffer[m_CursorPos], x, size);
    m_CursorPos += size;
    return TRUE;
}

CKBOOL CKBufferParser::Read(void *x, int size) {
    if (size < 0 || m_CursorPos + size > m_Size)
        return FALSE;

    memcpy(x, &m_Buffer[m_CursorPos], size);
    m_CursorPos += size;
    return TRUE;
}

char *CKBufferParser::ReadString() {
    int len = ReadInt();
    if (len == 0)
        return nullptr;

    if (m_CursorPos + len > m_Size)
        return nullptr;

    char *str = new char[len + 1];
    if (!str)
        return nullptr;

    memset(str, 0, len + 1);
    memcpy(str, &m_Buffer[m_CursorPos], len);
    m_CursorPos += len;
    return str;
}

int CKBufferParser::ReadInt() {
    if (m_CursorPos + sizeof(int) > m_Size)
        return 0;

    int val;
    memcpy(&val, &m_Buffer[m_CursorPos], sizeof(int));
    m_CursorPos += sizeof(int);
    return val;
}

void CKBufferParser::Seek(int Pos) {
    if (Pos < 0)
        m_CursorPos = 0;
    else if (Pos > m_Size)
        m_CursorPos = m_Size;
    else
        m_CursorPos = Pos;
}

void CKBufferParser::Skip(int Offset) {
    int newPos = m_CursorPos + Offset;
    if (newPos < 0)
        m_CursorPos = 0;
    else if (newPos > m_Size)
        m_CursorPos = m_Size;
    else
        m_CursorPos = newPos;
}

CKBOOL CKBufferParser::IsValid() {
    return m_Valid;
}

int CKBufferParser::Size() {
    return m_Size;
}

int CKBufferParser::CursorPos() {
    return m_CursorPos;
}

CKStateChunk *CKBufferParser::ExtractChunk(int Size, CKFile *f) {
    auto *chunk = CreateCKStateChunk(0, f);
    if (!chunk->ConvertFromBuffer(&m_Buffer[m_CursorPos])) {
        delete chunk;
        chunk = nullptr;
    }
    m_CursorPos += Size;
    return chunk;
}

void CKBufferParser::ExtractChunk(int Size, CKFile *f, CKFileChunk *chunk) {
}

CKDWORD CKBufferParser::ComputeCRC(int Size, CKDWORD PrevCRC) {
    return CKComputeDataCRC(&m_Buffer[m_CursorPos], Size, PrevCRC);
}

CKBufferParser *CKBufferParser::Extract(int Size) {
    if (Size < 0 || m_CursorPos + Size > m_Size)
        return nullptr;

    auto *buffer = new char[Size];
    if (!buffer)
        return nullptr;

    auto *parser = new CKBufferParser(buffer, Size);
    parser->m_Valid = TRUE;
    parser->Write(&m_Buffer[m_CursorPos], Size);
    return parser;
}

CKBOOL CKBufferParser::ExtractFile(char *Filename, int Size) {
    if (Size < 0 || m_CursorPos + Size > m_Size)
        return FALSE;

    FILE *fp = fopen(Filename, "wb");
    if (!fp)
        return FALSE;
    fwrite(&m_Buffer[m_CursorPos], sizeof(char), Size, fp);
    fclose(fp);
    m_CursorPos += Size;
    return TRUE;
}

CKBufferParser *CKBufferParser::ExtractDecoded(int Size, CKDWORD *Key) { return nullptr; }

CKBufferParser *CKBufferParser::UnPack(int UnpackSize, int PackSize) {
    char *buffer = CKUnPackData(UnpackSize, &m_Buffer[m_CursorPos], PackSize);
    if (!buffer) {
        return nullptr;
    }

    auto *parser = new CKBufferParser(buffer, UnpackSize);
    parser->m_Valid = TRUE;
    return parser;
}

void CKBufferParser::InsertChunk(CKStateChunk *chunk) {
    if (!chunk)
        return;

    int size = 0;
    size = chunk->ConvertToBuffer(&m_Buffer[m_CursorPos]);
    m_CursorPos += size;
    Write(&size, sizeof(int));
}

CKBufferParser *CKBufferParser::Pack(int Size, int CompressionLevel) {
    if (Size <= 0)
        return nullptr;

    int newSize = 0;
    char *buffer = CKPackData(&m_Buffer[m_CursorPos], Size, newSize, CompressionLevel);
    if (!buffer) {
        return nullptr;
    }

    auto *parser = new CKBufferParser(buffer, newSize);
    parser->m_Valid = TRUE;
    return parser;
}

void CKBufferParser::Encode(int Size, CKDWORD *Key) {
}

CKDWORD GetCurrentFileLoadOption() {
    return CurrentFileWriteMode;
}

CKDWORD GetCurrentFileVersion() {
    return CurrentFileVersion;
}

CKERROR CKFile::OpenFile(CKSTRING filename, CK_LOAD_FLAGS Flags) {
    ClearData();

    if (!filename) {
        return CKERR_INVALIDPARAMETER;
    }

    m_FileName = CKStrdup(filename);
    m_MappedFile = new VxMemoryMappedFile(m_FileName);
    if (m_MappedFile->GetErrorType() != CK_OK) {
        return CKERR_INVALIDFILE;
    }

    m_Context->SetLastCmoLoaded(filename);
    return OpenMemory(m_MappedFile->GetBase(), m_MappedFile->GetFileSize(), Flags);
}

CKERROR CKFile::OpenMemory(void *MemoryBuffer, int BufferSize, CK_LOAD_FLAGS Flags) {
    if (!MemoryBuffer) {
        return CKERR_INVALIDPARAMETER;
    }

    if (BufferSize < 32 || memcmp(MemoryBuffer, "Nemo", 4) != 0) {
        return CKERR_INVALIDFILE;
    }

    m_Parser = new CKBufferParser(MemoryBuffer, BufferSize);
    if (!m_Parser) {
        return CKERR_OUTOFMEMORY;
    }

    if (!m_Parser->m_Buffer) {
        delete m_Parser;
        m_Parser = nullptr;
        return CKERR_INVALIDPARAMETER;
    }

    WarningForOlderVersion = FALSE;
    m_Flags = Flags;
    m_IndexByClassId.Resize(g_MaxClassID);

    return ReadFileHeaders(&m_Parser);
}

CKERROR CKFile::LoadFileData(CKObjectArray *liste) {
    if (!m_Parser && !m_ReadFileDataDone)
        return CKERR_INVALIDFILE;

    CKERROR err = CK_OK;

    m_Context->ExecuteManagersPreLoad();
    m_Context->m_InLoad = TRUE;

    if (m_ReadFileDataDone) {
        FinishLoading(liste, m_Flags);
        if (WarningForOlderVersion)
            m_Context->OutputToConsole((CKSTRING) "Obsolete File Format,Please Re-Save...");
    } else {
        err = ReadFileData(&m_Parser);
        if (err == CK_OK) {
            if (m_Parser) {
                delete m_Parser;
                m_Parser = nullptr;
            }

            if (m_MappedFile) {
                delete m_MappedFile;
                m_MappedFile = nullptr;
            }

            FinishLoading(liste, m_Flags);
            if (WarningForOlderVersion)
                m_Context->OutputToConsole((CKSTRING) "Obsolete File Format,Please Re-Save...");
        }
    }

    m_Context->SetAutomaticLoadMode(CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID);
    m_Context->SetUserLoadCallback(nullptr, nullptr);

    if (m_Parser) {
        delete m_Parser;
        m_Parser = nullptr;
    }

    if (m_MappedFile) {
        delete m_MappedFile;
        m_MappedFile = nullptr;
    }

    m_Context->ExecuteManagersPostLoad();
    m_Context->m_InLoad = FALSE;

    return err;
}

void CKFile::ClearData() {
    for (XArray<CKFileObject>::Iterator it = m_FileObjects.Begin();
         it != m_FileObjects.End(); ++it) {
        if (it->Data) {
            it->Data->Clear();
            delete it->Data;
        }
        delete[] it->Name;
        it->Data = nullptr;
        it->Name = nullptr;
    }

    for (XArray<CKFileManagerData>::Iterator it = m_ManagersData.Begin();
         it != m_ManagersData.End(); ++it) {
        if (it->data) {
            it->data->Clear();
            delete it->data;
            it->data = nullptr;
        }
    }

    m_FileObjects.Clear();
    m_ManagersData.Clear();
    m_AlreadySavedMask.Clear();
    m_AlreadyReferencedMask.Clear();
    m_ReferencedObjects.Clear();
    m_IndexByClassId.Clear();

    delete[] m_FileName;
    m_FileName = nullptr;

    if (m_Parser) {
        delete m_Parser;
        m_Parser = nullptr;
    }

    if (m_MappedFile) {
        delete m_MappedFile;
        m_MappedFile = nullptr;
    }

    m_Flags = 0;
    m_SaveIDMax = 0;
}

CKERROR CKFile::ReadFileHeaders(CKBufferParser **ParserPtr) {
    CKBufferParser *parser = *ParserPtr;
    m_IncludedFiles.Clear();

    if (parser->Size() < 32) {
        return CKERR_INVALIDFILE;
    }

    CKFileHeader header = {};

    parser->Read(&header.Part0, sizeof(CKFileHeaderPart0));

    if (header.Part0.FileVersion2 != 0) {
        memset(&header.Part0, 0, sizeof(CKFileHeaderPart0));
        WarningForOlderVersion = TRUE;
    }

    if (header.Part0.FileVersion >= 10) {
        m_Context->OutputToConsole((CKSTRING) "This version is too old to load this file");
        return CKERR_OBSOLETEVIRTOOLS;
    }

    if (header.Part0.FileVersion < 5) {
        memset(&header.Part1, 0, sizeof(CKFileHeaderPart1));
    } else if (parser->Size() >= 64) {
        parser->Read(&header.Part1, sizeof(CKFileHeaderPart1));
    } else {
        return CKERR_INVALIDFILE;
    }

    if (header.Part1.ProductVersion >= 12) {
        header.Part1.ProductVersion = 0;
        header.Part1.ProductBuild = 0x1010000;
    }

    m_FileInfo.ProductVersion = header.Part1.ProductVersion;
    m_FileInfo.ProductBuild = header.Part1.ProductBuild;
    m_FileInfo.FileWriteMode = header.Part0.FileWriteMode;
    m_FileInfo.CKVersion = header.Part0.CKVersion;
    m_FileInfo.FileVersion = header.Part0.FileVersion;
    m_FileInfo.FileSize = parser->Size();
    m_FileInfo.ManagerCount = header.Part1.ManagerCount;
    m_FileInfo.ObjectCount = header.Part1.ObjectCount;
    m_FileInfo.MaxIDSaved = header.Part1.MaxIDSaved;
    m_FileInfo.Hdr1PackSize = header.Part0.Hdr1PackSize;
    m_FileInfo.Hdr1UnPackSize = header.Part1.Hdr1UnPackSize;
    m_FileInfo.DataPackSize = header.Part1.DataPackSize;
    m_FileInfo.DataUnPackSize = header.Part1.DataUnPackSize;
    m_FileInfo.Crc = header.Part0.Crc;

    if (header.Part0.FileVersion >= 8) {
        header.Part0.Crc = 0;
        CKDWORD crc = CKComputeDataCRC((char *) (&header.Part0), sizeof(CKFileHeaderPart0), 0);
        int prev = parser->CursorPos();
        parser->Seek(sizeof(CKFileHeaderPart0));
        crc = parser->ComputeCRC(sizeof(CKFileHeaderPart1), crc);
        parser->Skip(sizeof(CKFileHeaderPart1));
        crc = parser->ComputeCRC(m_FileInfo.Hdr1PackSize, crc);
        parser->Skip(m_FileInfo.Hdr1PackSize);
        crc = parser->ComputeCRC(m_FileInfo.DataPackSize, crc);
        parser->Seek(prev);
        if (crc != m_FileInfo.Crc) {
            m_Context->OutputToConsole((CKSTRING) "Crc Error in m_File");
            return CKERR_FILECRCERROR;
        }

        if (m_FileInfo.Hdr1PackSize != m_FileInfo.Hdr1UnPackSize) {
            CKBufferParser *unpacked = parser->UnPack(m_FileInfo.Hdr1UnPackSize, m_FileInfo.Hdr1PackSize);
            if (!unpacked) {
                m_Context->OutputToConsole((CKSTRING) "Error unpacking header chunk.");
                return CKERR_INVALIDFILE;
            }
            if (parser != *ParserPtr) {
                delete parser;
            }
            parser = unpacked;
        }
    }

    if (m_FileInfo.FileVersion >= 7) {
        m_SaveIDMax = m_FileInfo.MaxIDSaved;
        m_FileObjects.Resize(m_FileInfo.ObjectCount);
        for (XArray<CKFileObject>::Iterator oit = m_FileObjects.Begin(); oit != m_FileObjects.End(); ++oit) {
            oit->ObjPtr = nullptr;
            oit->Name = nullptr;
            oit->Data = nullptr;
            oit->Object = parser->ReadInt();
            oit->ObjectCid = parser->ReadInt();
            oit->FileIndex = parser->ReadInt();
            oit->Name = parser->ReadString();
        }
    }

    CKBOOL noPluginMissing = TRUE;

    if (m_FileInfo.FileVersion >= 8) {
        const int pluginsDepCount = parser->ReadInt();
        m_PluginsDep.Resize(pluginsDepCount);
        for (XArray<CKFilePluginDependencies>::Iterator pit = m_PluginsDep.Begin(); pit != m_PluginsDep.End(); ++pit) {
            pit->m_PluginCategory = parser->ReadInt();

            const int count = parser->ReadInt();
            pit->m_Guids.Resize(count);

            if (count > 0) {
                parser->Read(&pit->m_Guids[0], sizeof(CKGUID) * count);
            }

            if (m_Flags & CK_LOAD_CHECKDEPENDENCIES) {
                for (int j = 0; j < count; ++j) {
                    CKGUID guid = pit->m_Guids[j];
                    CKPluginEntry *entry = CKGetPluginManager()->FindComponent(guid, pit->m_PluginCategory);
                    if (entry) {
                        pit->ValidGuids.Set(j);
                    } else {
                        noPluginMissing = FALSE;
                        pit->ValidGuids.Unset(j);
                    }
                }
            }
        }

        int includedFileSize = parser->ReadInt();
        if (includedFileSize > 0) {
            int includedFileCount = parser->ReadInt();
            m_IncludedFiles.Resize(includedFileCount);
            includedFileSize -= 4;
        }
        parser->Skip(includedFileSize);
    }

    if (parser != *ParserPtr) {
        delete parser;
        parser = *ParserPtr;
        parser->Skip(m_FileInfo.Hdr1PackSize);
    }

    CurrentFileVersion = header.Part0.FileVersion;
    CurrentFileWriteMode = header.Part0.FileWriteMode;

    if (m_FileInfo.FileVersion < 8 && (m_Flags & CK_LOAD_CHECKDEPENDENCIES)) {
        m_ReadFileDataDone = TRUE;
        CKERROR ret = ReadFileData(ParserPtr);

        if (m_Parser) {
            delete m_Parser;
            m_Parser = nullptr;
        }

        if (m_MappedFile) {
            delete m_MappedFile;
            m_MappedFile = nullptr;
        }

        if (ret != CK_OK) {
            m_Context->SetAutomaticLoadMode(CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID);
            m_Context->SetUserLoadCallback(nullptr, nullptr);
            m_Context->m_InLoad = FALSE;
            return ret;
        }

        m_PluginsDep.Resize(2);
        m_PluginsDep[0].m_PluginCategory = CKPLUGIN_BEHAVIOR_DLL;
        m_PluginsDep[1].m_PluginCategory = CKPLUGIN_MANAGER_DLL;

        for (XArray<CKFileObject>::Iterator oit = m_FileObjects.Begin(); oit != m_FileObjects.End(); ++oit) {
            CKStateChunk *chunk = oit->Data;
            if (chunk && oit->ObjectCid == CKCID_BEHAVIOR) {
                CKGUID behGuid;

                chunk->StartRead();
                if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORPROTOGUID) != 0) {
                    behGuid = chunk->ReadGuid();
                } else {
                    if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORNEWDATA) != 0) {
                        if (chunk->GetDataVersion() < 5) {
                            behGuid = chunk->ReadGuid();
                            if ((chunk->ReadInt() & CKBEHAVIOR_BUILDINGBLOCK) == 0)
                                continue;
                        } else {
                            if ((chunk->ReadInt() & CKBEHAVIOR_BUILDINGBLOCK) == 0)
                                continue;
                            behGuid = chunk->ReadGuid();
                        }
                    }
                }

                if (behGuid.IsValid() && !CKGetPluginManager()->FindComponent(behGuid, CKPLUGIN_BEHAVIOR_DLL)) {
                    noPluginMissing = FALSE;
                    if (m_PluginsDep[0].m_Guids.Size() < m_PluginsDep[0].ValidGuids.Size()) {
                        m_PluginsDep[0].ValidGuids.Unset(m_PluginsDep[0].m_Guids.Size());
                    }
                    m_PluginsDep[0].m_Guids.PushBack(behGuid);
                }
            }
        }
    }

    return noPluginMissing ? CK_OK : CKERR_PLUGINSMISSING;
}

CKERROR CKFile::ReadFileData(CKBufferParser **ParserPtr) {
    CKBufferParser *parser = *ParserPtr;

    if ((m_FileInfo.FileWriteMode & (CKFILE_CHUNKCOMPRESSED_OLD | CKFILE_WHOLECOMPRESSED)) != 0) {
        CKBufferParser *unpacked = parser->UnPack(m_FileInfo.DataUnPackSize, m_FileInfo.DataPackSize);
        if (!unpacked) {
            m_Context->OutputToConsole((CKSTRING) "Error unpacking data chunk.");
            return CKERR_INVALIDFILE;
        }
        parser = unpacked;
        (*ParserPtr)->Skip(m_FileInfo.DataPackSize);
    }

    if (m_FileInfo.FileVersion < 8) {
        if (m_FileInfo.FileVersion >= 2) {
            if (m_FileInfo.Crc != parser->ComputeCRC(parser->Size() - parser->CursorPos())) {
                if (parser != *ParserPtr)
                    delete parser;

                m_Context->OutputToConsole((CKSTRING) "Crc Error in m_File");
                return CKERR_FILECRCERROR;
            }
        } else {
            WarningForOlderVersion = TRUE;
        }

        m_SaveIDMax = parser->ReadInt();
        m_FileInfo.ObjectCount = parser->ReadInt();

        if (m_FileObjects.Size() != m_FileInfo.ObjectCount) {
            m_FileObjects.Resize(m_FileInfo.ObjectCount);
            const int fileObjectCount = m_FileObjects.Size();
            for (int i = 0; i < fileObjectCount; ++i) {
                new(&m_FileObjects[i]) CKFileObject();
            }
        }
    }

    if (m_FileInfo.FileVersion >= 6) {
        if (m_FileInfo.ManagerCount > 0) {
            m_ManagersData.Resize(m_FileInfo.ManagerCount);
            for (XArray<CKFileManagerData>::Iterator mit = m_ManagersData.Begin(); mit != m_ManagersData.End(); ++mit) {
                parser->Read(mit->Manager.d, sizeof(CKGUID));
                const int managerDataSize = parser->ReadInt();
                if (managerDataSize > 0) {
                    mit->data = parser->ExtractChunk(managerDataSize, this);
                } else {
                    mit->data = nullptr;
                }
            }
        }
    }

    if (m_FileInfo.ObjectCount > 0) {
        if (m_FileInfo.FileVersion >= 4) {
            for (XArray<CKFileObject>::Iterator oit = m_FileObjects.Begin(); oit != m_FileObjects.End(); ++oit) {
                if (m_FileInfo.FileVersion < 7) {
                    oit->Object = parser->ReadInt();
                }

                const int fileObjectSize = parser->ReadInt();
                if (fileObjectSize > 0) {
                    if (m_FileInfo.FileVersion < 7 || !(m_Flags & CK_LOAD_ONLYBEHAVIORS) || oit->ObjectCid == CKCID_BEHAVIOR) {
                        CKStateChunk *chunk = parser->ExtractChunk(fileObjectSize, this);
                        if (chunk) {
                            oit->Data = chunk;
                            oit->PostPackSize = chunk->GetDataSize();
                            oit->PrePackSize = chunk->GetDataSize();
                        } else {
                            oit->Data = nullptr;
                        }
                    } else {
                        parser->Skip(fileObjectSize);
                    }
                } else {
                    oit->Data = nullptr;
                }
            }
        } else {
            const int fileObjectCount = m_FileObjects.Size();
            for (int o = 0; o < fileObjectCount; ++o) {
                CKFileObject *obj = &m_FileObjects[o];
                obj->Object = parser->ReadInt();
                const int fileObjectUnPackSize = parser->ReadInt();
                if (fileObjectUnPackSize > 0) {
                    WarningForOlderVersion = TRUE;
                    const int chunkCid = parser->ReadInt();
                    obj->Data = CreateCKStateChunk(chunkCid, this);
                    obj->SaveFlags = parser->ReadInt();
                    const int dataSize = parser->ReadInt();
                    if (m_FileInfo.FileWriteMode & CKFILE_CHUNKCOMPRESSED_OLD) {
                        obj->Data->m_ChunkSize = dataSize;
                    } else {
                        obj->Data->m_ChunkSize = dataSize >> 2;
                    }

                    if (dataSize == fileObjectUnPackSize) {
                        obj->Data->m_ChunkSize = dataSize >> 2;
                    }

                    obj->Data->m_Data = new int[dataSize];
                    parser->Read(obj->Data->m_Data, dataSize);
                    if (dataSize != fileObjectUnPackSize &&
                        (m_FileInfo.FileWriteMode & CKFILE_CHUNKCOMPRESSED_OLD) &&
                        !obj->Data->UnPack(fileObjectUnPackSize)) {
                        if (obj->Data) {
                            delete obj->Data;
                            obj->Data = nullptr;
                        }
                        m_Context->OutputToConsoleEx("Crc Error While Unpacking : Object=>%d \n", o);
                    }
                }
            }
        }
    }

    if (m_FileInfo.FileVersion < 7) {
        for (XArray<CKFileObject>::Iterator oit = m_FileObjects.Begin(); oit != m_FileObjects.End(); ++oit) {
            oit->Name = nullptr;
            if (oit->Data) {
                if (oit->Data->SeekIdentifier(1)) {
                    oit->Data->ReadString(&oit->Name);
                }
                oit->ObjectCid = oit->Data->GetChunkClassID();
            }
        }
    }

    if (m_IncludedFiles.Size() > 0) {
        for (XClassArray<XString>::Iterator iit = m_IncludedFiles.Begin();
             iit != m_IncludedFiles.End(); ++iit) {
            const int fileNameLength = parser->ReadInt();
            char fileName[CKMAX_PATH] = {0};
            if (fileNameLength > 0 && fileNameLength < CKMAX_PATH) {
                parser->Read(fileName, fileNameLength);
                fileName[fileNameLength] = '\0';
            }

            const int fileSize = parser->ReadInt();
            if (fileSize > 0) {
                XString temp = m_Context->GetPathManager()->GetVirtoolsTemporaryFolder();
                CKPathMaker pm(nullptr, temp.Str(), fileName, nullptr);
                char *filePath = pm.GetFileName();
                parser->ExtractFile(filePath, fileSize);
            }
        }
    }

    if (parser && parser != *ParserPtr) {
        delete parser;
    }

    return CK_OK;
}

CKERROR CKFile::Load(CKSTRING filename, CKObjectArray *list, CK_LOAD_FLAGS Flags) {
    CKERROR err = OpenFile(filename, Flags);
    if (err != CK_OK && err != CKERR_PLUGINSMISSING)
        return err;

    m_Context->SetLastCmoLoaded(filename);
    return LoadFileData(list);
}

CKERROR CKFile::Load(void *MemoryBuffer, int BufferSize, CKObjectArray *list, CK_LOAD_FLAGS Flags) {
    CKERROR err = OpenMemory(MemoryBuffer, BufferSize, Flags);
    if (err != CK_OK && err != CKERR_PLUGINSMISSING)
        return err;

    return LoadFileData(list);
}

void CKFile::UpdateAndApplyAnimationsTo(CKCharacter *character) {
}

CKERROR CKFile::StartSave(CKSTRING filename, CKDWORD Flags) {
    ClearData();

    m_Context->ExecuteManagersPreSave();
    m_Context->m_Saving = TRUE;

    m_Flags = Flags;
    m_SceneSaved = FALSE;
    m_FileObjects.Clear();
    m_IndexByClassId.Resize(g_MaxClassID);
    delete[] m_FileName;
    m_FileName = nullptr;
    if (!filename) {
        m_Context->m_Saving = FALSE;
        return CKERR_INVALIDFILE;
    }

    m_FileName = CKStrdup(filename);
    FILE *fp = fopen(m_FileName, "ab");
    if (!fp)
        return CKERR_CANTWRITETOFILE;

    fclose(fp);

    m_Context->WarnAllBehaviors(CKM_BEHAVIORPRESAVE);
    return CK_OK;
}

void CKFile::SaveObject(CKObject *obj, CKDWORD flags) {
    if (!obj)
        return;
    if (m_AlreadySavedMask.IsSet(obj->GetID()))
        return;

    m_AlreadySavedMask.Set(obj->GetID());
    if (obj->IsDynamic() || (obj->GetObjectFlags() & CK_OBJECT_NOTTOBESAVED))
        return;

    if (CKIsChildClassOf(obj->GetClassID(), CKCID_SCENE) || CKIsChildClassOf(obj->GetClassID(), CKCID_LEVEL))
        m_SceneSaved = TRUE;

    if (obj->GetID() > m_SaveIDMax)
        m_SaveIDMax = obj->GetID();

    if (!CKIsChildClassOf(obj->GetClassID(), CKCID_LEVEL))
        obj->PreSave(this, flags);

    if (m_AlreadyReferencedMask.IsSet(obj->GetID())) {
        m_ReferencedObjects.Remove(obj);
        m_AlreadyReferencedMask.Unset(obj->GetID());
    } else {
        CKFileObject fileObj;
        fileObj.Object = obj->GetID();
        fileObj.ObjPtr = obj;
        fileObj.ObjectCid = obj->GetClassID();
        fileObj.SaveFlags = flags;
        fileObj.Name = CKStrdup(obj->GetName());
        m_FileObjects.PushBack(fileObj);
        m_ObjectsHashTable.Insert(m_FileObjects.Size() - 1, obj->GetID());
        m_IndexByClassId[obj->GetClassID()].PushBack(m_FileObjects.Size() - 1);
    }

    if (CKIsChildClassOf(obj->GetClassID(), CKCID_LEVEL))
        obj->PreSave(this, flags);
}

void CKFile::SaveObjects(CKObjectArray *array, CKDWORD flags) {
    if (!array) return;

    array->Reset();
    while (!array->EndOfList()) {
        CKObject *obj = array->GetData(m_Context);
        SaveObject(obj, flags);
        array->Next();
    }
}

void CKFile::SaveObjects(CK_ID *ids, int count, CKDWORD flags) {
    if (count <= 0)
        return;
    for (int i = 0; i < count; ++i) {
        CKObject *obj = m_Context->GetObject(ids[i]);
        SaveObject(obj, flags);
    }
}

void CKFile::SaveObjects(CKObject **objs, int count, CKDWORD flags) {
    if (count <= 0)
        return;
    for (int i = 0; i < count; ++i) {
        SaveObject(objs[i], flags);
    }
}

void CKFile::SaveObjectAsReference(CKObject *obj) {
    if (!obj)
        return;

    if (obj->IsDynamic() || (obj->GetObjectFlags() & CK_OBJECT_NOTTOBESAVED) ||
        m_AlreadySavedMask.IsSet(obj->GetID()) ||
        m_AlreadyReferencedMask.IsSet(obj->GetID()))
        return;

    m_AlreadyReferencedMask.Set(obj->GetID());

    if (obj->GetID() > m_SaveIDMax)
        m_SaveIDMax = obj->GetID();

    m_ReferencedObjects.PushBack(obj);

    if (CKIsChildClassOf(obj->GetClassID(), CKCID_SCENE) || CKIsChildClassOf(obj->GetClassID(), CKCID_LEVEL))
        m_SceneSaved = TRUE;

    CKFileObject fileObj;
    fileObj.Object = obj->GetID();
    fileObj.ObjPtr = obj;
    fileObj.ObjectCid = obj->GetClassID();
    fileObj.SaveFlags = 0;
    fileObj.Name = CKStrdup(obj->GetName());
    m_FileObjects.PushBack(fileObj);
    m_ObjectsHashTable.Insert(m_FileObjects.Size() - 1, obj->GetID());
    m_IndexByClassId[obj->GetClassID()].PushBack(m_FileObjects.Size() - 1);
}

CKERROR CKFile::EndSave() {
    for (XObjectPointerArray::Iterator it = m_ReferencedObjects.Begin(); it != m_ReferencedObjects.End(); ++it) {
        if (*it) {
            (*it)->m_ObjectFlags |= CK_OBJECT_ONLYFORFILEREFERENCE;
        }
    }

    XObjectArray scripts;
    int fileObjectCount = m_FileObjects.Size();
    for (int i = 0; i < fileObjectCount; ++i) {
        CKFileObject &fileObject = m_FileObjects[i];
        CKObject *obj = m_Context->GetObject(fileObject.Object);
        if (obj) {
            if (CKIsChildClassOf(fileObject.ObjectCid, CKCID_BEHAVIOR)) {
                CKBehavior *beh = (CKBehavior *) fileObject.ObjPtr;
                if ((beh->GetFlags() & CKBEHAVIOR_SCRIPT) && !beh->GetInterfaceChunk()) {
                    scripts.PushBack(beh->GetID());
                }
            }
        }
    }

    if (m_Context->m_UICallBackFct) {
        CKUICallbackStruct cbs;
        cbs.Reason = CKUIM_CREATEINTERFACECHUNK;
        cbs.Param1 = (CKDWORD) &scripts;
        m_Context->m_UICallBackFct(cbs, m_Context->m_InterfaceModeData);
    }

    int interfaceDataSize = 0;
    for (int i = 0; i < fileObjectCount; ++i) {
        CKFileObject &fileObject = m_FileObjects[i];
        CKObject *obj = m_Context->GetObject(fileObject.Object);
        if (obj) {
            CKStateChunk *chunk = obj->Save(this, fileObject.SaveFlags);
            if (chunk) {
                chunk->CloseChunk();
            }
            fileObject.Data = chunk;

            if (CKIsChildClassOf(fileObject.ObjectCid, CKCID_BEHAVIOR)) {
                CKBehavior *beh = (CKBehavior *) fileObject.ObjPtr;
                CKStateChunk *interfaceChunk = beh->GetInterfaceChunk();
                if (interfaceChunk)
                    interfaceDataSize += interfaceChunk->GetDataSize() + 2 * (int) sizeof(CKDWORD);
            }
        }

        if (m_Context->m_UICallBackFct) {
            CKUICallbackStruct cbs;
            cbs.Reason = CKUIM_LOADSAVEPROGRESS;
            cbs.Param1 = i;
            cbs.Param2 = fileObjectCount;
            m_Context->m_UICallBackFct(cbs, m_Context->m_InterfaceModeData);
        }
    }

    m_Context->WarnAllBehaviors(CKM_BEHAVIORPOSTSAVE);

    int managerCount = m_Context->GetManagerCount();
    int savedManagerCount = 0;
    if (managerCount > 0) {
        m_ManagersData.Resize(managerCount);

        for (int i = 0; i < managerCount; ++i) {
            CKBaseManager *manager = m_Context->GetManager(i);

            CKFileManagerData *managerData = &m_ManagersData[i];
            managerData->data = nullptr;
            managerData->Manager = manager->GetGuid();

            managerData->data = manager->SaveData(this);
            if (managerData->data)
                ++savedManagerCount;
        }
    }
    m_ManagersData.Resize(savedManagerCount);

    CKPluginManager *pm = CKGetPluginManager();
    pm->ComputeDependenciesList(this);

    int objectInfoSize = 0;
    int objectDataSize = 0;
    for (int i = 0; i < fileObjectCount; ++i) {
        CKFileObject &fileObject = m_FileObjects[i];
        objectInfoSize += 4 * (int) sizeof(CKDWORD);
        if (fileObject.Name)
            objectInfoSize += strlen(fileObject.Name);

        int packSize;
        if (fileObject.Data) {
            packSize = fileObject.Data->ConvertToBuffer(nullptr);
        } else {
            packSize = 0;
        }

        fileObject.PrePackSize = packSize;
        fileObject.PostPackSize = packSize;
        objectDataSize += fileObject.PostPackSize + (int) sizeof(CKDWORD);
    }

    int managerDataSize = 0;
    for (int i = 0; i < savedManagerCount; ++i) {
        CKFileManagerData &managerData = m_ManagersData[i];
        if (managerData.data) {
            managerDataSize += managerData.data->ConvertToBuffer(nullptr);
        }
        managerDataSize += 3 * sizeof(CKDWORD);
    }

    int pluginDepCount = m_PluginsDep.Size();
    int pluginDepsSize = (int) sizeof(CKDWORD);
    for (int i = 0; i < pluginDepCount; ++i) {
        CKFilePluginDependencies &pluginDep = m_PluginsDep[i];
        pluginDepsSize += pluginDep.m_Guids.Size() * (int) sizeof(CKGUID) + 2 * (int) sizeof(CKDWORD);
    }

    int hdr1PackSize = objectInfoSize + pluginDepsSize + (int) (sizeof(CKDWORD) + sizeof(CKDWORD));
    int dataUnPackSize = objectDataSize + managerDataSize;

    if (fileObjectCount > 0) {
        m_FileObjects[0].FileIndex = (int) sizeof(CKFileHeader) + hdr1PackSize + managerDataSize;
        for (int i = 1; i < fileObjectCount; ++i) {
            CKFileObject &fileObject = m_FileObjects[i];
            CKFileObject &prevFileObject = m_FileObjects[i - 1];
            fileObject.FileIndex = prevFileObject.FileIndex + prevFileObject.PostPackSize + (int) sizeof(CKDWORD);
        }
    }

    CKFileHeader header = {};
    strcpy(header.Part0.Signature, "Nemo Fi");
    header.Part0.Crc = 0;
    header.Part0.FileVersion2 = 0;
    header.Part0.CKVersion = CKVERSION;
    header.Part0.FileVersion = 8;
    header.Part0.FileWriteMode = m_Context->GetFileWriteMode();
    header.Part1.ObjectCount = fileObjectCount;
    header.Part1.ManagerCount = savedManagerCount;
    header.Part1.Hdr1UnPackSize = hdr1PackSize;
    header.Part1.DataUnPackSize = dataUnPackSize;
    header.Part0.Hdr1PackSize = hdr1PackSize;
    header.Part1.DataPackSize = dataUnPackSize;
    header.Part1.ProductVersion = m_Context->m_VirtoolsVersion;
    header.Part1.ProductBuild = m_Context->m_VirtoolsBuild;
    header.Part1.MaxIDSaved = m_SaveIDMax;

    char *hdr1Buffer = new char[hdr1PackSize];
    CKBufferParser *hdr1BufferParser = new CKBufferParser(hdr1Buffer, hdr1PackSize);
    hdr1BufferParser->m_Valid = TRUE;

    CKBufferParser *parser = hdr1BufferParser;

    if (fileObjectCount > 0) {
        for (int i = 0; i < fileObjectCount; ++i) {
            CKFileObject &fileObject = m_FileObjects[i];
            int nameSize = (fileObject.Name) ? (int) strlen(fileObject.Name) : 0;

            CK_ID objId = fileObject.Object;
            if (fileObject.ObjPtr &&
                (fileObject.ObjPtr->m_ObjectFlags & CK_OBJECT_ONLYFORFILEREFERENCE) != 0) {
                objId |= 0x800000;
                fileObject.ObjPtr->m_ObjectFlags &= ~CK_OBJECT_ONLYFORFILEREFERENCE;
            }

            parser->Write(&objId, sizeof(CK_ID));
            parser->Write(&fileObject.ObjectCid, sizeof(CK_CLASSID));
            parser->Write(&fileObject.FileIndex, sizeof(int));
            parser->Write(&nameSize, sizeof(int));
            if (nameSize > 0) {
                parser->Write(fileObject.Name, nameSize);
            }
        }
    }

    parser->Write(&pluginDepCount, sizeof(int));
    for (int i = 0; i < pluginDepCount; ++i) {
        CKFilePluginDependencies &pluginDep = m_PluginsDep[i];

        parser->Write(&pluginDep.m_PluginCategory, sizeof(int));

        int guidCount = pluginDep.m_Guids.Size();
        parser->Write(&guidCount, sizeof(int));
        if (guidCount > 0) {
            parser->Write(pluginDep.m_Guids.Begin(), guidCount * (int) sizeof(CKGUID));
        }
    }

    parser->Seek(0);
    if ((header.Part0.FileWriteMode & (CKFILE_WHOLECOMPRESSED | CKFILE_CHUNKCOMPRESSED_OLD)) != 0) {
        parser = parser->Pack(hdr1PackSize, m_Context->GetCompressionLevel());
        if (parser && (CKDWORD) parser->Size() < header.Part1.Hdr1UnPackSize) {
            header.Part0.Hdr1PackSize = parser->Size();
            delete hdr1BufferParser;
            hdr1BufferParser = parser;
        } else {
            delete parser;
            parser = hdr1BufferParser;
            header.Part0.Hdr1PackSize = header.Part1.Hdr1UnPackSize;
        }
    }

    char *dataBuffer = new char[dataUnPackSize];
    CKBufferParser *dataBufferParser = new CKBufferParser(dataBuffer, dataUnPackSize);
    dataBufferParser->m_Valid = TRUE;
    parser = dataBufferParser;
    if (!parser || !parser->m_Buffer) {
        if (hdr1BufferParser)
            delete hdr1BufferParser;
        if (parser)
            delete parser;

        m_Context->m_Saving = FALSE;
        m_Context->ExecuteManagersPostSave();

        return CKERR_OUTOFMEMORY;
    }

    if (savedManagerCount > 0) {
        for (int i = 0; i < savedManagerCount; ++i) {
            CKFileManagerData &managerData = m_ManagersData[i];
            parser->Write(&managerData.Manager, sizeof(CKGUID));
            parser->InsertChunk(managerData.data);
            delete managerData.data;
            managerData.data = nullptr;
        }
    }

    if (fileObjectCount > 0) {
        for (int i = 0; i < fileObjectCount; ++i) {
            CKFileObject &fileObject = m_FileObjects[i];
            parser->Write(&fileObject.PrePackSize, sizeof(int));
            parser->InsertChunk(fileObject.Data);
            delete fileObject.Data;
            fileObject.Data = nullptr;
        }
    }

    parser->Seek(0);
    if ((header.Part0.FileWriteMode & (CKFILE_WHOLECOMPRESSED | CKFILE_CHUNKCOMPRESSED_OLD)) != 0) {
        parser = parser->Pack((int) header.Part1.Hdr1UnPackSize, m_Context->GetCompressionLevel());
        if (parser && (CKDWORD) parser->Size() < header.Part1.DataUnPackSize) {
            header.Part1.DataPackSize = parser->Size();
            delete dataBufferParser;
            dataBufferParser = parser;
        } else {
            delete parser;
            parser = dataBufferParser;
            header.Part1.DataPackSize = header.Part1.DataUnPackSize;
        }
    }

    CKDWORD crc = CKComputeDataCRC((char *) &header.Part0, sizeof(CKFileHeaderPart0));
    crc = CKComputeDataCRC((char *) &header.Part1, sizeof(CKFileHeaderPart1), crc);
    crc = hdr1BufferParser->ComputeCRC(hdr1BufferParser->Size(), crc);
    crc = dataBufferParser->ComputeCRC(dataBufferParser->Size(), crc);
    header.Part0.Crc = crc;

    m_FileInfo.ProductVersion = header.Part1.ProductVersion;
    m_FileInfo.ProductBuild = header.Part1.ProductBuild;
    m_FileInfo.FileWriteMode = header.Part0.FileWriteMode;
    m_FileInfo.CKVersion = header.Part0.CKVersion;
    m_FileInfo.FileVersion = header.Part0.FileVersion;
    m_FileInfo.Hdr1PackSize = header.Part0.Hdr1PackSize;
    m_FileInfo.FileSize = sizeof(CKFileHeader) + header.Part0.Hdr1PackSize + header.Part1.DataPackSize;
    m_FileInfo.Hdr1UnPackSize = header.Part1.Hdr1UnPackSize;
    m_FileInfo.ManagerCount = header.Part1.ManagerCount;
    m_FileInfo.DataPackSize = header.Part1.DataPackSize;
    m_FileInfo.ObjectCount = header.Part1.ObjectCount;
    m_FileInfo.DataUnPackSize = header.Part1.DataUnPackSize;
    m_FileInfo.MaxIDSaved = header.Part1.MaxIDSaved;
    m_FileInfo.Crc = crc;
    WriteStats(interfaceDataSize);

    ClearData();

    m_Context->SetAutomaticLoadMode(CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID);
    m_Context->SetUserLoadCallback(nullptr, nullptr);

    FILE *fp = fopen(m_FileName, "wb");
    if (!fp) {
        delete hdr1BufferParser;
        delete dataBufferParser;

        m_Context->m_Saving = FALSE;
        m_Context->ExecuteManagersPostSave();

        return CKERR_CANTWRITETOFILE;
    }

    CKERROR err = CK_OK;

    hdr1BufferParser->Seek(0);
    dataBufferParser->Seek(0);
    if (fwrite(&header.Part0, sizeof(CKFileHeaderPart0), 1, fp) == 1 &&
        fwrite(&header.Part1, sizeof(CKFileHeaderPart1), 1, fp) == 1 &&
        fwrite(&hdr1BufferParser->m_Buffer[hdr1BufferParser->CursorPos()], hdr1BufferParser->Size(), 1, fp) == 1 &&
        fwrite(&dataBufferParser->m_Buffer[dataBufferParser->CursorPos()], dataBufferParser->Size(), 1, fp) == 1) {
        int includeFileCount = m_IncludedFiles.Size();
        for (int i = 0; i < includeFileCount; ++i) {
            XString &filename = m_IncludedFiles[i];
            VxMemoryMappedFile mmf(filename.Str());

            XString name(CKJustFile(filename.Str()));
            int length = name.Length();
            fwrite(&length, sizeof(int), 1, fp);
            if (length != 0) {
                fwrite(name.Str(), length, 1, fp);
            }

            if (mmf.GetErrorType() == VxMMF_NoError) {
                CKDWORD size = mmf.GetFileSize();
                fwrite(&size, sizeof(CKDWORD), 1, fp);
                fwrite(mmf.GetBase(), size, 1, fp);
            } else {
                CKDWORD stub = 0;
                fwrite(&stub, sizeof(CKDWORD), 1, fp);
            }
        }
        m_IncludedFiles.Clear();
    } else {
        err = CKERR_NOTENOUGHDISKPLACE;
    }

    delete hdr1BufferParser;
    delete dataBufferParser;

    fclose(fp);

    m_Context->ExecuteManagersPostSave();
    m_Context->m_Saving = FALSE;

    return err;
}

CKBOOL CKFile::IncludeFile(CKSTRING FileName, int SearchPathCategory) {
    if (!FileName || strlen(FileName) == 0)
        return FALSE;

    XString filename = FileName;
    if (SearchPathCategory <= -1 || m_Context->GetPathManager()->ResolveFileName(filename, SearchPathCategory) == CK_OK) {
        m_IncludedFiles.PushBack(filename);
        return TRUE;
    }

    return FALSE;
}

CKBOOL CKFile::IsObjectToBeSaved(CK_ID iID) {
    if (m_FileObjects.IsEmpty())
        return FALSE;

    for (XArray<CKFileObject>::Iterator it = m_FileObjects.Begin(); it != m_FileObjects.End(); ++it)
        if ((*it).Object == iID)
            return TRUE;

    return FALSE;
}

void CKFile::LoadAndSave(CKSTRING filename, CKSTRING filename_new) {
    CKAttributeManager *am = m_Context->GetAttributeManager();
    CKObjectArray array = CKObjectArray();
    if (!Load(filename, &array, CK_LOAD_DEFAULT)) {
        am->m_Saving = 1;
        m_Context->SetFileWriteMode((CK_FILE_WRITEMODE) CurrentFileWriteMode);
        if (!StartSave(filename_new, 0)) {
            SaveObjects(&array, 0xFFFFFFFF);
            EndSave();
        }
        am->m_Saving = 0;
    }
    array.Clear();
}

void CKFile::RemapManagerInt(CKGUID Manager, int *ConversionTable, int TableSize) {
    if (m_FileObjects.IsEmpty())
        return;

    for (XArray<CKFileObject>::Iterator it = m_FileObjects.Begin(); it != m_FileObjects.End(); ++it) {
        if (it->Data)
            it->Data->RemapManagerInt(Manager, ConversionTable, TableSize);
    }
}

XClassArray<CKFilePluginDependencies> *CKFile::GetMissingPlugins() {
    return &m_PluginsDep;
}

CKFile::CKFile(CKContext *Context)
    : m_SaveIDMax(0),
      m_SceneSaved(FALSE),
      m_FileName(nullptr),
      m_Context(Context),
      m_Flags(0),
      m_Parser(nullptr),
      m_MappedFile(nullptr),
      m_ReadFileDataDone(FALSE) {
}

CKFile::~CKFile() {
    ClearData();
    m_Context->SetAutomaticLoadMode(CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID);
    m_Context->SetUserLoadCallback(nullptr, nullptr);
}

void CKFile::FinishLoading(CKObjectArray *list, CKDWORD flags) {
    XBitArray exclusion;
    exclusion.Set(CKCID_PARAMETER);
    exclusion.Set(CKCID_PARAMETEROUT);
    exclusion.Set(CKCID_PARAMETERLOCAL);
    exclusion.Set(CKCID_BEHAVIOR);

    XBitArray inclusion;
    inclusion.Or(g_CKClassInfo[CKCID_BEOBJECT].Children);
    inclusion.Or(g_CKClassInfo[CKCID_OBJECTANIMATION].Children);
    inclusion.Or(g_CKClassInfo[CKCID_ANIMATION].Children);

    CKObjectManager *objectManager = m_Context->m_ObjectManager;
    objectManager->StartLoadSession(m_SaveIDMax + 1);

    int options = CK_OBJECTCREATION_NONAMECHECK;
    if (flags & (CK_LOAD_DODIALOG | CK_LOAD_AUTOMATICMODE | CK_LOAD_CHECKDUPLICATES)) {
        options = CK_OBJECTCREATION_ASK;
    }
    if (flags & CK_LOAD_AS_DYNAMIC_OBJECT) {
        options |= CK_OBJECTCREATION_DYNAMIC;
    }

    for (int i = 0; i < m_FileObjects.Size(); ++i) {
        CKFileObject *it = &m_FileObjects[i];
        m_IndexByClassId[it->ObjectCid].PushBack(i);
        if (it->Data && it->ObjectCid != CKCID_RENDERCONTEXT) {
            int id = *(int *) &it->Object;
            CKObject *obj = nullptr;
            if (id >= 0) {
                CK_CREATIONMODE res;
                obj = m_Context->CreateObject(it->ObjectCid, it->Name, (CK_OBJECTCREATION_OPTIONS) options, &res);
                it->Options = (res == CKLOAD_USECURRENT)
                                  ? CKFileObject::CK_FO_RENAMEOBJECT
                                  : CKFileObject::CK_FO_DEFAULT;
            } else {
                it->Object = -id;
                obj = ResolveReference(it);
                it->Options = CKFileObject::CK_FO_RENAMEOBJECT;
            }
            objectManager->RegisterLoadObject(obj, it->Object);
            it->ObjPtr = obj;
            it->CreatedObject = obj ? obj->GetID() : 0;
        }
    }

    if (!m_IndexByClassId[CKCID_LEVEL].IsEmpty()) {
        if (m_FileInfo.ProductVersion <= 1 && m_FileInfo.ProductBuild <= 0x2000000) {
            m_Context->m_PVInformation = 0;
        } else {
            m_Context->m_PVInformation = m_FileInfo.ProductVersion;
        }
    }

    if (!(m_Flags & CK_LOAD_ONLYBEHAVIORS)) {
        for (XArray<CKFileObject>::Iterator it = m_FileObjects.Begin(); it != m_FileObjects.End(); ++it) {
            if (it->Data) {
                it->Data->RemapObjects(m_Context);
            }
        }
        for (XArray<CKFileManagerData>::Iterator it = m_ManagersData.Begin(); it != m_ManagersData.End(); ++it) {
            if (it->data) {
                it->data->RemapObjects(m_Context);
            }
        }
    }

    if (!m_IndexByClassId[CKCID_LEVEL].IsEmpty()) {
        auto &levels = m_IndexByClassId[CKCID_LEVEL];
        int index = levels[0];
        CKFileObject &fileObject = m_FileObjects[index];
        CKLevel *level = (CKLevel *) fileObject.ObjPtr;
        if (level) {
            if (!m_Context->GetCurrentLevel()) {
                m_Context->SetCurrentLevel(level);
            }
        }
    }

    int count = 0;

    if (!(m_Flags & CK_LOAD_ONLYBEHAVIORS)) {
        bool hasGridManager = false;
        for (XArray<CKFileManagerData>::Iterator it = m_ManagersData.Begin(); it != m_ManagersData.End(); ++it) {
            CKBaseManager *manager = m_Context->GetManagerByGuid(it->Manager);
            if (manager) {
                manager->LoadData(it->data, this);
                if (manager->GetGuid() == GRID_MANAGER_GUID) {
                    hasGridManager = true;
                }

                if (it->data) {
                    delete it->data;
                    it->data = nullptr;
                }
            }
        }

        if (hasGridManager) {
            CKBaseManager *manager = m_Context->GetManagerByGuid(GRID_MANAGER_GUID);
            manager->LoadData(nullptr, this);
        }

        bool levelLoaded = false;
        for (XArray<CKFileObject>::Iterator it = m_FileObjects.Begin(); it != m_FileObjects.End(); ++it) {
            if (!it->Data)
                continue;

            if (it->Options != CKFileObject::CK_FO_DEFAULT)
                continue;

            if (exclusion.IsSet(it->ObjectCid))
                continue;

            CKObject *obj = it->ObjPtr;
            if (!obj)
                continue;

            if (CKIsChildClassOf(obj, CKCID_LEVEL)) {
                if (levelLoaded)
                    continue;
                levelLoaded = true;
            }

            obj->Load(it->Data, this);
            ++count;

            if (m_Context->m_UICallBackFct) {
                CKUICallbackStruct cbs;
                cbs.Reason = CKUIM_LOADSAVEPROGRESS;
                cbs.Param1 = count;
                cbs.Param2 = m_FileObjects.Size();
                m_Context->m_UICallBackFct(cbs, m_Context->m_InterfaceModeData);
            }

            if (list && inclusion.IsSet(it->ObjectCid)) {
                list->InsertRear(obj);
            }
        }

        for (XArray<int>::Iterator iit = m_IndexByClassId[CKCID_PARAMETERLOCAL].Begin();
             iit != m_IndexByClassId[CKCID_PARAMETERLOCAL].End(); ++iit) {
            CKFileObject *it = &m_FileObjects[*iit];
            if (!it->Data || it->Options != CKFileObject::CK_FO_DEFAULT)
                continue;

            CKObject *obj = it->ObjPtr;
            if (obj) {
                obj->Load(it->Data, this);
                ++count;

                if (m_Context->m_UICallBackFct) {
                    CKUICallbackStruct cbs;
                    cbs.Reason = CKUIM_LOADSAVEPROGRESS;
                    cbs.Param1 = count;
                    cbs.Param2 = m_FileObjects.Size();
                    m_Context->m_UICallBackFct(cbs, m_Context->m_InterfaceModeData);
                }

                if (list && inclusion.IsSet(it->ObjectCid)) {
                    list->InsertRear(obj);
                }
            }

            if (it->Data) {
                delete it->Data;
                it->Data = nullptr;
            }
        }

        for (XArray<int>::Iterator iit = m_IndexByClassId[CKCID_PARAMETER].Begin(); iit != m_IndexByClassId[CKCID_PARAMETER].End(); ++iit) {
            CKFileObject *it = &m_FileObjects[*iit];
            if (!it->Data || it->Options != CKFileObject::CK_FO_DEFAULT)
                continue;

            CKObject *obj = it->ObjPtr;
            if (obj) {
                obj->Load(it->Data, this);
                ++count;

                if (m_Context->m_UICallBackFct) {
                    CKUICallbackStruct cbs;
                    cbs.Reason = CKUIM_LOADSAVEPROGRESS;
                    cbs.Param1 = count;
                    cbs.Param2 = m_FileObjects.Size();
                    m_Context->m_UICallBackFct(cbs, m_Context->m_InterfaceModeData);
                }

                if (list && inclusion.IsSet(it->ObjectCid)) {
                    list->InsertRear(obj);
                }
            }

            if (it->Data) {
                delete it->Data;
                it->Data = nullptr;
            }
        }

        for (XArray<int>::Iterator iit = m_IndexByClassId[CKCID_PARAMETEROUT].Begin();
             iit != m_IndexByClassId[CKCID_PARAMETEROUT].End(); ++iit) {
            CKFileObject *it = &m_FileObjects[*iit];
            if (!it->Data || it->Options != CKFileObject::CK_FO_DEFAULT)
                continue;

            CKObject *obj = it->ObjPtr;
            if (obj) {
                obj->Load(it->Data, this);
                ++count;

                if (m_Context->m_UICallBackFct) {
                    CKUICallbackStruct cbs;
                    cbs.Reason = CKUIM_LOADSAVEPROGRESS;
                    cbs.Param1 = count;
                    cbs.Param2 = m_FileObjects.Size();
                    m_Context->m_UICallBackFct(cbs, m_Context->m_InterfaceModeData);
                }

                if (list && inclusion.IsSet(it->ObjectCid)) {
                    list->InsertRear(obj);
                }
            }

            if (it->Data) {
                delete it->Data;
                it->Data = nullptr;
            }
        }
    }

    for (XArray<int>::Iterator iit = m_IndexByClassId[CKCID_BEHAVIOR].Begin();
         iit != m_IndexByClassId[CKCID_BEHAVIOR].End(); ++iit) {
        CKFileObject *it = &m_FileObjects[*iit];
        if (!it->Data || it->Options != CKFileObject::CK_FO_DEFAULT)
            continue;

        CKBehavior *beh = (CKBehavior *) it->ObjPtr;
        if (beh) {
            beh->Load(it->Data, this);
            ++count;

            if (m_Context->m_UICallBackFct) {
                CKUICallbackStruct cbs;
                cbs.Reason = CKUIM_LOADSAVEPROGRESS;
                cbs.Param1 = count;
                cbs.Param2 = m_FileObjects.Size();
                m_Context->m_UICallBackFct(cbs, m_Context->m_InterfaceModeData);
            }

            if (list && inclusion.IsSet(it->ObjectCid)) {
                list->InsertRear(beh);
            }
        }

        if ((beh->GetFlags() & CKBEHAVIOR_TOPMOST) || beh->GetType() == CKBEHAVIORTYPE_SCRIPT) {
            if (list) {
                list->InsertRear(beh);
            }
        }

        if (it->Data) {
            delete it->Data;
            it->Data = nullptr;
        }
    }

    if (!(m_Flags & CK_LOAD_ONLYBEHAVIORS)) {
        for (XArray<int>::Iterator iit = m_IndexByClassId[CKCID_INTERFACEOBJECTMANAGER].Begin();
             iit != m_IndexByClassId[CKCID_INTERFACEOBJECTMANAGER].End(); ++iit) {
            CKFileObject *it = &m_FileObjects[*iit];
            CKInterfaceObjectManager *obj = (CKInterfaceObjectManager *) it->ObjPtr;
            if (obj && list) {
                list->InsertRear(obj);
            }
        }

        for (XArray<CKFileObject>::Iterator it = m_FileObjects.Begin(); it != m_FileObjects.End(); ++it) {
            if (it->ObjPtr && it->Data && it->Options == CKFileObject::CK_FO_DEFAULT) {
                if (CKIsChildClassOf(it->ObjectCid, CKCID_BEOBJECT)) {
                    CKBeObject *beo = (CKBeObject *) it->ObjPtr;
                    beo->ApplyOwner();
                }
            }
        }

        for (XArray<CKFileObject>::Iterator it = m_FileObjects.Begin(); it != m_FileObjects.End(); ++it) {
            if (it->ObjPtr && it->Options == CKFileObject::CK_FO_DEFAULT) {
                CKObject *obj = it->ObjPtr;
                obj->PostLoad();
            }
        }

        for (XArray<CKFileObject>::Iterator it = m_FileObjects.Begin(); it != m_FileObjects.End(); ++it) {
            CKObject *obj = m_Context->GetObject(it->CreatedObject);
            if (obj) {
                if (CKIsChildClassOf(it->ObjectCid, CKCID_BEHAVIOR)) {
                    CKBehavior *beh = (CKBehavior *) it->ObjPtr;
                    beh->ApplyPatchLoad();
                    beh->CallCallbackFunction(CKM_BEHAVIORLOAD);
                }
                if (it->Data && CKIsChildClassOf(it->ObjectCid, CKCID_BEOBJECT)) {
                    CKBeObject *beo = (CKBeObject *) obj;
                    beo->ApplyPatchForOlderVersion(m_FileObjects.Size(), it);
                }
            }
        }
    }

    objectManager->EndLoadSession();
}

void CKFile::WriteStats(int InterfaceDataSize) {
    /* Empty */
}

CKObject *CKFile::ResolveReference(CKFileObject *Data) {
    if (!CKIsChildClassOf(Data->ObjectCid, CKCID_PARAMETER))
        return nullptr;

    if (!Data->Data)
        return nullptr;

    CKStateChunk *chunk = Data->Data;
    chunk->StartRead();
    if (!chunk->SeekIdentifier(64))
        return nullptr;

    CKParameterManager *pm = m_Context->GetParameterManager();
    CKGUID paramGuid = chunk->ReadGuid();
    CKParameterType paramType = pm->ParameterGuidToType(paramGuid);
    const int paramCount = m_Context->GetObjectsCountByClassID(Data->ObjectCid);
    if (paramCount <= 0)
        return nullptr;

    CK_ID *paramIds = m_Context->GetObjectsListByClassID(Data->ObjectCid);
    CKParameter *target = nullptr;
    for (int i = 0; i < paramCount; ++i) {
        CKParameter *param = (CKParameter *) m_Context->GetObject(paramIds[i]);
        if (param && param->GetName() && param->GetType() == paramType && strcmp(param->GetName(), Data->Name) == 0) {
            target = param;
            break;
        }
    }
    return target;
}
