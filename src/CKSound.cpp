#include "CKSound.h"

#include "CKFile.h"
#include "CKStateChunk.h"
#include "CKPathManager.h"

extern CKSTRING CKJustFile(CKSTRING path);

CK_CLASSID CKSound::m_ClassID = CKCID_SOUND;

CK_SOUND_SAVEOPTIONS CKSound::GetSaveOptions() {
    return m_SaveOptions;
}

void CKSound::SetSaveOptions(CK_SOUND_SAVEOPTIONS Options) {
    m_SaveOptions = Options;
}

CKSound::CKSound(CKContext *Context, CKSTRING name) : CKBeObject(Context, name) {
    m_FileName = nullptr;
    m_SaveOptions = CKSOUND_USEGLOBAL;
}

CKSound::~CKSound() {
    delete[] m_FileName;
}

CK_CLASSID CKSound::GetClassID() {
    return m_ClassID;
}

CKStateChunk *CKSound::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKBeObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_SOUND, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    CK_SOUND_SAVEOPTIONS saveOptions = m_SaveOptions;
    if (saveOptions == CKSOUND_USEGLOBAL)
        saveOptions = m_Context->m_GlobalSoundsSaveOptions;

    if (file && saveOptions == CKSOUND_INCLUDEORIGINALFILE)
        file->IncludeFile(m_FileName, SOUND_PATH_IDX);

    chunk->WriteIdentifier(CK_STATESAVE_SOUNDFILENAME);
    chunk->WriteInt(m_SaveOptions);
    chunk->WriteString(m_FileName ? CKJustFile(m_FileName) : NULL); // Write the file name without path

    if (GetClassID() == CKCID_SOUND)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();
    return chunk;
}

CKERROR CKSound::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    CKBeObject::Load(chunk, file);

    if (chunk->SeekIdentifier(CK_STATESAVE_SOUNDFILENAME)) {
        delete[] m_FileName;
        m_FileName = nullptr;

        m_SaveOptions = (CK_SOUND_SAVEOPTIONS) chunk->ReadDword();

        chunk->ReadString(&m_FileName);
    }

    return CK_OK;
}

int CKSound::GetMemoryOccupation() {
    return CKBeObject::GetMemoryOccupation() + 8;
}

CKSTRING CKSound::GetClassName() {
    return "Sound";
}

int CKSound::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKSound::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKSound::Register() {
    CKPARAMETERFROMCLASS(CKSound, CKPGUID_SOUND);
}

CKSound *CKSound::CreateInstance(CKContext *Context) {
    return new CKSound(Context);
}
