#include "CKMidiSound.h"

#include "CKPathManager.h"
#include "CKMidiManager.h"
#include "CKStateChunk.h"

CK_CLASSID CKMidiSound::m_ClassID = CKCID_MIDISOUND;

CKERROR CKMidiSound::SetSoundFileName(CKSTRING filename) {
    XString path = filename ? filename : "";
    m_Context->m_PathManager->ResolveFileName(path, SOUND_PATH_IDX);

    delete[] m_FileName;
    m_FileName = CKStrdup(path.Str() ? path.Str() : "");

    if (!m_MidiManager) {
        return CKERR_NOTINITIALIZED;
    }

    return m_MidiManager->SetSoundFileName(m_Source, path.Str() ? path.Str() : "");
}

CKSTRING CKMidiSound::GetSoundFileName() {
    return m_FileName;
}

CKDWORD CKMidiSound::GetCurrentPos() {
    if (!m_MidiManager) {
        return 0;
    }

    CKDWORD ticks = 0;
    m_MidiManager->Time(m_Source, &ticks);
    return m_MidiManager->TicksToMillisecs(m_Source, ticks);
}

CKERROR CKMidiSound::Play() {
    if (!m_MidiManager) {
        return CKERR_NOTINITIALIZED;
    }
    return m_MidiManager->Play(m_Source);
}

CKERROR CKMidiSound::Stop() {
    if (!m_MidiManager) {
        return CKERR_NOTINITIALIZED;
    }
    return m_MidiManager->Stop(m_Source);
}

CKERROR CKMidiSound::Pause(CKBOOL pause) {
    if (!m_MidiManager) {
        return CKERR_NOTINITIALIZED;
    }
    if (pause) {
        return m_MidiManager->Pause(m_Source);
    } else {
        return m_MidiManager->Restart(m_Source);
    }
}

CKBOOL CKMidiSound::IsPlaying() {
    if (!m_MidiManager) {
        return FALSE;
    }
    return m_MidiManager->IsPlaying(m_Source);
}

CKBOOL CKMidiSound::IsPaused() {
    if (!m_MidiManager) {
        return FALSE;
    }
    return m_MidiManager->IsPaused(m_Source);
}

CKMidiSound::CKMidiSound(CKContext *Context, CKSTRING name) : CKSound(Context, name) {
    m_MidiManager = (CKMidiManager *)m_Context->GetManagerByGuid(MIDI_MANAGER_GUID);
    if (m_MidiManager) {
        m_Source = m_MidiManager->Create(m_Context->GetMainWindow());
    } else {
        m_Source = nullptr;
    }
}

CKMidiSound::~CKMidiSound() {
    if (m_MidiManager) {
        m_MidiManager->Release(m_Source);
    }
}

CK_CLASSID CKMidiSound::GetClassID() {
    return m_ClassID;
}

CKStateChunk *CKMidiSound::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKSound::Save(file, flags);
    if (!file && (flags & CK_STATESAVE_MIDISOUNDONLY) == 0)
        return baseChunk;

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_MIDISOUND, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    if (GetClassID() == CKCID_MIDISOUND)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();
    return chunk;
}

CKERROR CKMidiSound::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    CKSound::Load(chunk, file);

    if (chunk->SeekIdentifier(CK_STATESAVE_MIDISOUNDFILE)) {
        char *tempFileName = nullptr;
        chunk->ReadString(&tempFileName);
        SetSoundFileName(tempFileName);

        delete[] tempFileName;
    } else {
        SetSoundFileName(m_FileName);
    }

    return CK_OK;
}

int CKMidiSound::GetMemoryOccupation() {
    return CKSound::GetMemoryOccupation() + (int) (sizeof(CKMidiSound) - sizeof(CKSound));
}

CKSTRING CKMidiSound::GetClassName() {
    return "Midi Sound";
}

int CKMidiSound::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKMidiSound::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKMidiSound::Register() {
    CKPARAMETERFROMCLASS(CKMidiSound, CKPGUID_MIDISOUND);
}

CKMidiSound *CKMidiSound::CreateInstance(CKContext *Context) {
    return new CKMidiSound(Context);
}
