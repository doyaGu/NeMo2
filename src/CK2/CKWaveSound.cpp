#include "CKWaveSound.h"

#include "CKPluginManager.h"
#include "CKPathManager.h"
#include "CKTimeManager.h"
#include "CKSoundReader.h"
#include "CK3dEntity.h"
#include "CKStateChunk.h"

void PositionSource(
    CKSoundManager *soundManager,
    CKSOUNDHANDLE soundSource,
    CK3dEntity *entity,
    VxVector &position,
    VxVector &direction,
    VxVector &lastPosition,
    float timeDeltaMs) {
    if (!soundManager) return;

    VxVector worldPos = position;
    VxVector worldDir = direction;

    if (entity) {
        entity->Transform(&worldPos, &position);
    }

    VxVector velocity;
    if (timeDeltaMs > 0.0f) {
        velocity = (worldPos - lastPosition) / (timeDeltaMs * 0.001f);
    } else {
        velocity = VxVector(0.0f, 0.0f, 0.0f);
    }

    if (entity) {
        entity->TransformVector(&worldDir, &direction);
    }
    worldDir.Normalize();

    CKWaveSound3DSettings sound3DSettings;
    sound3DSettings.m_Position = worldPos;
    sound3DSettings.m_Velocity = velocity;
    sound3DSettings.m_OrientationDir = worldDir;

    soundManager->Update3DSettings(
        soundSource,
        (CK_SOUNDMANAGER_CAPS) (CK_WAVESOUND_3DSETTINGS_POSITION |
            CK_WAVESOUND_3DSETTINGS_VELOCITY |
            CK_WAVESOUND_3DSETTINGS_ORIENTATION),
        sound3DSettings,
        TRUE);

    lastPosition = worldPos;
}

CK_CLASSID CKWaveSound::m_ClassID = CKCID_WAVESOUND;

CKSOUNDHANDLE CKWaveSound::PlayMinion(CKBOOL Background, CK3dEntity *Ent, VxVector *Position, VxVector *Direction, float MinDelay) {
    if (!m_SoundManager || !m_Source)
        return nullptr;

    if (m_State & CK_WAVESOUND_NODUPLICATE)
        return nullptr;

    // Create minion through sound manager
    SoundMinion *minion = m_SoundManager->CreateMinion(m_Source, MinDelay);
    if (!minion)
        return nullptr;

    minion->m_OriginalSound = GetID();

    CK_WAVESOUND_TYPE soundType = GetType();
    if (Background) {
        if (soundType != CK_WAVESOUND_BACKGROUND) {
            m_SoundManager->SetType(minion->m_Source, CK_WAVESOUND_BACKGROUND);
        }
        minion->m_Entity = 0;
    } else {
        if (soundType != CK_WAVESOUND_POINT) {
            m_SoundManager->SetType(minion->m_Source, CK_WAVESOUND_POINT);
        }

        // Configure 2D audio settings
        CKWaveSoundSettings settings;
        settings.m_Gain = 1.0f;
        settings.m_Eq = 1.0f;
        settings.m_Pitch = 1.0f;
        settings.m_Priority = 0.5f;
        settings.m_Pan = 0.0f;
        m_SoundManager->UpdateSettings(m_Source, CK_WAVESOUND_SETTINGS_ALL, settings, FALSE);
        m_SoundManager->UpdateSettings(minion->m_Source, CK_WAVESOUND_SETTINGS_ALL, settings, TRUE);

        minion->m_Entity = Ent ? Ent->GetID() : 0;

        // Set position parameters
        VxVector pos = (Position) ? *Position : VxVector(0.0f, 0.0f, 0.0f);
        VxVector dir = (Direction) ? *Direction : VxVector(0.0f, 0.0f, 1.0f);
        minion->m_Position = pos;
        minion->m_Direction = dir;

        CKWaveSound3DSettings settings3D;
        m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_ALL, settings3D, FALSE);
        settings3D.m_Position = pos;
        settings3D.m_OrientationDir = dir;
        m_SoundManager->Update3DSettings(minion->m_Source, CK_WAVESOUND_3DSETTINGS_ALL, settings3D, TRUE);
    }

    m_SoundManager->SetPlayPosition(minion->m_Source, 0);
    // CK2.dll passes CKWaveSound*=NULL and the SoundMinion* wrapper as the source argument.
    m_SoundManager->Play(nullptr, minion, FALSE);

    static int s_MinionPlayChecksLeft = 8;
    if (s_MinionPlayChecksLeft > 0) {
        --s_MinionPlayChecksLeft;
        if (!m_SoundManager->IsPlaying(minion->m_Source)) {
            m_Context->OutputToConsoleEx("CKError : Minion Play() did not start playback (file '%s')", m_FileName ? m_FileName : "");
        }
    }
    return minion->m_Source;
}

CKERROR CKWaveSound::SetSoundFileName(const CKSTRING FileName) {
    if (!FileName)
        return CKERR_INVALIDPARAMETER;
    XString filename = FileName;
    m_Context->GetPathManager()->ResolveFileName(filename, SOUND_PATH_IDX);
    delete[] m_FileName;
    m_FileName = CKStrdup(filename.Str() ? filename.Str() : "");
    return CK_OK;
}

CKSTRING CKWaveSound::GetSoundFileName() {
    return m_FileName;
}

int CKWaveSound::GetSoundLength() {
    if (!m_Source)
        return 0;

    if (!GetFileStreaming() || (m_State & CK_WAVESOUND_STREAMFULLYLOADED)) {
        if (m_SoundManager) {
            int waveSize = m_SoundManager->GetWaveSize(m_Source);
            if (waveSize > 0 && m_WaveFormat.nAvgBytesPerSec > 0) {
                int length = (int) ((waveSize * 1000.0) / m_WaveFormat.nAvgBytesPerSec);
                return length;
            }
        }
        return 0;
    }

    if (m_SoundReader) {
        int duration = m_SoundReader->GetDuration();
        if (duration > 0)
            return duration;
    }

    return m_Duration;
}

CKERROR CKWaveSound::GetSoundFormat(CKWaveFormat &Format) {
    Format = m_WaveFormat;
    return CK_OK;
}

CK_WAVESOUND_TYPE CKWaveSound::GetType() {
    if (m_SoundManager) {
        return m_SoundManager->GetType(m_Source);
    }
    return CK_WAVESOUND_BACKGROUND;
}

void CKWaveSound::SetType(CK_WAVESOUND_TYPE Type) {
    if (m_SoundManager && (m_State & CK_WAVESOUND_ALLTYPE) != Type) {
        SaveSettings();
        m_State &= ~CK_WAVESOUND_ALLTYPE;
        m_State |= Type;
        if (m_SoundManager->GetCaps() & CK_SOUNDMANAGER_ONFLYTYPE)
            m_SoundManager->SetType(m_Source, Type);
        else
            Recreate();
    }
}

CKDWORD CKWaveSound::GetState() {
    return m_State;
}

void CKWaveSound::SetState(CKDWORD State) {
    SetFileStreaming(State & CK_WAVESOUND_FILESTREAMED, FALSE);
    SetType((CK_WAVESOUND_TYPE) (State & CK_WAVESOUND_ALLTYPE));
    if (State & CK_WAVESOUND_COULDBERESTARTED)
        m_State |= CK_WAVESOUND_COULDBERESTARTED;
    else
        m_State &= ~CK_WAVESOUND_COULDBERESTARTED;
    if (State & CK_WAVESOUND_NODUPLICATE)
        m_State |= CK_WAVESOUND_NODUPLICATE;
    else
        m_State &= ~CK_WAVESOUND_NODUPLICATE;
}

void CKWaveSound::SetPriority(float Priority) {
    if (m_SoundManager) {
        CKWaveSoundSettings settings;
        settings.m_Gain = 1.0f;
        settings.m_Eq = 1.0f;
        settings.m_Pitch = 1.0f;
        settings.m_Priority = Priority;
        settings.m_Pan = 0.0f;
        m_SoundManager->UpdateSettings(m_Source, CK_WAVESOUND_SETTINGS_PRIORITY, settings, TRUE);
    }
}

float CKWaveSound::GetPriority() {
    if (!m_SoundManager)
        return 0.0f;

    CKWaveSoundSettings settings;
    settings.m_Gain = 1.0f;
    settings.m_Eq = 1.0f;
    settings.m_Priority = 0.5f;
    settings.m_Pitch = 1.0f;
    settings.m_Pan = 0.0f;
    m_SoundManager->UpdateSettings(m_Source, CK_WAVESOUND_SETTINGS_PRIORITY, settings, FALSE);
    return settings.m_Priority;
}

void CKWaveSound::SetLoopMode(CKBOOL Enabled) {
    if (Enabled)
        m_State |= CK_WAVESOUND_LOOPED;
    else
        m_State &= ~CK_WAVESOUND_LOOPED;
    if (IsPlaying() && (!GetFileStreaming() || (m_State & CK_WAVESOUND_STREAMFULLYLOADED))) {
        if (m_SoundManager)
            m_SoundManager->Play(this, m_Source, Enabled);
    }
}

CKBOOL CKWaveSound::GetLoopMode() {
    return (m_State & CK_WAVESOUND_LOOPED) != 0;
}

CKERROR CKWaveSound::SetFileStreaming(CKBOOL Enabled, CKBOOL RecreateSound) {
    CKBOOL isCurrentStreaming = (m_State & CK_WAVESOUND_FILESTREAMED) ? TRUE : FALSE;
    if (RecreateSound || (Enabled != isCurrentStreaming)) {
        if (Enabled) {
            m_State |= CK_WAVESOUND_FILESTREAMED;
        } else {
            m_State &= ~CK_WAVESOUND_FILESTREAMED;
        }

        SaveSettings();

        if (RecreateSound) {
            return Recreate();
        }
    }

    return CK_OK;
}

CKBOOL CKWaveSound::GetFileStreaming() {
    return (m_State & CK_WAVESOUND_FILESTREAMED) != 0;
}

void CKWaveSound::Play(float FadeIn, float FinalGain) {
    if (!m_SoundManager)
        return;

    if (!m_Source) {
        Recreate(FALSE);
        if (!m_Source)
            return;
    }

    m_State &= ~(CK_WAVESOUND_FADEIN | CK_WAVESOUND_FADEOUT);

    m_CurrentTime = 0.0f;
    m_FadeTime = FadeIn;

    if (FadeIn > 0.0f) {
        m_State |= CK_WAVESOUND_FADEIN;
        m_FinalGain = FinalGain;
        InternalSetGain(0.0f);
    }

    if (m_State & CK_WAVESOUND_NEEDREWIND) {
        Rewind();
    }

    m_State &= ~CK_WAVESOUND_PAUSED;

    if (GetFileStreaming() && !(m_State & CK_WAVESOUND_STREAMFULLYLOADED)) {
        if (m_SoundReader && m_SoundReader->Play() == CK_OK) {
            WriteDataFromReader();
            m_SoundManager->Play(this, m_Source, TRUE);
        }
    } else {
        if (m_State & CK_WAVESOUND_COULDBERESTARTED) {
            Rewind();
        }
        m_SoundManager->Play(this, m_Source, m_State & CK_WAVESOUND_LOOPED);
    }

    static int s_PlayChecksLeft = 8;
    if (s_PlayChecksLeft > 0) {
        --s_PlayChecksLeft;
        if (m_Source && !m_SoundManager->IsPlaying(m_Source)) {
            m_Context->OutputToConsoleEx("CKError : Play() returned but source is not playing (file '%s')", m_FileName ? m_FileName : "");
        }
    }

}

void CKWaveSound::Resume() {
    if (!m_SoundManager)
        return;

    if (GetFileStreaming() && !(m_State & CK_WAVESOUND_STREAMFULLYLOADED)) {
        if (m_SoundReader) {
            m_SoundReader->Resume();
            m_SoundManager->Play(this, m_Source, TRUE);
        }
    } else {
        m_SoundManager->Play(this, m_Source, GetLoopMode());
    }

    m_State &= ~CK_WAVESOUND_PAUSED;
}

void CKWaveSound::Rewind() {
    if (!m_SoundManager || !m_Source)
        return;

    m_SoundManager->SetPlayPosition(m_Source, 0);

    if (GetFileStreaming() && !(m_State & CK_WAVESOUND_STREAMFULLYLOADED)) {
        if (m_SoundReader) {
            m_SoundReader->Seek(0);
        }

        m_State &= ~CK_WAVESOUND_STREAMOVERLAP;

        m_BufferPos = -1;
        m_DataRead = 0;
        m_DataPlayed = 0;
        m_OldCursorPos = 0;
    }

    m_State &= ~CK_WAVESOUND_NEEDREWIND;
}

void CKWaveSound::Stop(float FadeOut) {
    if (!m_Source)
        return;

    if (FadeOut == 0.0f)
        InternalStop();

    if (m_State & CK_WAVESOUND_FADEIN) {
        const float newFadeTime = FadeOut + m_CurrentTime;
        const float remainingFade = m_FadeTime - m_CurrentTime;

        m_State &= ~CK_WAVESOUND_FADEIN;
        m_CurrentTime = (remainingFade / m_FadeTime) * newFadeTime;
        m_FadeTime = newFadeTime;
    } else {
        m_FadeTime = FadeOut;
        m_CurrentTime = 0.0f;
    }

    if (m_FadeTime > 0.0f)
        m_State |= CK_WAVESOUND_FADEOUT;

    m_State &= ~CK_WAVESOUND_PAUSED;
}

void CKWaveSound::Pause() {
    if (m_SoundReader) {
        m_SoundReader->Pause();
    }
    if (m_SoundManager) {
        m_SoundManager->Pause(this, m_Source);
    }
    m_State |= CK_WAVESOUND_PAUSED;
}

CKBOOL CKWaveSound::IsPlaying() {
    if (m_SoundManager)
        return m_SoundManager->IsPlaying(m_Source);
    return FALSE;
}

CKBOOL CKWaveSound::IsPaused() {
    return (m_State & CK_WAVESOUND_PAUSED) != 0;
}

void CKWaveSound::SetGain(float Gain) {
    m_FinalGain = Gain;
    InternalSetGain(Gain);
}

float CKWaveSound::GetGain() {
    if (m_SoundManager)
        return m_FinalGain;
    return 0.0f;
}

void CKWaveSound::SetPitch(float Rate) {
    if (m_SoundManager) {
        CKWaveSoundSettings settings;
        settings.m_Gain = 1.0f;
        settings.m_Eq = 1.0f;
        settings.m_Pan = 0.0f;
        settings.m_Priority = 0.5f;
        settings.m_Pitch = Rate;
        m_SoundManager->UpdateSettings(m_Source, CK_WAVESOUND_SETTINGS_PITCH, settings, TRUE);
    }
}

float CKWaveSound::GetPitch() {
    if (!m_SoundManager)
        return 0.0f;

    CKWaveSoundSettings settings;
    settings.m_Gain = 1.0f;
    settings.m_Eq = 1.0f;
    settings.m_Priority = 0.5f;
    settings.m_Pitch = 1.0f;
    settings.m_Pan = 0.0f;
    m_SoundManager->UpdateSettings(m_Source, CK_WAVESOUND_SETTINGS_PITCH, settings, FALSE);
    return settings.m_Pitch;
}

void CKWaveSound::SetPan(float Pan) {
    if (m_SoundManager) {
        CKWaveSoundSettings settings;
        settings.m_Gain = 1.0f;
        settings.m_Eq = 1.0f;
        settings.m_Pitch = 1.0f;
        settings.m_Priority = 0.5f;
        settings.m_Pan = Pan;
        m_SoundManager->UpdateSettings(m_Source, CK_WAVESOUND_SETTINGS_PAN, settings, TRUE);
    }
}

float CKWaveSound::GetPan() {
    if (!m_SoundManager)
        return 0.0f;

    CKWaveSoundSettings settings;
    settings.m_Gain = 1.0f;
    settings.m_Eq = 1.0f;
    settings.m_Priority = 0.5f;
    settings.m_Pitch = 1.0f;
    settings.m_Pan = 0.0f;
    m_SoundManager->UpdateSettings(m_Source, CK_WAVESOUND_SETTINGS_PAN, settings, FALSE);
    return settings.m_Pan;
}

CKSOUNDHANDLE CKWaveSound::GetSource() {
    return m_Source;
}

void CKWaveSound::PositionSound(CK3dEntity *Object, VxVector *Position, VxVector *Direction, CKBOOL Commit) {
    m_AttachedObject = Object ? Object->GetID() : 0;
    if (Position)
        m_Position = *Position;
    if (Direction)
        m_Direction = *Direction;
    m_State |= CK_WAVESOUND_HASMOVED;
    if (Commit)
        UpdatePosition(0.0f);
}

CK3dEntity *CKWaveSound::GetAttachedEntity() {
    return (CK3dEntity *) m_Context->GetObject(m_AttachedObject);
}

void CKWaveSound::GetPosition(VxVector &Pos) {
    Pos = m_Position;
}

void CKWaveSound::GetDirection(VxVector &Dir) {
    Dir = m_Direction;
}

void CKWaveSound::GetSound3DInformation(VxVector &Pos, VxVector &Dir, float &DistanceFromListener) {
    if (m_SoundManager) {
        Pos = m_Position;
        Dir = m_Direction;
        CK3dEntity *listener = m_SoundManager->GetListener();
        if (listener) {
            CK3dEntity *entity = GetAttachedEntity();
            VxVector dest;
            listener->Transform(&dest, &Pos, entity);
            DistanceFromListener = SquareMagnitude(dest);
        }
    }
}

void CKWaveSound::SetCone(float InAngle, float OutAngle, float OutsideGain) {
    if (m_SoundManager) {
        CKWaveSound3DSettings settings;
        settings.m_InAngle = InAngle;
        settings.m_OutAngle = OutAngle;
        settings.m_OutsideGain = OutsideGain;
        m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_CONE, settings, TRUE);
    }
}

void CKWaveSound::GetCone(float *InAngle, float *OutAngle, float *OutsideGain) {
    if (m_SoundManager) {
        CKWaveSound3DSettings settings;
        m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_CONE, settings, FALSE);
        if (InAngle)
            *InAngle = settings.m_InAngle;
        if (OutAngle)
            *OutAngle = settings.m_OutAngle;
        if (OutsideGain)
            *OutsideGain = settings.m_OutsideGain;
    }
}

void CKWaveSound::SetMinMaxDistance(float MinDistance, float MaxDistance, CKDWORD MaxDistanceBehavior) {
    if (m_SoundManager) {
        CKWaveSound3DSettings settings;
        settings.m_MinDistance = MinDistance;
        settings.m_MaxDistance = MaxDistance;
        settings.m_MuteAfterMax = (CKWORD) MaxDistanceBehavior;
        m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_MINMAXDISTANCE, settings, TRUE);
    }
}

void CKWaveSound::GetMinMaxDistance(float *MinDistance, float *MaxDistance, CKDWORD *MaxDistanceBehavior) {
    if (m_SoundManager) {
        CKWaveSound3DSettings settings;
        m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_MINMAXDISTANCE, settings, FALSE);
        if (MinDistance)
            *MinDistance = settings.m_MinDistance;
        if (MaxDistance)
            *MaxDistance = settings.m_MaxDistance;
        if (MaxDistanceBehavior)
            *MaxDistanceBehavior = settings.m_MuteAfterMax;
    }
}

void CKWaveSound::SetVelocity(VxVector &Pos) {
    if (m_SoundManager) {
        CKWaveSound3DSettings settings;
        settings.m_Velocity = Pos;
        m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_VELOCITY, settings, TRUE);
    }
}

void CKWaveSound::GetVelocity(VxVector &Pos) {
    if (m_SoundManager) {
        CKWaveSound3DSettings settings;
        m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_VELOCITY, settings, FALSE);
        Pos = settings.m_Velocity;
    }
}

void CKWaveSound::SetOrientation(VxVector &Dir, VxVector &Up) {
    if (m_SoundManager) {
        CKWaveSound3DSettings settings;
        settings.m_OrientationDir = Dir;
        settings.m_OrientationUp = Up;
        m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_ORIENTATION, settings, TRUE);
    }
}

void CKWaveSound::GetOrientation(VxVector &Dir, VxVector &Up) {
    if (m_SoundManager) {
        CKWaveSound3DSettings settings;
        m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_ORIENTATION, settings, FALSE);
        Dir = settings.m_OrientationDir;
        Up = settings.m_OrientationUp;
    }
}

CKERROR CKWaveSound::WriteData(CKBYTE *Buffer, int BufferSize) {
    if (!m_SoundManager || !m_Source)
        return CKERR_CANTWRITETOFILE;
    if (!Buffer || BufferSize <= 0)
        return CKERR_INVALIDPARAMETER;

    // -1 is used as an internal sentinel (e.g. before the first stream fill),
    // but locking/writing at cursor -1 is invalid. Start from 0 on first write.
    if (m_BufferPos == -1)
        m_BufferPos = 0;

    const int bufferSize = m_SoundManager->GetWaveSize(m_Source);
    const CKBOOL playing = m_SoundManager->IsPlaying(m_Source);
    const int playPos = playing ? (int) GetPlayPosition() : -1;

    const int writeEnd = m_BufferPos + BufferSize;
    const bool wrapsAround = (writeEnd >= bufferSize);

    if (playing) {
        if ((!wrapsAround && (playPos >= m_BufferPos) && (playPos <= writeEnd)) ||
            (wrapsAround && (playPos >= m_BufferPos || playPos < (writeEnd % bufferSize)))) {
            return CKERR_INVALIDSIZE;
        }
    }

    void *ptr1 = nullptr;
    void *ptr2 = nullptr;
    CKDWORD bytes1 = 0;
    CKDWORD bytes2 = 0;

    CKERROR err = m_SoundManager->Lock(
        m_Source,
        m_BufferPos,
        BufferSize,
        &ptr1, &bytes1,
        &ptr2, &bytes2,
        (CK_WAVESOUND_LOCKMODE) 0);
    if (err != CK_OK)
        return err;

    memcpy(ptr1, Buffer, bytes1);
    if (ptr2 && bytes2 > 0) {
        memcpy(ptr2, Buffer + bytes1, bytes2);
    }

    m_SoundManager->Unlock(m_Source, ptr1, bytes1, ptr2, bytes2);
    m_BufferPos = (m_BufferPos + BufferSize) % bufferSize;
    return CK_OK;
}

CKERROR
CKWaveSound::Lock(CKDWORD WriteCursor, CKDWORD NumBytes, void **Ptr1, CKDWORD *Bytes1, void **Ptr2, CKDWORD *Bytes2,
                  CK_WAVESOUND_LOCKMODE Flags) {
    if (!m_SoundManager)
        return CKERR_INVALIDPARAMETER;
    return m_SoundManager->Lock(m_Source, WriteCursor, NumBytes, Ptr1, Bytes1, Ptr2, Bytes2, Flags);
}

CKERROR CKWaveSound::Unlock(void *Ptr1, CKDWORD Bytes1, void *Ptr2, CKDWORD Bytes2) {
    if (!m_SoundManager)
        return CKERR_INVALIDPARAMETER;
    return m_SoundManager->Unlock(m_Source, Ptr1, Bytes1, Ptr2, Bytes2);
}

CKDWORD CKWaveSound::GetPlayPosition() {
    if (!m_SoundManager)
        return 0;
    return m_SoundManager->GetPlayPosition(m_Source);
}

int CKWaveSound::GetPlayedMs() {
    if (!m_SoundManager)
        return 0;

    int playedMs = 0;
    int bytesPerSec = (int) m_WaveFormat.nAvgBytesPerSec;

    if (GetFileStreaming() && !(m_State & CK_WAVESOUND_STREAMFULLYLOADED)) {
        playedMs = (int) ((m_DataPlayed * 1000.0) / bytesPerSec);
        if (playedMs < 0) {
            playedMs += m_Duration;
        }
    } else {
        playedMs = (int) ((m_DataPlayed * 1000.0) / bytesPerSec);
        GetPlayPosition();
    }

    return playedMs;
}

CKERROR CKWaveSound::Create(CKBOOL FileStreaming, CKSTRING Filename) {
    if (FileStreaming) {
        m_State |= CK_WAVESOUND_FILESTREAMED;
    } else {
        m_State &= ~CK_WAVESOUND_FILESTREAMED;
    }
    SetSoundFileName(Filename);
    SaveSettings();
    return Recreate();
}

CKERROR CKWaveSound::Create(CK_WAVESOUND_TYPE Type, CKWaveFormat *Format, int Size) {
    if (!Format || !m_SoundManager)
        return CKERR_INVALIDPARAMETER;
    Release();

    m_Source = m_SoundManager->CreateSource(Type, Format, Size, GetFileStreaming());
    if (!m_Source)
        return CKERR_INVALIDPARAMETER;

    m_BufferSize = Size;
    m_WaveFormat = *Format;
    return CK_OK;
}

CKERROR CKWaveSound::SetReader(CKSoundReader *Reader) {
    if (m_SoundReader) {
        Stop(0.0f);
        Rewind();
        m_SoundReader->Release();
    }
    m_State |= CK_WAVESOUND_FILESTREAMED;
    m_SoundReader = Reader;
    return CK_OK;
}

CKSoundReader *CKWaveSound::GetReader() {
    return m_SoundReader;
}

void CKWaveSound::SetDataToRead(int Size) {
    m_DataToRead = Size;
}

CKERROR CKWaveSound::Recreate(CKBOOL Safe) {
    SaveSettings();

    CKERROR err = TryRecreate();
    if (err != CK_OK && Safe) {
        m_Context->OutputToConsole("CKError : Failed to load the sound file, creating a default blank sound.");

        if (m_SoundReader) {
            m_SoundReader->Release();
            m_SoundReader = nullptr;
        }

        CKWaveFormat &wf = m_WaveFormat;
        wf.wFormatTag = 1;
        wf.nChannels = 1;
        wf.nSamplesPerSec = 22100;
        wf.nAvgBytesPerSec = 22100;
        wf.nBlockAlign = 1;
        wf.wBitsPerSample = 8;
        wf.cbSize = 0;

        m_State = (m_State & ~(CK_WAVESOUND_STREAMFULLYLOADED | CK_WAVESOUND_STREAMOVERLAP)) |
            CK_WAVESOUND_STREAMFULLYLOADED;

        CKDWORD bufferDuration = m_SoundManager->GetStreamedBufferSize();
        m_DataToRead = (int) ((double) bufferDuration * (double) m_WaveFormat.nAvgBytesPerSec * 0.001);
        m_BufferSize = m_DataToRead;

        // Create new audio source with fallback format
        m_Source = m_SoundManager->CreateSource(
            (CK_WAVESOUND_TYPE)(m_State & CK_WAVESOUND_ALLTYPE),
            &m_WaveFormat,
            m_BufferSize,
            FALSE);

        if (!m_Source)
            return CKERR_INVALIDPARAMETER;

        FillWithBlanks(FALSE);

        RestoreSettings();

        err = CK_OK;
    } else {
        m_State &= ~CK_WAVESOUND_NEEDREWIND;
    }

    return err;
}

void CKWaveSound::Release() {
    if (m_SoundManager) {
        if (m_SoundReader) {
            m_SoundReader->Release();
            m_SoundReader = nullptr;
        }
        m_SoundManager->ReleaseSource(m_Source);
        m_Source = nullptr;
    }
}

CKERROR CKWaveSound::TryRecreate() {
    m_DataToRead = 0;
    m_DataRead = 0;
    m_DataPlayed = 0;
    m_OldCursorPos = 0;
    m_Duration = 0;
    m_BufferSize = 0;

    Release();

    if (!m_FileName)
        return CKERR_INVALIDPARAMETER;

    m_BufferPos = -1;

    XString fileToOpen = m_FileName;
    CKERROR resolveErr = CKERR_NOTFOUND;
    if (m_Context && m_Context->GetPathManager())
        resolveErr = m_Context->GetPathManager()->ResolveFileName(fileToOpen, SOUND_PATH_IDX);

    CKPathSplitter pathSplitter(fileToOpen.Str());
    CKFileExtension ext(pathSplitter.GetExtension());

    CKPluginManager *pluginManager = CKGetPluginManager();
    m_SoundReader = pluginManager->GetSoundReader(ext);
    if (!m_SoundReader)
        return CKERR_INVALIDPARAMETER;

    if (m_SoundReader->OpenFile(fileToOpen.Str()) != CK_OK) {
        if (m_Context) {
            if (resolveErr == CK_OK) {
                m_Context->OutputToConsoleEx("CKError : Failed to open sound file '%s' (resolved '%s').", m_FileName, fileToOpen.Str());
            } else {
                m_Context->OutputToConsoleEx("CKError : Failed to open sound file '%s'.", m_FileName);
            }
        }
        m_SoundReader->Release();
        m_SoundReader = nullptr;
        return CKERR_INVALIDPARAMETER;
    }

    if (m_SoundReader->GetWaveFormat(&m_WaveFormat) != CK_OK) {
        m_SoundReader->Release();
        m_SoundReader = nullptr;
        return CKERR_INVALIDPARAMETER;
    }

    m_State &= ~(CK_WAVESOUND_STREAMOVERLAP | CK_WAVESOUND_STREAMFULLYLOADED);

    const int loadedDuration = m_SoundReader->GetDuration();
    m_DataToRead = m_SoundReader->GetDataSize();
    if (m_DataToRead <= 0) {
        if (loadedDuration <= 0) {
            m_SoundReader->Release();
            m_SoundReader = nullptr;
            return CKERR_INVALIDPARAMETER;
        }
        m_DataToRead = (int) ((double) m_WaveFormat.nAvgBytesPerSec * (double) loadedDuration * 0.001);
    }

    const int streamBufferSize = (int) ((double) m_WaveFormat.nAvgBytesPerSec *
        (double) (int) m_SoundManager->GetStreamedBufferSize() * 0.001);
    m_BufferSize = m_DataToRead;
    if (GetFileStreaming() && streamBufferSize < m_BufferSize)
        m_BufferSize = streamBufferSize;

    const CKBOOL streamingEnabled = (GetFileStreaming() && m_BufferSize < m_DataToRead) ? TRUE : FALSE;
    if (!streamingEnabled)
        m_State |= CK_WAVESOUND_STREAMFULLYLOADED;

    m_Source = m_SoundManager->CreateSource(
        (CK_WAVESOUND_TYPE) (m_State & CK_WAVESOUND_ALLTYPE),
        &m_WaveFormat,
        m_BufferSize,
        streamingEnabled);
    if (!m_Source) {
        m_SoundReader->Release();
        m_SoundReader = nullptr;
        return CKERR_INVALIDPARAMETER;
    }

    if (streamingEnabled) {
        m_OldCursorPos = 0;
        m_DataPlayed = 0;
        WriteDataFromReader();
    } else {
        m_SoundReader->Play();

        int totalWritten = 0;
        while (m_SoundReader && m_SoundReader->Decode() != CKSOUND_READER_EOF) {
            CKBYTE *dataBuffer = nullptr;
            int bufferSize = 0;
            if (m_SoundReader->GetDataBuffer(&dataBuffer, &bufferSize) == CK_OK) {
                CKERROR err = WriteData(dataBuffer, bufferSize);
                if (err != CK_OK) {
                    if (err == CKERR_INVALIDSIZE) {
                        void *newSource = ReallocSource(m_Source, totalWritten, totalWritten + bufferSize);
                        m_Source = newSource;
                        if (newSource && WriteData(dataBuffer, bufferSize) == CK_OK) {
                            totalWritten += bufferSize;
                            m_BufferPos = totalWritten;
                        }
                    } else {
                        m_SoundReader->Release();
                        m_SoundReader = nullptr;
                        Release();
                    }
                } else {
                    totalWritten += bufferSize;
                }
            }
        }

        if (m_Source && totalWritten < m_BufferSize)
            m_Source = ReallocSource(m_Source, totalWritten, totalWritten);

        if (m_SoundReader) {
            m_SoundReader->Release();
            m_SoundReader = nullptr;
        }
    }

    RestoreSettings();

    if (loadedDuration > 0)
        m_Duration = loadedDuration;

    return CK_OK;
}

void CKWaveSound::UpdatePosition(float deltaT) {
    CK3dEntity *entity = GetAttachedEntity();
    PositionSource(m_SoundManager, m_Source, entity, m_Position, m_Direction, m_OldPosition, deltaT);
}

void CKWaveSound::UpdateFade() {
    if (!m_SoundManager)
        return;

    if (m_State & CK_WAVESOUND_FADE) {
        const float deltaTime = m_Context->GetTimeManager()->GetLastDeltaTime();
        m_CurrentTime = XMin(m_CurrentTime + deltaTime, m_FadeTime);

        float gain = 0.0f;
        if (m_State & CK_WAVESOUND_FADEIN) {
            gain = (m_CurrentTime / m_FadeTime) * m_FinalGain;
        } else {
            gain = (1.0f - (m_CurrentTime / m_FadeTime)) * m_FinalGain;
        }
        InternalSetGain(gain);

        if (m_CurrentTime == m_FadeTime) {
            if (m_State & CK_WAVESOUND_FADEOUT) {
                InternalStop();
            }
            m_State &= ~CK_WAVESOUND_FADE;
        }
    }
}

CKERROR CKWaveSound::WriteDataFromReader() {
    if (!m_Source || !m_SoundReader)
        return CKERR_CANTWRITETOFILE;

    const int currentPos = (int) GetPlayPosition();
    const int bufferSize = m_SoundManager->GetWaveSize(m_Source);

    int deltaBytes = 0;
    if ((unsigned int)currentPos < (unsigned int)m_OldCursorPos) {
        deltaBytes = bufferSize - m_OldCursorPos + currentPos;
    } else {
        deltaBytes = currentPos - m_OldCursorPos;
    }
    m_DataPlayed += deltaBytes;
    if (m_DataToRead < m_DataPlayed) {
        if (IsPlaying())
            InternalStop();
        return CK_OK;
    }

    m_OldCursorPos = currentPos;

    CKBYTE *dataBuffer = nullptr;
    int dataSize = 0;

    // If we previously failed to write (overlap), CK2.dll attempts a single flush of the existing buffer and returns.
    if (m_State & CK_WAVESOUND_STREAMOVERLAP) {
        m_SoundReader->GetDataBuffer(&dataBuffer, &dataSize);
        if (dataSize != 0 && WriteData(dataBuffer, dataSize) == CK_OK) {
            m_DataRead += dataSize;
            m_State &= ~CK_WAVESOUND_STREAMOVERLAP;
        }
        return CK_OK;
    }

    CKBOOL done = FALSE;
    while (!(m_State & CK_WAVESOUND_STREAMOVERLAP) && !done) {
        CKERROR err = m_SoundReader->Decode();
        if (err == CK_OK) {
            m_SoundReader->GetDataBuffer(&dataBuffer, &dataSize);
            if (dataSize != 0 && WriteData(dataBuffer, dataSize) == CK_OK) {
                m_DataRead += dataSize;
            } else {
                m_State |= CK_WAVESOUND_STREAMOVERLAP;
            }
        } else {
            if (err == CKSOUND_READER_NO_DATA_READY) {
                FillWithBlanks(FALSE);
                if (m_BufferPos == -1) {
                    m_BufferPos = m_BufferSize / 4;
                } else {
                    int distFromCursor = GetDistanceFromCursor();
                    int threshold = m_BufferSize / 10;
                    if (distFromCursor < threshold) {
                        int fillSize = m_BufferSize / 4;
                        int advance = (fillSize > deltaBytes) ? fillSize : deltaBytes;
                        m_BufferPos = (m_BufferPos + advance) % m_BufferSize;
                        int subtract = (fillSize > deltaBytes) ? fillSize : deltaBytes;
                        m_DataPlayed -= subtract;
                    }
                }
            } else if (err == CKSOUND_READER_EOF) {
                if (GetLoopMode()) {
                    int dataRead = m_DataRead;
                    m_DataRead = 0;
                    m_DataPlayed -= dataRead;
                    m_SoundReader->Seek(0);
                    m_SoundReader->Play();
                } else {
                    done = TRUE;
                    FillWithBlanks(TRUE);
                }
            }
            done = TRUE;
        }
    }

    return CK_OK;
}

void CKWaveSound::FillWithBlanks(CKBOOL IncBf) {
    if (!m_SoundManager || !m_Source)
        return;

    void *ptr1 = nullptr;
    void *ptr2 = nullptr;
    CKDWORD bytes1 = 0;
    CKDWORD bytes2 = 0;

    const int playPos = (int) GetPlayPosition();
    int fillStart = m_BufferPos;
    int fillSize;

    if (fillStart == -1) {
        fillStart = 0;
        fillSize = m_BufferSize;
    } else {
        fillSize = (fillStart >= playPos)
                       ? playPos + m_BufferSize - fillStart
                       : playPos - fillStart;
    }

    CKERROR err = m_SoundManager->Lock(
        m_Source,
        fillStart,
        fillSize,
        &ptr1, &bytes1,
        &ptr2, &bytes2,
        (CK_WAVESOUND_LOCKMODE) 0
    );
    if (err != CK_OK)
        return;

    const int silence = (m_WaveFormat.wBitsPerSample != 16) ? 0x80 : 0x00;
    memset(ptr1, silence, bytes1);
    if (ptr2)
        memset(ptr2, silence, bytes2);

    m_SoundManager->Unlock(m_Source, ptr1, bytes1, ptr2, bytes2);

    if (IncBf) {
        m_BufferPos = (fillStart + fillSize) % m_BufferSize;
    }
}

void CKWaveSound::InternalStop() {
    if (m_SoundManager) {
        m_SoundManager->Stop(this, m_Source);
        if (m_SoundReader) {
            m_SoundReader->Stop();
        }
        m_State |= CK_WAVESOUND_NEEDREWIND;
    }
}

CKWaveSound::CKWaveSound(CKContext *Context, CKSTRING Name): CKSound(Context, Name) {
    m_Source = nullptr;
    m_FinalGain = 1.0f;
    m_FadeTime = 0.0f;
    m_CurrentTime = 0.0f;
    
    m_2DSetting.m_Priority = 0.5f;
    m_2DSetting.m_Gain = 1.0f;
    m_2DSetting.m_Eq = 1.0f;
    m_2DSetting.m_Pitch = 1.0f;
    m_2DSetting.m_Pan = 0.0f;
    
    // m_3DSetting constructor is called separately
    
    m_State = 1;
    m_Position = VxVector(0.0f, 0.0f, 0.0f);
    m_OldPosition = VxVector(0.0f, 0.0f, 0.0f);
    m_Direction = VxVector(0.0f, 0.0f, 1.0f);
    m_BufferPos = -1;
    m_SoundReader = nullptr;
    m_DataRead = 0;
    m_DataPlayed = 0;
    m_OldCursorPos = 0;
    m_DataToRead = 0;
    m_BufferSize = 0;
    m_AttachedObject = 0;
    m_Duration = 0;
    memset(&m_WaveFormat, 0, sizeof(m_WaveFormat));
    m_SoundManager = (CKSoundManager *) m_Context->GetManagerByGuid(SOUND_MANAGER_GUID);
}

CKWaveSound::~CKWaveSound() {
    Release();
}

CK_CLASSID CKWaveSound::GetClassID() {
    return m_ClassID;
}

CKStateChunk *CKWaveSound::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKSound::Save(file, flags);

    if (!file && !(flags & CK_STATESAVE_WAVSOUNDONLY))
        return baseChunk;

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_WAVESOUND, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    if (file || (flags & CK_STATESAVE_WAVSOUNDDURATION)) {
        chunk->WriteIdentifier(CK_STATESAVE_WAVSOUNDDURATION);
        chunk->WriteInt(m_Duration);
    }

    if (flags & CK_STATESAVE_WAVSOUNDDATA2) {
        chunk->WriteIdentifier(CK_STATESAVE_WAVSOUNDDATA2);

        chunk->WriteDword(GetState());
        chunk->WriteFloat(GetPriority());
        chunk->WriteFloat(m_FinalGain);

        chunk->WriteFloat(GetPan());
        chunk->WriteFloat(GetPitch());

        chunk->WriteFloat(0.0f); // Reserved
        chunk->WriteFloat(0.0f); // Reserved
        chunk->WriteFloat(0.0f); // Reserved

        float inAngle, outAngle, outGain;
        GetCone(&inAngle, &outAngle, &outGain);
        chunk->WriteFloat(inAngle);
        chunk->WriteFloat(outAngle);
        chunk->WriteFloat(outGain);

        float minDist, maxDist;
        CKDWORD distBehavior;
        GetMinMaxDistance(&minDist, &maxDist, &distBehavior);
        chunk->WriteFloat(minDist);
        chunk->WriteFloat(maxDist);
        chunk->WriteDword(distBehavior);

        CKObject *attachedObj = m_Context->GetObject(m_AttachedObject);
        chunk->WriteObject(attachedObj);

        chunk->WriteBuffer(sizeof(VxVector), &m_Position);
        chunk->WriteBuffer(sizeof(VxVector), &m_Direction);

        chunk->WriteInt(0); // Reserved
    }

    if (GetClassID() == CKCID_WAVESOUND) {
        chunk->CloseChunk();
    } else {
        chunk->UpdateDataSize();
    }
    return chunk;
}

CKERROR CKWaveSound::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) return CKERR_INVALIDPARAMETER;

    CKSound::Load(chunk, file);

    char *fileName = nullptr;
    if (chunk->SeekIdentifier(CK_STATESAVE_WAVSOUNDFILE)) {
        if (file) {
            chunk->ReadString(&fileName);
        } else {
            char *tempName = nullptr;
            chunk->ReadString(&tempName);
            XString resolvedPath = tempName;
            m_Context->GetPathManager()->ResolveFileName(resolvedPath, SOUND_PATH_IDX);
            fileName = CKStrdup(resolvedPath.Str());
            delete[] tempName;
        }
    }

    if (chunk->SeekIdentifier(CK_STATESAVE_WAVSOUNDDURATION)) {
        m_Duration = chunk->ReadInt();
    }

    if (!chunk->SeekIdentifier(CK_STATESAVE_WAVSOUNDDATA2)) {
        SetSoundFileName(fileName);
        delete[] fileName;
        Recreate(FALSE);
        return CK_OK;
    }

    int dataVersion = chunk->GetDataVersion();
    if (dataVersion >= 3) {
        m_State = chunk->ReadInt();
        SetSoundFileName(fileName);
        delete[] fileName;

        Recreate(TRUE); // Safe recreation

        SetPriority(chunk->ReadFloat());
        SetGain(chunk->ReadFloat());
        SetPan(chunk->ReadFloat());
        SetPitch(chunk->ReadFloat());

        chunk->ReadFloat(); // Reserved
        chunk->ReadFloat(); // Reserved
        chunk->ReadFloat(); // Reserved

        float inAngle = chunk->ReadFloat();
        float outAngle = chunk->ReadFloat();
        float outGain = chunk->ReadFloat();
        if (dataVersion >= 10)
            SetCone(inAngle, outAngle, outGain);

        float minDist = chunk->ReadFloat();
        float maxDist = chunk->ReadFloat();
        CKDWORD distBehavior = chunk->ReadInt();
        SetMinMaxDistance(minDist, maxDist, distBehavior);

        m_AttachedObject = chunk->ReadObjectID();
        chunk->ReadAndFillBuffer(sizeof(VxVector), &m_Position);
        chunk->ReadAndFillBuffer(sizeof(VxVector), &m_Direction);
        chunk->ReadInt(); // Reserved
    } else if (dataVersion >= 2) {
        m_State = chunk->ReadInt();
        SetSoundFileName(fileName);
        delete[] fileName;

        Recreate(FALSE);

        SetPriority(chunk->ReadFloat());
        chunk->ReadInt(); // Reserved
        chunk->ReadInt(); // Reserved
        SetGain(chunk->ReadFloat());
        SetPan(chunk->ReadFloat());
        SetPitch(chunk->ReadFloat());
        chunk->ReadFloat(); // Reserved

        if (GetType() != CK_WAVESOUND_BACKGROUND) {
            chunk->ReadFloat(); // Reserved
            chunk->ReadFloat(); // Reserved
            float inAngle = chunk->ReadFloat();
            float outAngle = chunk->ReadFloat();
            float outGain = chunk->ReadFloat();
            SetCone(inAngle, outAngle, outGain);

            float minDist = chunk->ReadFloat();
            float maxDist = chunk->ReadFloat();
            CKDWORD distBehavior = chunk->ReadDword();
            SetMinMaxDistance(minDist, maxDist, distBehavior);

            m_AttachedObject = chunk->ReadObjectID();
            chunk->ReadAndFillBuffer(sizeof(VxVector), &m_Position);
            chunk->ReadAndFillBuffer(sizeof(VxVector), &m_Direction);
            chunk->ReadInt(); // Reserved
        }
    } else {
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        SetLoopMode(chunk->ReadInt());
        chunk->ReadFloat();
        chunk->ReadFloat();
        chunk->ReadFloat();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadInt();
        chunk->ReadFloat();
        chunk->ReadFloat();
    }

    return CK_OK;
}

CKERROR CKWaveSound::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKBeObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    m_AttachedObject = context.RemapID(m_AttachedObject);
    return CK_OK;
}

CKERROR CKWaveSound::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKBeObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    CKWaveSound *ws = (CKWaveSound *) &o;
    delete[] m_FileName;
    m_FileName = CKStrdup(ws->m_FileName);
    m_SaveOptions = ws->m_SaveOptions;
    m_SoundManager = ws->m_SoundManager;
    m_AttachedObject = ws->m_AttachedObject;
    m_OldPosition = ws->m_OldPosition;
    m_Position = ws->m_Position;
    m_Direction = ws->m_Direction;
    memcpy(&m_WaveFormat, &ws->m_WaveFormat, sizeof(m_WaveFormat));
    Recreate(TRUE);
    return CK_OK;
}

void CKWaveSound::CheckPostDeletion() {
    CKObject::CheckPostDeletion();
    if (GetAttachedEntity())
        m_AttachedObject = 0;
}

int CKWaveSound::GetMemoryOccupation() {
    return CKSound::GetMemoryOccupation() + (int) (sizeof(CKWaveSound) - sizeof(CKSound));
}

CKSTRING CKWaveSound::GetClassName() {
    return "Wave Sound";
}

int CKWaveSound::GetDependenciesCount(int Mode) {
    return CKObject::GetDependenciesCount(Mode);
}

CKSTRING CKWaveSound::GetDependencies(int i, int Mode) {
    return CKObject::GetDependencies(i, Mode);
}

void CKWaveSound::Register() {
    CKCLASSNOTIFYFROM(CKWaveSound, CKBeObject);
    CKPARAMETERFROMCLASS(CKWaveSound, CKPGUID_WAVESOUND);
}

CKWaveSound *CKWaveSound::CreateInstance(CKContext *Context) {
    return new CKWaveSound(Context);
}

int CKWaveSound::GetDistanceFromCursor() {
    if (!m_Source)
        return 0;
    int distance = m_BufferPos - (int) GetPlayPosition();
    if (distance < 0)
        distance += m_BufferSize;
    return distance;
}

void CKWaveSound::InternalSetGain(float Gain) {
    if (m_SoundManager) {
        CKWaveSoundSettings settings;
        settings.m_Eq = 1.0f;
        settings.m_Pitch = 1.0f;
        settings.m_Pan = 0.0f;
        settings.m_Priority = 0.5f;
        settings.m_Gain = Gain;
        m_SoundManager->UpdateSettings(m_Source, CK_WAVESOUND_SETTINGS_GAIN, settings, TRUE);
    }
}

void CKWaveSound::SaveSettings() {
    if (m_Source) {
        if (GetType() == CK_WAVESOUND_BACKGROUND) {
            m_SoundManager->UpdateSettings(
                m_Source,
                CK_WAVESOUND_SETTINGS_PAN,
                m_2DSetting,
                FALSE);
        } else if (GetType() == CK_WAVESOUND_POINT) {
            m_SoundManager->UpdateSettings(
                m_Source,
                (CK_SOUNDMANAGER_CAPS) (CK_WAVESOUND_SETTINGS_PRIORITY |
                    CK_WAVESOUND_SETTINGS_PITCH |
                    CK_WAVESOUND_SETTINGS_EQUALIZATION |
                    CK_WAVESOUND_SETTINGS_GAIN),
                m_2DSetting,
                FALSE);
            m_SoundManager->Update3DSettings(
                m_Source,
                CK_WAVESOUND_3DSETTINGS_ALL,
                m_3DSetting,
                FALSE);
        }
    } else {
        m_2DSetting.m_Eq = 1.0f;
        m_2DSetting.m_Gain = 1.0f;
        m_2DSetting.m_Pan = 0.0f;
        m_2DSetting.m_Pitch = 1.0f;
        m_2DSetting.m_Priority = 1.0f;

        memset(&m_3DSetting, 0, sizeof(m_3DSetting));
        m_3DSetting.m_HeadRelative = 0;
        m_3DSetting.m_InAngle = 360.0f;
        m_3DSetting.m_OutAngle = 360.0f;
        m_3DSetting.m_MaxDistance = 2.0f;
        m_3DSetting.m_MuteAfterMax = 0;
        m_3DSetting.m_MinDistance = 1.0f;
    }
}

void CKWaveSound::RestoreSettings() {
    m_SoundManager->UpdateSettings(m_Source, CK_WAVESOUND_SETTINGS_ALL, m_2DSetting, TRUE);
    m_SoundManager->Update3DSettings(m_Source, CK_WAVESOUND_3DSETTINGS_ALL, m_3DSetting, TRUE);
}

void *CKWaveSound::ReallocSource(void *oSource, int alreadyWritten, int newSize) {
    void *newSource = m_SoundManager->CreateSource((CK_WAVESOUND_TYPE) (m_State & CK_WAVESOUND_ALLTYPE), &m_WaveFormat, newSize, FALSE);
    if (!newSource)
        return oSource;

    void *srcBuffer1 = nullptr;
    void *srcBuffer2 = nullptr;
    CKDWORD srcSize1 = 0, srcSize2 = 0;

    if (m_SoundManager->Lock(oSource, 0, alreadyWritten, &srcBuffer1, &srcSize1, &srcBuffer2, &srcSize2, (CK_WAVESOUND_LOCKMODE) 0)) {
        m_SoundManager->ReleaseSource(newSource);
        return nullptr;
    }

    void *dstBuffer1 = nullptr;
    void *dstBuffer2 = nullptr;
    CKDWORD dstSize1 = 0, dstSize2 = 0;

    m_SoundManager->Lock(newSource, 0, alreadyWritten, &dstBuffer1, &dstSize1, &dstBuffer2, &dstSize2, CK_WAVESOUND_LOCKMODE(0));

    if (newSize) {
        if (dstSize1)
            memcpy(dstBuffer1, srcBuffer1, newSize);
    }

    m_SoundManager->Unlock(oSource, srcBuffer1, newSize, srcBuffer2, srcSize2);
    m_SoundManager->Unlock(newSource, dstBuffer1, dstSize1, dstBuffer2, dstSize2);

    m_SoundManager->ReleaseSource(oSource);
    return newSource;
}
