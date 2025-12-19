#include "CKSoundManager.h"

#include "CKTimeManager.h"
#include "CKRenderContext.h"
#include "CK3dEntity.h"

void CKSoundManager::SetListener(CK3dEntity *listener) {
    if (listener) {
        if (listener->GetID() != m_ListenerEntity) {
            listener->GetPosition(&m_OldListenerPos);
        }
        m_ListenerEntity = listener->GetID();
    } else {
        m_ListenerEntity = 0;
    }
}

CK3dEntity *CKSoundManager::GetListener() {
    CK3dEntity *listener = (CK3dEntity *)m_Context->GetObject(m_ListenerEntity);
    if (!listener) {
        CKRenderContext *dev = m_Context->GetPlayerRenderContext();
        if (dev) {
            listener = dev->GetViewpoint();
        }
    }
    return listener;
}

void CKSoundManager::SetStreamedBufferSize(CKDWORD bsize) {
    m_BufferSize = bsize;
}

CKDWORD CKSoundManager::GetStreamedBufferSize() {
    return m_BufferSize;
}

SoundMinion *CKSoundManager::CreateMinion(CKSOUNDHANDLE source, float minimumDelay) {
    float currentTime = m_Context->m_TimeManager->GetAbsoluteTime();

    if (minimumDelay > 0.0f) {
        for (auto it = m_Minions.Begin(); it != m_Minions.End(); ++it) {
            SoundMinion *minion = *it;
            if (minion && minion->m_OriginalSource == source &&
                (currentTime - minion->m_TimeStamp) < minimumDelay) {
                return nullptr;
            }
        }
    }

    SoundMinion *minion = new SoundMinion();
    if (!minion)
        return nullptr;

    memset(minion, 0, sizeof(SoundMinion));

    minion->m_OriginalSource = source;
    minion->m_TimeStamp = currentTime;

    minion->m_Source = DuplicateSource(source);
    if (!minion->m_Source) {
        delete minion;
        return nullptr;
    }
    m_Minions.PushBack(minion);
    return minion;
}

void CKSoundManager::ReleaseMinions() {
    for (auto it = m_Minions.Begin(); it != m_Minions.End(); ++it) {
        SoundMinion *minion = *it;
        Stop(nullptr, minion->m_Source);
        ReleaseSource(minion->m_Source);
        delete minion;
    }
    m_Minions.Clear();
}

void CKSoundManager::PauseMinions() {
    for (auto it = m_Minions.Begin(); it != m_Minions.End(); ++it) {
        SoundMinion *minion = *it;
        InternalPause(minion->m_Source);
    }
}

void CKSoundManager::ResumeMinions() {
    for (auto it = m_Minions.Begin(); it != m_Minions.End(); ++it) {
        SoundMinion *minion = *it;
        InternalPlay(minion->m_Source, FALSE);
    }
}

void CKSoundManager::ProcessMinions() {
    for (auto it = m_Minions.Begin(); it != m_Minions.End();) {
        SoundMinion *minion = *it;
        if (IsPlaying(minion->m_Source)) {
            ++it;
        } else {
            ReleaseSource(minion->m_Source);
            delete minion;
            it = m_Minions.Remove(it);
        }
    }
}

CKSoundManager::CKSoundManager(CKContext *Context, CKSTRING smname) : CKBaseManager(
    Context, SOUND_MANAGER_GUID, smname) {
    m_BufferSize = 2000;
    m_ListenerEntity = 0;
    m_OldListenerPos = VxVector::axis0();
    RegisterAttribute();
}

CKSoundManager::~CKSoundManager() {
    m_Minions.Clear();
}

void CKSoundManager::RegisterAttribute() {}

CKERROR CKSoundManager::SequenceDeleted(CK_ID *objids, int count) {
    return PostClearAll();
}

CKERROR CKSoundManager::PostClearAll() {
    CK3dEntity *listener = (CK3dEntity *)m_Context->GetObject(m_ListenerEntity);
    if (!listener) {
        m_ListenerEntity = 0;
    }

    return CK_OK;
}
