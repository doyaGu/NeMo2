#include "CKTimeManager.h"
#include "CKContext.h"
#include "CKStateChunk.h"
#include "CKFile.h"

void CKTimeManager::ChangeLimitOptions(CK_FRAMERATE_LIMITS FpsOptions, CK_FRAMERATE_LIMITS BehOptions) {
    if (FpsOptions != CK_RATE_NOP)
    {
        m_LimitOptions |= FpsOptions & CK_FRAMERATE_MASK;
    }

    if (BehOptions != CK_RATE_NOP)
    {
        m_LimitOptions |= BehOptions & CK_BEHRATE_MASK;
    }
}

void CKTimeManager::SetFrameRateLimit(float FRLimit) {
    if (FRLimit > 0.0f) {
        m_LimitFrameRate = FRLimit;
        m_LimitFPSDeltaTime = 1000.0f / FRLimit;
    }
}

void CKTimeManager::SetBehavioralRateLimit(float BRLimit) {
    if (BRLimit > 0.0f) {
        m_LimitBehRate = BRLimit;
        m_LimitBehDeltaTime = 1000.0f / BRLimit;
    }
}

void CKTimeManager::SetMinimumDeltaTime(float DtMin) {
    if ( DtMin > 0.0f )
        m_MinimumDeltaTime = DtMin;
}

void CKTimeManager::SetMaximumDeltaTime(float DtMax) {
    if ( DtMax > 0.0f )
        m_MaximumDeltaTime = DtMax;
}

void CKTimeManager::GetTimeToWaitForLimits(float &TimeBeforeRender, float &TimeBeforeBeh) {
    if (m_LimitOptions & (CK_FRAMERATE_SYNC | CK_FRAMERATE_FREE))
        TimeBeforeRender = 0.0f;
    else
        TimeBeforeRender = m_LimitFPSDeltaTime - m_FpsChrono.Current();

    if (m_LimitOptions & CK_BEHRATE_SYNC)
        TimeBeforeBeh = TimeBeforeRender;
    else
        TimeBeforeBeh = m_LimitBehDeltaTime - m_BehChrono.Current();
}

void CKTimeManager::ResetChronos(CKBOOL RenderChrono, CKBOOL BehavioralChrono) {
    if (RenderChrono)
        m_FpsChrono.Reset();
    if (BehavioralChrono)
        m_BehChrono.Reset();
}

CKERROR CKTimeManager::PreProcess() {
    CKStats &stats = m_Context->m_Stats;
    m_Context->m_Stats.TotalFrameTime = m_CurrentChrono.Current();
    m_CurrentChrono.Reset();
    stats.EstimatedInterfaceTime = stats.TotalFrameTime - (stats.RenderTime + stats.ProcessTime);
    if (stats.EstimatedInterfaceTime < 0.0f)
        stats.EstimatedInterfaceTime = 0.0f;

    m_Context->m_ProfileStats = stats;

    stats.ProcessTime = 0.0f;
    stats.RenderTime = 0.0f;
    stats.ParametricOperations = 0.0f;
    stats.AnimationManagement = 0.0f;
    stats.BehaviorCodeExecution = 0.0f;
    stats.TotalBehaviorExecution = 0.0f;
    stats.BehaviorsExecuted = 0;
    stats.BuildingBlockExecuted = 0;
    stats.BehaviorLinksParsed = 0;
    stats.ActiveObjectsExecuted = 0;
    stats.BehaviorDelayedLinks = 0;
    memset(stats.UserProfiles, 0, sizeof(stats.UserProfiles));
    memset(m_Context->m_UserProfileTime, 0, sizeof(m_Context->m_UserProfileTime));

    if (m_Paused)
        return CKERR_PAUSED;

    ++m_MainTickCount;

    m_LastTime = m_CurrentTime;
    m_CurrentTime = m_StartupChrono.Current();

    m_DeltaTime = m_CurrentTime - m_LastTime;
    m_DeltaTimeFree = m_DeltaTime;

    if (m_DeltaTime < m_MinimumDeltaTime)
        m_DeltaTime = m_MinimumDeltaTime;
    if (m_DeltaTime > m_MaximumDeltaTime)
        m_DeltaTime = m_MaximumDeltaTime;

    m_DeltaTime *= m_TimeScaleFactor;
    m_PlayTime += m_DeltaTime;

    return CK_OK;
}

CKERROR CKTimeManager::OnCKPlay() {
    float current = m_StartupChrono.Current();
    m_LastTime = current;
    m_CurrentTime = current;
    m_Paused = FALSE;
    return CK_OK;
}

CKERROR CKTimeManager::OnCKPause() {
    m_Paused = TRUE;
    m_CurrentTime = m_StartupChrono.Current();
    return CK_OK;
}

CKERROR CKTimeManager::OnCKReset() {
    float current = m_StartupChrono.Current();
    m_LastTime = current;
    m_CurrentTime = current;
    m_Paused = TRUE;
    m_PlayTime = 0.0f;
    return CK_OK;
}

CKERROR CKTimeManager::PostClearAll() {
    return CK_OK;
}

CKBOOL CKTimeManager::IsPaused() {
    return m_Paused;
}

CKERROR CKTimeManager::LoadData(CKStateChunk *chunk, CKFile *LoadedFile) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    chunk->StartRead();
    int size = chunk->SeekIdentifierAndReturnSize(0x80);
    if (size > 0)
    {
        SetFrameRateLimit(chunk->ReadFloat());
        SetBehavioralRateLimit(chunk->ReadFloat());
        float savedRate = chunk->ReadFloat();
        if (savedRate > 0.0f)
            SetMaximumDeltaTime(1000.0f / savedRate);
        m_TimeScaleFactor = chunk->ReadFloat();
        m_LimitOptions = chunk->ReadDword();
        if (size > 20)
        {
            SetMinimumDeltaTime(chunk->ReadFloat());
        }
    }

    return CK_OK;
}

CKStateChunk *CKTimeManager::SaveData(CKFile *SavedFile) {
    if (SavedFile->m_IndexByClassId[CKCID_LEVEL].Size() == 0)
        return nullptr;

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_OBJECT, SavedFile);
    if (chunk)
    {
        chunk->StartWrite();
        chunk->WriteIdentifier(0x80);
        chunk->WriteFloat(m_LimitFrameRate);
        chunk->WriteFloat(m_LimitBehRate);
        chunk->WriteFloat(1000.0f / m_MaximumDeltaTime);
        chunk->WriteFloat(m_TimeScaleFactor);
        chunk->WriteDword(m_LimitOptions);
        chunk->WriteFloat(m_MinimumDeltaTime);
        chunk->CloseChunk();
    }

    return chunk;
}

CKTimeManager::CKTimeManager(CKContext *Context) : CKBaseManager(Context, TIME_MANAGER_GUID, "Time Manager") {
    m_BehChrono.Reset();
    m_FpsChrono.Reset();
    m_CurrentChrono.Reset();
    m_StartupChrono.Reset();
    m_MainTickCount = 0;

    float current = m_StartupChrono.Current();
    m_LastTime = current;
    m_CurrentTime = current;
    m_TimeScaleFactor = 1.0f;
    m_Paused = TRUE;
    m_LimitOptions = CK_FRAMERATE_SYNC | CK_BEHRATE_SYNC;
    m_PlayTime = 0.0f;
    SetFrameRateLimit(60.0f);
    SetBehavioralRateLimit(60.0f);
    m_MaximumDeltaTime = 200.0f;
    m_MinimumDeltaTime = 1.0f;
    m_Context->RegisterNewManager(this);
}
