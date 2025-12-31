#ifndef CKWAVESOUND_H
#define CKWAVESOUND_H

#include "CKSound.h"
#include "CKSoundManager.h"

class CKSoundReader;

/**************************************************************************
{filename:CKWaveSound}
Name: CKWaveSound

Summary: Wave sound.

Remarks:
+ CKWaveSound provides methods for playing a wave sound file (MP3,Wav,etc..)

+ The sound can be played using 3D, stereo, frequency control,
panning and volume control.
+ Sound file are not loaded but read from disk by a separate thread.

+ Its class id is CKCID_WAVESOUND
See also: CKMidiSound,CKSoundManager
******************************************************************************/
class CKWaveSound : public CKSound
{
    friend class CKSoundManager;

public:
    //-----------------------------------------------------
    // Sound Duplication for Instance Playing
    DLL_EXPORT CKSOUNDHANDLE PlayMinion(CKBOOL Background = TRUE, CK3dEntity *Ent = NULL, VxVector *Position = NULL, VxVector *Direction = NULL, float MinDelay = 0.0f);

    //---------------------------------------------------------
    // Associated filename
    DLL_EXPORT CKERROR SetSoundFileName(const CKSTRING FileName);
    DLL_EXPORT CKSTRING GetSoundFileName();

    //----------------------------------------------------------
    // Parameter PCM
    DLL_EXPORT int GetSoundLength();
    DLL_EXPORT CKERROR GetSoundFormat(CKWaveFormat &Format);
    //----------------------------------------------------------
    // Sound Type
    DLL_EXPORT CK_WAVESOUND_TYPE GetType();
    DLL_EXPORT void SetType(CK_WAVESOUND_TYPE Type);
    //----------------------------------------------------------
    // Sound State
    DLL_EXPORT CKDWORD GetState();
    DLL_EXPORT void SetState(CKDWORD State);

    //----------------------------------------------------------
    // Priority of the sound : between 0.0(lowest) and 1.0(highest). Default is 0.5
    DLL_EXPORT void SetPriority(float Priority);
    DLL_EXPORT float GetPriority();

    //----------------------------------------------------------
    // Activation of the loop mode : TOSETUP
    DLL_EXPORT void SetLoopMode(CKBOOL Enabled);
    DLL_EXPORT CKBOOL GetLoopMode();

    //----------------------------------------------------------
    // File Streaming
    DLL_EXPORT CKERROR SetFileStreaming(CKBOOL Enabled, CKBOOL RecreateSound = FALSE);
    DLL_EXPORT CKBOOL GetFileStreaming();

    //-----------------------------------------------------
    // PlayBack Control
    // plays the sound with a faded
    DLL_EXPORT void Play(float FadeIn = 0, float FinalGain = 1.0f);
    DLL_EXPORT void Resume();
    DLL_EXPORT void Rewind();

    //----------------------------------------------------------
    // Stops the sound with a fade
    DLL_EXPORT void Stop(float FadeOut = 0);

    DLL_EXPORT void Pause();

    //----------------------------------------------------------
    DLL_EXPORT CKBOOL IsPlaying();
    DLL_EXPORT CKBOOL IsPaused();

    //------------------------------------------------
    // 2D/3D Members Functions

    // Sets and gets the playback gain of a source. 0.....1.0
    DLL_EXPORT void SetGain(float Gain);
    DLL_EXPORT float GetGain();
    //----------------------------------------------------------
    // Sets and gets the playback pitch bend of a source. 0.5....2.0
    DLL_EXPORT void SetPitch(float Rate);
    DLL_EXPORT float GetPitch();
    //----------------------------------------------------------
    // Sets the gains for multichannel, non-specialized sources. -1.0....1.0 default(0.0)
    DLL_EXPORT void SetPan(float Pan);
    DLL_EXPORT float GetPan();

    DLL_EXPORT CKSOUNDHANDLE GetSource();
    //----------------------------------------------------------
    // 3D Members Functions

    //----------------------------------------------------------
    // Attach the sound to an object : TOSETUP
    DLL_EXPORT void PositionSound(CK3dEntity *Object, VxVector *Position = NULL, VxVector *Direction = NULL, CKBOOL Commit = FALSE);
    DLL_EXPORT CK3dEntity *GetAttachedEntity();
    DLL_EXPORT void GetPosition(VxVector &Pos);
    DLL_EXPORT void GetDirection(VxVector &Dir);
    DLL_EXPORT void GetSound3DInformation(VxVector &Pos, VxVector &Dir, float &DistanceFromListener);

    //----------------------------------------------------------
    // Sets the directionality of a source cone. 0.....180 for the angles, gain 0....1.0 : TOSETUP
    DLL_EXPORT void SetCone(float InAngle, float OutAngle, float OutsideGain);
    DLL_EXPORT void GetCone(float *InAngle, float *OutAngle, float *OutsideGain);
    //----------------------------------------------------------
    // Distance min/maximum de perception (maxdbehavior = 1(mute) ou 0(audible)), distances form 0.....n with max>min : TOSETUP
    DLL_EXPORT void SetMinMaxDistance(float MinDistance, float MaxDistance, CKDWORD MaxDistanceBehavior = 1);
    DLL_EXPORT void GetMinMaxDistance(float *MinDistance, float *MaxDistance, CKDWORD *MaxDistanceBehavior);

    // These functions are for direct access to the source : You don't have to use them normally.
    //----------------------------------------------------------
    // Velocity of the Source
    DLL_EXPORT void SetVelocity(VxVector &Pos);
    DLL_EXPORT void GetVelocity(VxVector &Pos);

    //----------------------------------------------------------
    // Orientation of the source
    DLL_EXPORT void SetOrientation(VxVector &Dir, VxVector &Up);
    DLL_EXPORT void GetOrientation(VxVector &Dir, VxVector &Up);

    //----------------------------------------------------------
    // Write Data in the sound buffer
    DLL_EXPORT CKERROR WriteData(CKBYTE *Buffer, int Buffersize);

    //----------------------------------------------------------
    // Buffer access
    DLL_EXPORT CKERROR Lock(CKDWORD WriteCursor, CKDWORD NumBytes, void **Ptr1, CKDWORD *Bytes1, void **Ptr2, CKDWORD *Bytes2, CK_WAVESOUND_LOCKMODE Flags);
    DLL_EXPORT CKERROR Unlock(void *Ptr1, CKDWORD Bytes1, void *Ptr2, CKDWORD Bytes2);

    //----------------------------------------------------------
    // Position in the sound buffer
    DLL_EXPORT CKDWORD GetPlayPosition();

    DLL_EXPORT int GetPlayedMs();

    //----------------------------------------------------------
    // Creation of the buffer and
    DLL_EXPORT CKERROR Create(CKBOOL FileStreaming, CKSTRING Filename);
    DLL_EXPORT CKERROR Create(CK_WAVESOUND_TYPE Type, CKWaveFormat *Format, int Size);
    DLL_EXPORT CKERROR SetReader(CKSoundReader *Reader);
    DLL_EXPORT CKSoundReader *GetReader();

    DLL_EXPORT void SetDataToRead(int Size);

    DLL_EXPORT CKERROR Recreate(CKBOOL Safe = FALSE);
    DLL_EXPORT void Release();

    DLL_EXPORT CKERROR TryRecreate();

    //----------------------------------------------------------
    // Update the position, according to the attached object

    DLL_EXPORT void UpdatePosition(float deltaT);

    //-------------------------------------------------------------------------
    // Internal functions

    DLL_EXPORT void UpdateFade();
    DLL_EXPORT CKERROR WriteDataFromReader();
    DLL_EXPORT void FillWithBlanks(CKBOOL IncBf = FALSE);
    DLL_EXPORT void InternalStop();

    CKWaveSound(CKContext *Context, CKSTRING Name = NULL);
    virtual ~CKWaveSound();
    virtual CK_CLASSID GetClassID();

    virtual CKStateChunk *Save(CKFile *file, CKDWORD flags);
    virtual CKERROR Load(CKStateChunk *Chunk, CKFile *File);

    virtual CKERROR RemapDependencies(CKDependenciesContext &context);
    virtual CKERROR Copy(CKObject &o, CKDependenciesContext &context);

    virtual void CheckPostDeletion();

    virtual size_t GetMemoryOccupation();

    //--------------------------------------------
    // Class Registering
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int Mode);
    static CKSTRING GetDependencies(int i, int Mode);
    static void Register();
    static CKWaveSound *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

    // Dynamic Cast method (returns NULL if the object can't be cast)
    static CKWaveSound *Cast(CKObject *iO)
    {
        return CKIsChildClassOf(iO, CKCID_WAVESOUND) ? (CKWaveSound *)iO : NULL;
    }

protected:
    CKSoundManager *m_SoundManager;
    void *m_Source;
    CKDWORD m_State;
    CK_ID m_AttachedObject;
    VxVector m_OldPosition;
    VxVector m_Position;
    VxVector m_Direction;
    float m_FinalGain;
    float m_FadeTime;
    float m_CurrentTime;
    int m_BufferPos;
    int m_BufferSize;
    CKSoundReader *m_SoundReader;
    int m_DataRead;
    int m_DataToRead;
    int m_DataPlayed;
    int m_OldCursorPos;
    int m_Duration; // Duration in milliseconds
    CKWaveFormat m_WaveFormat;

    int GetDistanceFromCursor();
    void InternalSetGain(float Gain);

    void SaveSettings();
    void RestoreSettings();

    void *ReallocSource(void *oSource, int alreadyWritten, int newSize);

    // Save Information From Current source
    CKWaveSoundSettings m_2DSetting;
    CKWaveSound3DSettings m_3DSetting;
};

#endif // CKWAVESOUND_H
