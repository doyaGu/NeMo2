#ifndef CKBEHAVIOR_H
#define CKBEHAVIOR_H

#include "CKSceneObject.h"
#include "XObjectArray.h"

struct BehaviorBlockData
{
    CKGUID m_Guid;
    CKBEHAVIORFCT m_Function;
    CKBEHAVIORCALLBACKFCT m_Callback;
    CKDWORD m_CallbackMask;
    void *m_CallbackArg;
    CKDWORD m_Version;

    BehaviorBlockData()
    {
        m_Guid = CKGUID();
        m_Function = nullptr;
        m_Callback = nullptr;
        m_CallbackMask = 0;
        m_CallbackArg = nullptr;
        m_Version = 0;
    }

    BehaviorBlockData(const BehaviorBlockData &data)
    {
        m_Guid = data.m_Guid;
        m_Function = data.m_Function;
        m_Callback = data.m_Callback;
        m_CallbackMask = data.m_CallbackMask;
        m_CallbackArg = data.m_CallbackArg;
        m_Version = data.m_Version;
    }
};

struct BehaviorGraphData
{
    XObjectPointerArray m_Operations;
    XObjectPointerArray m_SubBehaviors;
    XObjectPointerArray m_SubBehaviorLinks;
    XObjectPointerArray m_Links;
    CKBehavior **m_BehaviorIterators;
    CKWORD m_BehaviorIteratorCount;
    CKWORD m_BehaviorIteratorIndex;

    BehaviorGraphData()
    {
        m_BehaviorIterators = nullptr;
        m_BehaviorIteratorCount = 0;
        m_BehaviorIteratorIndex = 0;
    }

    BehaviorGraphData(const BehaviorGraphData &data)
    {
        m_Operations = data.m_Operations;
        m_SubBehaviors = data.m_SubBehaviors;
        m_SubBehaviorLinks = data.m_SubBehaviorLinks;
        m_Links = data.m_Links;
        m_BehaviorIterators = nullptr;
        m_BehaviorIteratorCount = 0;
        m_BehaviorIteratorIndex = 0;
    }

    ~BehaviorGraphData();
};

/**************************************************************************
{filename:CKBehavior}
Name: CKBehavior

Summary: Class for defining behaviors

Remarks:
    + The CKBehavior class provides functionalities for defining a behavior as a function or as a graph,
    to set various parameters about how and when the behavior is activated,
    to link them to other behaviors through CKBehaviorLink instances or to create
    them from CKBehaviorPrototype, for creating CKBehaviorPrototype from a behavior.

    + Each behavior has an owner, which is an instance of CKBeObject. The owner saves its
    behaviors with itself when saved into a file, or with CKSaveObjectState. The behavior
    usually applies to its owner, or to parts of its owner, though it can apply to other
    objects as well. If the behavior is a graph, all of its sub-behaviors inherits the same owner.

    + A Behavior is considered to as a box with inputs/outputs and input,output and local parameters.
    Inside this box the behavior may be implemented using a graph of sub-behaviors or using a C++ function,
    depending on which method UseGraph or UseFunction has been specified. For behaviors using a function,
    the function may be defined internally, or in a external code file (dll, or other depending on the OS).

    + If you want to implement the behavior with a C++ function without defining a prototype, you should specify the function with the
    SetFunction() method. This function will be called at each process cycle, with one argument:
    the elapsed time since the last call, in milliseconds.

    + Behaviors are real time pieces of algorithm. When a behavior is active and enabled or is waiting for a message, it gets
    called in an iterative fashion at each loop of the process cycle. Each time it is called, the behavior
    does the minimum amount of processing depending on the time elapsed since the last time it was called.
    No behavior should hold on the CPU more than necessary, as each other active behaviors have to be called
    in the process cycle.

    + Behavior execution is managed by its owner, and by the scene to which its owner belongs.
    A CKBeObject instance is declared in its scene as being active and/or enabled with the CKScene::SetObjectFlags or CKScene::Activate method.
    A disabled object's behaviors are never called. Setting an object as active has the effect of having
    all of its behaviors called during the next process cycle. If you add an object to a scene and
    want its behaviors to be called when the scene is started with CKLevel::LaunchScene, you should
    set the flags accordingly with CKScene::SetObjectFlags.
    If you don't use Scene or have only one scene (the level) setting the flags is not necessary.

    + If the behavior is implemented using an execution function, this function decides whether the
    behavior should remain active or not. This is done with the return value of the
    function. If the return value is CKBR_OK, the behavior is considered as being over and is set to be inactive
    at the next process cycle. It may become active again at a later time, if one of its entries gets
    activated.

    + If the behavior is implemented as a graph of sub-behaviors, it is considered active if at least one of its sub-behaviors is active, and
    will be called during the next process cycle.

    + Behavior prototypes are a sort of model of behaviors. One can create a behavior from a prototype.
    Prototypes are typically used in external pieces of code like dlls.	Prototypes are typically used
    for storing reusable behaviors.

    + Behaviors usually have inputs and outputs, which may be active or inactive. The inputs and outputs have
    no value, they are like switches which may be on or off. A behavior is activated if at least one of its entries is
    active. When all of the entries are inactive and all of its sub-behaviors are inactive and the behavior is not waiting for a message, the behavior
    is deactivated.

    + The outputs of a behavior are activated or deactivated by the behavior, as a result of its internal
    processing and states. Outputs of a behavior may be linked to other behaviors through links, in which case
    activating an output will trigger the activation of one or more other behaviors. This is the basic principle
    through which processing travels along behaviors.

    + A behavior may have zero or more inputs and zero or more outputs. The output of a behavior may be linked to
    an input of the same behavior.

    + Outputs of a behavior are linked to inputs or other behaviors through instances of CKBehaviorLink.
    These links have an activation delay, which may be 0. If it is zero, the input gets activated at the same
    process cycle which may introduce infinite loops. Such Loops are detected at execution time when exceeding
    the value defined by CKSetBehaviorMaxIteration (Default : 8000)


    + Behaviors may exchange or store values using parameters, which are instances of
    CKParameterOut, CKParameterIn. These values may be arbitrarily complex, and
    you may define your own type of parameters. CKParameterOut instances are used by the behavior to export values,
    and CKParameterIn instances are used to get values from the outside.

    + Components of a behavior like parameters, I/O may be named, like all objects of CK. This name
    is only a convenience and is not mandatory. These names are saved in the state or in the file
    with the behavior and may eat up space, so you should use them carefully if this is an issue for
    your application.

    + Behaviors may wait for messages. They specify that they wait for messages by registering
    themselves with the message manager, using the CKMessageManager::RegisterWait method and by providing
    an input or an output to activate when a corresponding message is received.

    + The class id of CKBehavior is CKCID_BEHAVIOR.




See also: CKBeObject, CKScene, CKParameterOut, CKParameterIn, CKBehaviorLink, CKBehaviorPrototype
************************************************************************************/
class CKBehavior : public CKSceneObject
{
    friend class CKBehaviorManager;
    friend class CKBeObject;
    friend class CKFile;
    friend class CKParameterIn;
    friend class CKParameter;
    friend class CKParameterLocal;
    friend class CKParameterOut;
    friend class CKParameterOperation;
    friend class CKBehaviorLink;
    friend class CKBehaviorIO;
    friend class CKDebugContext;

public:
    //----------------------------------------------------------------
    // Behavior Typage
    DLL_EXPORT CK_BEHAVIOR_TYPE GetType();
    DLL_EXPORT void SetType(CK_BEHAVIOR_TYPE type);

    DLL_EXPORT void SetFlags(CK_BEHAVIOR_FLAGS flags);
    DLL_EXPORT CK_BEHAVIOR_FLAGS GetFlags();
    DLL_EXPORT CK_BEHAVIOR_FLAGS ModifyFlags(CKDWORD Add, CKDWORD Remove);

    //---------- BuildingBlock or Graph
    DLL_EXPORT void UseGraph();
    DLL_EXPORT void UseFunction();
    DLL_EXPORT CKBOOL IsUsingFunction();

    //----------------------------------------------------------------
    // Targetable Behavior
    DLL_EXPORT CKBOOL IsTargetable();
    DLL_EXPORT CKBeObject *GetTarget();
    DLL_EXPORT CKERROR UseTarget(CKBOOL Use = TRUE);
    DLL_EXPORT CKBOOL IsUsingTarget();
    DLL_EXPORT CKParameterIn *GetTargetParameter();
    DLL_EXPORT void SetAsTargetable(CKBOOL target = TRUE);
    DLL_EXPORT CKParameterIn *ReplaceTargetParameter(CKParameterIn *targetParam);
    DLL_EXPORT CKParameterIn *RemoveTargetParameter();

    //----------------------------------------------------------------
    // Object type to which this behavior applies
    DLL_EXPORT CK_CLASSID GetCompatibleClassID();
    DLL_EXPORT void SetCompatibleClassID(CK_CLASSID cid);

    //----------------------------------------------------------------
    // Execution Function
    DLL_EXPORT void SetFunction(CKBEHAVIORFCT fct);
    DLL_EXPORT CKBEHAVIORFCT GetFunction();

    //----------------------------------------------------------------
    // Callbacks Function
    DLL_EXPORT void SetCallbackFunction(CKBEHAVIORCALLBACKFCT fct);
    DLL_EXPORT int CallCallbackFunction(CKDWORD Message);
    DLL_EXPORT int CallSubBehaviorsCallbackFunction(CKDWORD Message, CKGUID *behguid = NULL);

    //--------------------------------------------------------------
    // RunTime Functions
    DLL_EXPORT CKBOOL IsActive();
    DLL_EXPORT int Execute(float delta);

    //--------------------------------------------------------------
    //
    DLL_EXPORT CKBOOL IsParentScriptActiveInScene(CKScene *scn);
    DLL_EXPORT int GetShortestDelay(CKBehavior *beh);

    //--------------------------------------------------------------
    // Owner Functions
    DLL_EXPORT CKBeObject *GetOwner();
    DLL_EXPORT CKBehavior *GetParent();
    DLL_EXPORT CKBehavior *GetOwnerScript();

    //----------------------------------------------------------------
    // Creation From prototype
    DLL_EXPORT CKERROR InitFromPrototype(CKBehaviorPrototype *proto);
    DLL_EXPORT CKERROR InitFromGuid(CKGUID Guid);
    DLL_EXPORT CKERROR InitFctPtrFromGuid(CKGUID Guid);
    DLL_EXPORT CKERROR InitFctPtrFromPrototype(CKBehaviorPrototype *proto);
    DLL_EXPORT CKGUID GetPrototypeGuid();
    DLL_EXPORT CKBehaviorPrototype *GetPrototype();
    DLL_EXPORT CKSTRING GetPrototypeName();
    DLL_EXPORT CKDWORD GetVersion();
    DLL_EXPORT void SetVersion(CKDWORD version);

    //----------------------------------------------------------------
    // Outputs
    DLL_EXPORT void ActivateOutput(int pos, CKBOOL active = TRUE);
    DLL_EXPORT CKBOOL IsOutputActive(int pos);
    DLL_EXPORT CKBehaviorIO *RemoveOutput(int pos);
    DLL_EXPORT CKERROR DeleteOutput(int pos);
    DLL_EXPORT CKBehaviorIO *GetOutput(int pos);
    DLL_EXPORT int GetOutputCount();
    DLL_EXPORT int GetOutputPosition(CKBehaviorIO *io);
    DLL_EXPORT int AddOutput(CKSTRING name);
    DLL_EXPORT CKBehaviorIO *ReplaceOutput(int pos, CKBehaviorIO *io);
    DLL_EXPORT CKBehaviorIO *CreateOutput(CKSTRING name);

    //----------------------------------------------------------------
    // Inputs
    DLL_EXPORT void ActivateInput(int pos, CKBOOL active = TRUE);
    DLL_EXPORT CKBOOL IsInputActive(int pos);
    DLL_EXPORT CKBehaviorIO *RemoveInput(int pos);
    DLL_EXPORT CKERROR DeleteInput(int pos);
    DLL_EXPORT CKBehaviorIO *GetInput(int pos);
    DLL_EXPORT int GetInputCount();
    DLL_EXPORT int GetInputPosition(CKBehaviorIO *io);
    DLL_EXPORT int AddInput(CKSTRING name);
    DLL_EXPORT CKBehaviorIO *ReplaceInput(int pos, CKBehaviorIO *io);
    DLL_EXPORT CKBehaviorIO *CreateInput(CKSTRING name);

    //----------------------------------------------------------------
    // inputs Parameters
    DLL_EXPORT CKERROR ExportInputParameter(CKParameterIn *p);
    DLL_EXPORT CKParameterIn *CreateInputParameter(CKSTRING name, CKParameterType type);
    DLL_EXPORT CKParameterIn *CreateInputParameter(CKSTRING name, CKGUID guid);
    DLL_EXPORT CKParameterIn *InsertInputParameter(int pos, CKSTRING name, CKParameterType type);
    DLL_EXPORT void AddInputParameter(CKParameterIn *in);
    DLL_EXPORT int GetInputParameterPosition(CKParameterIn *in);
    DLL_EXPORT CKParameterIn *GetInputParameter(int pos);
    DLL_EXPORT CKParameterIn *RemoveInputParameter(int pos);
    DLL_EXPORT CKParameterIn *ReplaceInputParameter(int pos, CKParameterIn *param);
    DLL_EXPORT int GetInputParameterCount();
    DLL_EXPORT CKERROR GetInputParameterValue(int pos, void *buf);
    DLL_EXPORT void *GetInputParameterReadDataPtr(int pos);
    DLL_EXPORT CKObject *GetInputParameterObject(int pos);
    DLL_EXPORT CKBOOL IsInputParameterEnabled(int pos);
    DLL_EXPORT void EnableInputParameter(int pos, CKBOOL enable);

    //----------------------------------------------------------------
    // outputs Parameters
    DLL_EXPORT CKERROR ExportOutputParameter(CKParameterOut *p);
    DLL_EXPORT CKParameterOut *CreateOutputParameter(CKSTRING name, CKParameterType type);
    DLL_EXPORT CKParameterOut *CreateOutputParameter(CKSTRING name, CKGUID guid);
    DLL_EXPORT CKParameterOut *InsertOutputParameter(int pos, CKSTRING name, CKParameterType type);
    DLL_EXPORT CKParameterOut *GetOutputParameter(int pos);
    DLL_EXPORT int GetOutputParameterPosition(CKParameterOut *p);
    DLL_EXPORT CKParameterOut *ReplaceOutputParameter(int pos, CKParameterOut *p);
    DLL_EXPORT CKParameterOut *RemoveOutputParameter(int pos);
    DLL_EXPORT void AddOutputParameter(CKParameterOut *out);
    DLL_EXPORT int GetOutputParameterCount();
    DLL_EXPORT CKERROR GetOutputParameterValue(int pos, void *buf);
    DLL_EXPORT CKERROR SetOutputParameterValue(int pos, const void *buf, int size = 0);
    DLL_EXPORT void *GetOutputParameterWriteDataPtr(int pos);
    DLL_EXPORT CKERROR SetOutputParameterObject(int pos, CKObject *obj);
    DLL_EXPORT CKObject *GetOutputParameterObject(int pos);
    DLL_EXPORT CKBOOL IsOutputParameterEnabled(int pos);
    DLL_EXPORT void EnableOutputParameter(int pos, CKBOOL enable);

    DLL_EXPORT void SetInputParameterDefaultValue(CKParameterIn *pin, CKParameter *plink);

    //----------------------------------------------------------------
    // Local Parameters
    DLL_EXPORT CKParameterLocal *CreateLocalParameter(CKSTRING name, CKParameterType type);
    DLL_EXPORT CKParameterLocal *CreateLocalParameter(CKSTRING name, CKGUID guid);
    DLL_EXPORT CKParameterLocal *GetLocalParameter(int pos);
    DLL_EXPORT CKParameterLocal *RemoveLocalParameter(int pos);
    DLL_EXPORT void AddLocalParameter(CKParameterLocal *loc);
    DLL_EXPORT int GetLocalParameterPosition(CKParameterLocal *);
    DLL_EXPORT int GetLocalParameterCount();
    DLL_EXPORT CKERROR GetLocalParameterValue(int pos, void *buf);
    DLL_EXPORT CKERROR SetLocalParameterValue(int pos, const void *buf, int size = 0);
    DLL_EXPORT void *GetLocalParameterWriteDataPtr(int pos);
    DLL_EXPORT void *GetLocalParameterReadDataPtr(int pos);
    DLL_EXPORT CKObject *GetLocalParameterObject(int pos);
    DLL_EXPORT CKERROR SetLocalParameterObject(int pos, CKObject *obj);
    DLL_EXPORT CKBOOL IsLocalParameterSetting(int pos);

    //----------------------------------------------------------------
    // Run Time Graph : Activation
    DLL_EXPORT void Activate(CKBOOL Active = TRUE, CKBOOL breset = FALSE);

    //----------------------------------------------------------------
    // Run Time Graph : Sub Behaviors
    DLL_EXPORT CKERROR AddSubBehavior(CKBehavior *cbk);
    DLL_EXPORT CKBehavior *RemoveSubBehavior(CKBehavior *cbk);
    DLL_EXPORT CKBehavior *RemoveSubBehavior(int pos);
    DLL_EXPORT CKBehavior *GetSubBehavior(int pos);
    DLL_EXPORT int GetSubBehaviorCount();

    //----------------------------------------------------------------
    // Run Time Graph : Links between sub behaviors
    DLL_EXPORT CKERROR AddSubBehaviorLink(CKBehaviorLink *cbkl);
    DLL_EXPORT CKBehaviorLink *RemoveSubBehaviorLink(CKBehaviorLink *cbkl);
    DLL_EXPORT CKBehaviorLink *RemoveSubBehaviorLink(int pos);
    DLL_EXPORT CKBehaviorLink *GetSubBehaviorLink(int pos);
    DLL_EXPORT int GetSubBehaviorLinkCount();

    //----------------------------------------------------------------
    // Run Time Graph : Parameter Operation
    DLL_EXPORT CKERROR AddParameterOperation(CKParameterOperation *op);
    DLL_EXPORT CKParameterOperation *GetParameterOperation(int pos);
    DLL_EXPORT CKParameterOperation *RemoveParameterOperation(int pos);
    DLL_EXPORT CKParameterOperation *RemoveParameterOperation(CKParameterOperation *op);
    DLL_EXPORT int GetParameterOperationCount();

    //----------------------------------------------------------------
    // Run Time Graph : Behavior Priority
    DLL_EXPORT int GetPriority();
    DLL_EXPORT void SetPriority(int priority);

    //------- Profiling ---------------------
    DLL_EXPORT float GetLastExecutionTime();

    //-------------------------------------------------------------------------
    // Internal functions

    DLL_EXPORT CKERROR SetOwner(CKBeObject *owner, CKBOOL callback = TRUE);
    DLL_EXPORT CKERROR SetSubBehaviorOwner(CKBeObject *owner, CKBOOL callback = TRUE);

    DLL_EXPORT void NotifyEdition();
    DLL_EXPORT void NotifySettingsEdition();

    DLL_EXPORT CKStateChunk *GetInterfaceChunk();
    DLL_EXPORT void SetInterfaceChunk(CKStateChunk *state);

    CKBehavior(CKContext *Context, CKSTRING name = NULL);
    virtual ~CKBehavior();
    virtual CK_CLASSID GetClassID();

    virtual void PreSave(CKFile *file, CKDWORD flags);
    virtual CKStateChunk *Save(CKFile *file, CKDWORD flags);
    virtual CKERROR Load(CKStateChunk *chunk, CKFile *file);
    virtual void PostLoad();

    virtual void PreDelete();

    virtual int GetMemoryOccupation();
    virtual CKBOOL IsObjectUsed(CKObject *obj, CK_CLASSID cid);

    //--------------------------------------------
    // Dependencies functions	{Secret}
    virtual CKERROR PrepareDependencies(CKDependenciesContext &context);
    virtual CKERROR RemapDependencies(CKDependenciesContext &context);
    virtual CKERROR Copy(CKObject &o, CKDependenciesContext &context);

    //--------------------------------------------
    // Class Registering {secret}
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKBehavior *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

    // Dynamic Cast method (returns NULL if the object can't be casted)
    static CKBehavior *Cast(CKObject *iO)
    {
        return CKIsChildClassOf(iO, CKCID_BEHAVIOR) ? (CKBehavior *)iO : NULL;
    }

    void Reset();
    void ErrorMessage(CKSTRING Error, CKSTRING Context, CKBOOL ShowOwner = TRUE, CKBOOL ShowScript = TRUE);
    void ErrorMessage(CKSTRING Error, CKDWORD Context, CKBOOL ShowOwner = TRUE, CKBOOL ShowScript = TRUE);
    void SetPrototypeGuid(CKGUID ckguid);

protected:
    CK_ID m_Owner;
    CK_ID m_BehParent;
    XSObjectPointerArray m_InputArray;
    XSObjectPointerArray m_OutputArray;
    XSObjectPointerArray m_InParameter;
    XSObjectPointerArray m_OutParameter;
    XSObjectPointerArray m_LocalParameter;
    CK_CLASSID m_CompatibleClassID;
    int m_Priority;
    CKDWORD m_Flags;
    CKStateChunk *m_InterfaceChunk;
    BehaviorGraphData *m_GraphData;
    BehaviorBlockData *m_BlockData;
    float m_LastExecutionTime;
    CK_ID m_InputTargetParam;

    void SetParent(CKBehavior *parent);
    void SortSubs();
    void ResetExecutionTime();
    void ExecuteStepStart();
    int ExecuteStep(float delta, CKDebugContext *Context);
    void WarnInfiniteLoop();
    int InternalGetShortestDelay(CKBehavior *beh, XObjectPointerArray &behparsed);
    void CheckIOsActivation();
    void CheckBehaviorActivity();
    int ExecuteFunction();
    void FindNextBehaviorsToExecute(CKBehavior *beh);
    void HierarchyPostLoad();

    static int BehaviorPrioritySort(CKObject *o1, CKObject *o2);
    static int BehaviorPrioritySort(const void *elem1, const void *elem2);

    void ApplyPatchLoad();
};

#endif // CKBEHAVIOR_H
