#include "CKDebugContext.h"
#include "CKBehavior.h"
#include "CKBeObject.h"
#include "CKCharacter.h"

void CKDebugContext::Init(XObjectPointerArray &array, float deltat) {
    delta = deltat;
    CurrentObject = nullptr;
    CurrentScript = nullptr;
    CurrentBehavior = nullptr;
    SubBehavior = nullptr;

    for (XObjectPointerArray::Iterator it = array.Begin(); it != array.End(); ++it)
        ObjectsToExecute.InsertRear(*it);
    ScriptsToExecute.Clear();
    BehaviorStack.Clear();

    InDebug = TRUE;
}

void CKDebugContext::StepInto(CKBehavior *beh) {
    BehaviorStack.InsertRear(CurrentBehavior);
    beh->ExecuteStepStart();
    CurrentBehavior = beh;
    CurrentBehaviorAction = CKDEBUG_BEHEXECUTE;
}

void CKDebugContext::StepBehavior() {
    CurrentBehavior->ExecuteStep(delta, this);
    if (CurrentBehaviorAction == CKDEBUG_BEHEXECUTEDONE) {
        if (BehaviorStack.ListEmpty()) {
            CurrentBehavior = nullptr;
            CurrentBehaviorAction = CKDEBUG_SCRIPTEXECUTEDONE;
            if (CurrentObject) {
                if (CurrentScript) {
                    CurrentObject->m_LastExecutionTime = CurrentScript->GetLastExecutionTime() + CurrentObject->m_LastExecutionTime;
                }
            }
            CurrentScript = nullptr;
        } else {
            SubBehavior = CurrentBehavior;
            CK_ID id = BehaviorStack.RemoveRear();
            CurrentBehavior = (CKBehavior *) m_Context->GetObject(id);
        }
    }
}

CKBOOL CKDebugContext::DebugStep() {
    if (CurrentBehavior) {
        StepBehavior();
    } else if (CurrentScript) {
        BehaviorStack.Clear();
        BehaviorStack.InsertRear(CurrentScript);
        CurrentScript->ResetExecutionTime();
        CurrentScript->ExecuteStepStart();
        SubBehavior = nullptr;
        CurrentBehavior = CurrentScript;
        CurrentBehaviorAction = CKDEBUG_BEHEXECUTE;
    } else if (CurrentObject) {
        if (ScriptsToExecute.ListEmpty()) {
            CurrentObject = nullptr;
        } else {
            while (!ScriptsToExecute.ListEmpty()) {
                CK_ID id = ScriptsToExecute.RemoveFront();
                CKBehavior *beh = (CKBehavior *) m_Context->GetObject(id);
                CurrentScript = beh;
                if (!beh)
                    continue;

                CK_BEHAVIOR_FLAGS flags = beh->GetFlags();
                if ((flags & (CKBEHAVIOR_ACTIVATENEXTFRAME | CKBEHAVIOR_DEACTIVATENEXTFRAME)) != 0) {
                    CurrentScript->Activate(flags & CKBEHAVIOR_ACTIVATENEXTFRAME, flags & CKBEHAVIOR_RESETNEXTFRAME);
                    CurrentScript->ModifyFlags(0, CKBEHAVIOR_ACTIVATENEXTFRAME | CKBEHAVIOR_RESETNEXTFRAME | CKBEHAVIOR_DEACTIVATENEXTFRAME);
                }
                if (CurrentScript->IsActive())
                    break;
                CurrentScript = nullptr;
            }
            if (!CurrentScript)
                CurrentObject = nullptr;
        }
    } else {
        if (ObjectsToExecute.ListEmpty())
            return FALSE;

        while (true) {
            if (ObjectsToExecute.ListEmpty())
                break;
            CK_ID id = ObjectsToExecute.RemoveFront();
            CurrentObject = (CKBeObject *) m_Context->GetObject(id);
            if (CurrentObject)
                break;
        }

        if (!CurrentObject)
            return FALSE;

        ++m_Context->m_Stats.ActiveObjectsExecuted;
        CurrentObject->m_LastExecutionTime = 0.0f;
        ScriptsToExecute.Clear();
        CurrentScript = nullptr;
        CurrentBehavior = nullptr;
        SubBehavior = nullptr;

        XObjectPointerArray *array = CurrentObject->m_ScriptArray;
        if (array) {
            for (XObjectPointerArray::Iterator it = array->Begin(); it != array->End(); ++it) {
                CK_ID id = (*it) ? (*it)->m_ID : 0;
                ScriptsToExecute.InsertRear(id);
                array = CurrentObject->m_ScriptArray;
            }
        }

        VxTimeProfiler profiler;
        if (CurrentObject->GetClassID() == CKCID_CHARACTER) {
            CKCharacter *character = (CKCharacter *)CurrentObject;
            if (character->IsAutomaticProcess()) {
                character->ProcessAnimation(delta);
            }
        }

        float d = profiler.Current();
        CurrentObject->m_LastExecutionTime += d;
        m_Context->m_Stats.BehaviorCodeExecution += d;
        m_Context->m_Stats.TotalBehaviorExecution += d;
    }

    return TRUE;
}

void CKDebugContext::Clear() {
    CurrentObject = nullptr;
    CurrentScript = nullptr;
    CurrentBehavior = nullptr;
    SubBehavior = nullptr;

    ObjectsToExecute.Clear();
    ScriptsToExecute.Clear();
    BehaviorStack.Clear();

    InDebug = FALSE;
}
