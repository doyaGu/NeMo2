#include "CKObjectArray.h"
#include "CKObjectManager.h"

CKObjectArray::CKObjectArray(CKObjectArray *src) {
    m_Position = -1;
    m_Current = nullptr;
    m_Next = nullptr;
    m_Previous = nullptr;
    m_Count = 0;

    if (src && !src->ListEmpty()) {
        src->Reset();
        while (!src->EndOfList()) {
            if (src->GetDataId()) {
                InsertRear(src->GetDataId());
            }
            src->Next();
        }
    }
}

int CKObjectArray::GetCount() {
    return m_Count;
}

int CKObjectArray::GetCurrentPos() {
    return m_Position;
}

CKObject *CKObjectArray::GetData(CKContext *context) {
    return (m_Current) ? context->GetObject(m_Current->m_Data) : nullptr;
}

CK_ID CKObjectArray::GetDataId() {
    return (m_Current) ? m_Current->m_Data : 0;
}

CK_ID CKObjectArray::SetDataId(CK_ID id) {
    if (!m_Current)
        return 0;

    CK_ID oldId = m_Current->m_Data;
    m_Current->m_Data = id;
    return oldId;
}

CK_ID CKObjectArray::SetData(CKObject *obj) {
    if (!m_Current)
        return 0;
    CK_ID oldId = m_Current->m_Data;
    m_Current->m_Data = (obj) ? obj->m_ID : 0;
    return oldId;
}

void CKObjectArray::Reset() {
    m_Position = 0;
    m_Current = m_Next;
}

CKBOOL CKObjectArray::PtrSeek(CKObject *obj) {
    CK_ID id = (obj) ? obj->m_ID : 0;
    return IDSeek(id);
}

CKBOOL CKObjectArray::IDSeek(CK_ID id) {
    Reset();
    for (CK_ID dataId = GetDataId(); dataId != 0; Next(), dataId = GetDataId()) {
        if (dataId == id)
            return TRUE;
    }
    return FALSE;
}

CKBOOL CKObjectArray::PositionSeek(int Pos) {
    if (Pos < 0 || Pos >= m_Count)
        return FALSE;

    if (!m_Current || m_Position == -1)
        Reset();

    if (Pos == m_Position)
        return TRUE;

    if (Pos == m_Count - 1) {
        m_Current = m_Previous;
        m_Position = m_Count - 1;
        return TRUE;
    }

    if (Pos == 0) {
        m_Current = m_Next;
        m_Position = 0;
        return TRUE;
    }

    if (Pos > m_Position) {
        do {
            m_Current = m_Current->m_Next;
        } while (++m_Position != Pos);
        return TRUE;
    }

    if (Pos < m_Position) {
        do {
            m_Current = m_Current->m_Prev;
        } while (--m_Position != Pos);
        return TRUE;
    }

    return FALSE;
}

CK_ID CKObjectArray::Seek(int Pos) {
    return (PositionSeek(Pos) && m_Current) ? m_Current->m_Data : 0;
}

void CKObjectArray::Next() {
    if (m_Current) {
        m_Current = m_Current->m_Next;
        if (m_Current)
            ++m_Position;
        else
            m_Position = -1;
    }
}

void CKObjectArray::Previous() {
    if (m_Current) {
        m_Current = m_Current->m_Prev;
        if (m_Current)
            --m_Position;
        else
            m_Position = -1;
    }
}

int CKObjectArray::GetPosition(CKObject *o) {
    Node *prevCurrent = m_Current;
    int prevPosition = m_Position;

    int pos = -1;
    if (PtrSeek(o))
        pos = m_Position;

    m_Current = prevCurrent;
    m_Position = prevPosition;
    return pos;
}

int CKObjectArray::GetPosition(CK_ID id) {
    Node *prevCurrent = m_Current;
    int prevPosition = m_Position;

    int pos = -1;
    if (IDSeek(id))
        pos = m_Position;

    m_Current = prevCurrent;
    m_Position = prevPosition;
    return pos;
}

CK_ID CKObjectArray::PtrFind(CKObject *o) {
    Node *prevCurrent = m_Current;
    int prevPosition = m_Position;

    CK_ID id = 0;
    if (PtrSeek(o))
        id = GetDataId();

    m_Current = prevCurrent;
    m_Position = prevPosition;
    return id;
}

CK_ID CKObjectArray::IDFind(CK_ID i) {
    Node *prevCurrent = m_Current;
    int prevPosition = m_Position;

    CK_ID id = 0;
    if (IDSeek(i))
        id = GetDataId();

    m_Current = prevCurrent;
    m_Position = prevPosition;
    return id;
}

CK_ID CKObjectArray::PositionFind(int pos) {
    Node *prevCurrent = m_Current;
    int prevPosition = m_Position;

    CK_ID id = 0;
    Reset();
    if (PositionSeek(pos))
        id = GetDataId();

    m_Position = prevPosition;
    m_Current = prevCurrent;
    return id;
}

void CKObjectArray::InsertFront(CKObject *obj) {
    if (obj)
        InsertFront(obj->m_ID);
}

void CKObjectArray::InsertRear(CKObject *obj) {
    if (obj)
        InsertRear(obj->m_ID);
}

void CKObjectArray::InsertAt(CKObject *obj) {
    if (obj)
        InsertAt(obj->m_ID);
}

CKBOOL CKObjectArray::AddIfNotHere(CKObject *obj) {
    if (PtrFind(obj))
        return FALSE;
    InsertRear(obj);
    return TRUE;
}

CKBOOL CKObjectArray::AddIfNotHereSorted(CKObject *obj, OBJECTARRAYCMPFCT CmpFct, CKContext *context) {
    if (PtrFind(obj))
        return FALSE;
    InsertSorted(obj, CmpFct, context);
    return TRUE;
}

void CKObjectArray::InsertFront(CK_ID id) {
    Node *node = new Node;
    if (m_Next) {
        node->m_Prev = nullptr;
        node->m_Next = m_Next;
        m_Next->m_Prev = node;
        node->m_Data = id;
        ++m_Count;
        ++m_Position;
        m_Next = node;
    } else {
        m_Previous = node;
        m_Next = node;
        node->m_Prev = nullptr;
        node->m_Next = nullptr;
        node->m_Data = id;
        ++m_Count;
        m_Position = 0;
        m_Current = m_Next;
    }
}

void CKObjectArray::InsertRear(CK_ID id) {
    Node *node = new Node;
    ++m_Count;
    if (m_Next) {
        node->m_Next = nullptr;
        node->m_Prev = m_Previous;
        m_Previous->m_Next = node;
        node->m_Data = id;
        m_Previous = node;
    } else {
        m_Previous = node;
        m_Next = node;
        node->m_Prev = nullptr;
        node->m_Next = nullptr;
        node->m_Data = id;
        m_Position = 0;
        m_Current = m_Next;
    }
}

void CKObjectArray::InsertAt(CK_ID id) {
    if (m_Current) {
        Node *node = new Node;
        node->m_Next = m_Current;
        node->m_Prev = m_Current->m_Prev;
        node->m_Data = id;
        m_Current->m_Prev = node;
        if (node->m_Prev)
            node->m_Prev->m_Next = node;
        else
            m_Next = node;
        ++m_Position;
        ++m_Count;
    } else {
        InsertFront(id);
    }
}

CKBOOL CKObjectArray::AddIfNotHere(CK_ID id) {
    if (IDFind(id))
        return FALSE;
    InsertRear(id);
    return TRUE;
}

CKBOOL CKObjectArray::AddIfNotHereSorted(CK_ID id, OBJECTARRAYCMPFCT CmpFct, CKContext *context) {
    if (IDFind(id))
        return FALSE;
    InsertSorted(id, CmpFct, context);
    return TRUE;
}

CKERROR CKObjectArray::Append(CKObjectArray *array) {
    if (!array)
        return CKERR_INVALIDPARAMETER;

    if (!array->ListEmpty()) {
        array->Reset();
        while (!array->EndOfList()) {
            CK_ID id = array->GetDataId();
            InsertRear(id);
            array->Next();
        }
    }

    return CK_OK;
}

CK_ID CKObjectArray::RemoveFront() {
    Node *node = m_Next;
    if (node) {
        if (node->m_Next) {
            node->m_Next->m_Prev = nullptr;
            m_Next = node->m_Next;
            if (m_Current == node)
                m_Current = node->m_Next;
            else
                --m_Position;
        } else {
            m_Position = -1;
            m_Next = nullptr;
            m_Previous = nullptr;
            m_Current = nullptr;
        }
        CK_ID id = node->m_Data;
        delete node;
        --m_Count;
        return id;
    } else {
        m_Count = 0;
        return 0;
    }
}

CK_ID CKObjectArray::RemoveRear() {
    Node *node = m_Previous;
    if (node) {
        if (node->m_Prev) {
            node->m_Prev->m_Next = nullptr;
            m_Previous = node->m_Prev;
            if (m_Current == node) {
                --m_Position;
                m_Current = node->m_Prev;
            }
        } else {
            m_Position = -1;
            m_Next = nullptr;
            m_Previous = nullptr;
            m_Current = nullptr;
        }
        CK_ID id = node->m_Data;
        delete node;
        --m_Count;
        return id;
    } else {
        m_Count = 0;
        return 0;
    }
}

CK_ID CKObjectArray::RemoveAt() {
    Node *node = m_Current;
    if (node) {
        if (node == m_Previous) {
            return RemoveRear();
        } else if (node == m_Next) {
            return RemoveFront();
        } else {
            node->m_Prev->m_Next = node->m_Next;
            node->m_Next->m_Prev = node->m_Prev;
            --m_Count;
            m_Current = node->m_Next;
            CK_ID id = node->m_Data;
            delete node;
            return id;
        }
    } else {
        return 0;
    }
}

CKBOOL CKObjectArray::Remove(CKObject *obj) {
    CK_ID id = (obj) ? obj->m_ID : 0;
    return Remove(id);
}

CKBOOL CKObjectArray::Remove(CK_ID id) {
    Node *prevCurrent = m_Current;
    int prevPosition = m_Position;
    CKBOOL result;
    if (IDSeek(id)) {
        if (prevCurrent == m_Current) {
            RemoveAt();
            return TRUE;
        }
        RemoveAt();
        result = TRUE;
    } else {
        m_Position = prevPosition;
        result = FALSE;
    }
    m_Current = prevCurrent;
    return result;
}

void CKObjectArray::Clear() {
    if (m_Count == 0)
        return;

    Reset();
    int previousCount = m_Count + 1;
    while (m_Count > 0 && m_Count < previousCount) {
        previousCount = m_Count;
        Node *node = m_Next;
        if (!node) {
            m_Count = 0;
            return;
        }

        if (node->m_Next) {
            node->m_Next->m_Prev = nullptr;
            Node *next = node->m_Next;
            bool currentWasHead = (m_Current == node);
            m_Next = next;
            if (currentWasHead)
                m_Current = next;
            else
                --m_Position;
            --m_Count;
        } else {
            m_Position = -1;
            m_Next = nullptr;
            m_Previous = nullptr;
            m_Current = nullptr;
            m_Count = 0;
        }

        delete node;
    }
}

CKBOOL CKObjectArray::EndOfList() {
    return m_Current == nullptr;
}

CKBOOL CKObjectArray::ListEmpty() {
    return m_Count == 0;
}

void CKObjectArray::SwapCurrentWithNext() {
    Node *node = m_Current;
    if (node) {
        Node *nextNode = node->m_Next;
        if (nextNode) {
            CK_ID id = node->m_Data;
            node->m_Data = nextNode->m_Data;
            nextNode->m_Data = id;
        }
    }
}

void CKObjectArray::SwapCurrentWithPrevious() {
    Node *node = m_Current;
    if (node) {
        if (node->m_Prev) {
            CK_ID id = node->m_Data;
            node->m_Data = node->m_Prev->m_Data;
            m_Current->m_Prev->m_Data = id;
        }
    }
}

CKBOOL CKObjectArray::Check(CKContext *context) {
    Reset();
    if (!m_Current)
        return FALSE;

    CKBOOL result = FALSE;
    Node *node;
    do {
        node = m_Current->m_Next;
        if (!context->GetObject(m_Current->m_Data)) {
            Node *prev = m_Current->m_Prev;
            if (prev)
                prev->m_Next = m_Current->m_Next;
            Node *next = m_Current->m_Next;
            if (next)
                next->m_Prev = m_Current->m_Prev;
            Node *current = m_Current;
            if (m_Current == m_Next)
                m_Next = current->m_Next;
            if (current == m_Previous)
                m_Previous = current->m_Prev;
            --m_Count;
            result = TRUE;
            delete current;
            m_Current = nullptr;
        }
        m_Current = node;
    } while (node);

    return result;
}

void CKObjectArray::Sort(OBJECTARRAYCMPFCT CmpFct, CKContext *context) {
    if (!CmpFct)
        return;

    for (int i = 1; i < m_Count; ++i) {
        CKBOOL noSwap = TRUE;
        m_Current = m_Previous;
        m_Position = m_Count - 1;
        if (i > m_Count - 1)
            break;

        for (int j = m_Count - i; j != 0; --j) {
            CKObject *obj1 = context->GetObject(m_Current->m_Data);
            Previous();
            CKObject *obj2 = context->GetObject(m_Current->m_Data);
            if (CmpFct(obj1, obj2) < 0) {
                SwapCurrentWithNext();
                noSwap = FALSE;
            }
        }

        if (noSwap)
            break;
    }
}

void CKObjectArray::InsertSorted(CKObject *o, OBJECTARRAYCMPFCT CmpFct, CKContext *context) {
    InsertRear(o);
    Sort(CmpFct, context);
}

void CKObjectArray::InsertSorted(CK_ID id, OBJECTARRAYCMPFCT CmpFct, CKContext *context) {
    InsertRear(id);
    Sort(CmpFct, context);
}
