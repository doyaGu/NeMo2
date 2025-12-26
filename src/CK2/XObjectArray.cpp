#include "XObjectArray.h"

#include "CKStateChunk.h"

CK_ID XObjectPointerArray::GetObjectID(unsigned int i) const {
    if (i >= Size())
        return 0;
    return m_Begin[i]->m_ID;
}

CKBOOL XObjectPointerArray::Check() {
    CKObject **it = m_Begin;
    const int size = Size();
    int keptCount = 0;
    int i;
    for (i = 0; i < size; ++i) {
        CKObject *obj = *it++;
        if (obj->IsToBeDeleted())
            break;
        ++keptCount;
    }

    const int firstDeletedIndex = i + 1;
    if (firstDeletedIndex < size) {
        int remaining = size - firstDeletedIndex;
        do {
            CKObject *obj = *it++;
            if (!obj->IsToBeDeleted())
                m_Begin[keptCount++] = obj;
            --remaining;
        } while (remaining);
    }

    m_End = &m_Begin[keptCount];
    return keptCount != size;
}

void XObjectPointerArray::Load(CKContext *Context, CKStateChunk *chunk) {
    *this = chunk->ReadXObjectArray(Context);
}

void XObjectPointerArray::Save(CKStateChunk *chunk) const {
    chunk->StartObjectIDSequence(Size());
    for (CKObject **it = m_Begin; it != m_End; ++it) {
        chunk->WriteObjectSequence(*it);
    }
}

void XObjectPointerArray::Prepare(CKDependenciesContext &context) const {
    for (CKObject **it = m_Begin; it != m_End; ++it) {
        CKObject *obj = *it;
        if (obj)
            obj->PrepareDependencies(context);
    }
}

void XObjectPointerArray::Remap(CKDependenciesContext &context) {
    for (CKObject **it = m_Begin; it != m_End; ++it) {
        if (*it)
            *it = context.Remap(*it);
    }
}

void XSObjectArray::ConvertFromObjects(const XSArray<CKObject *> &array) {
    Clear();
    for (auto it = array.Begin(); it != array.End(); ++it) {
        if (it && *it)
            PushBack((*it)->GetID());
    }
}

CKBOOL XSObjectArray::Check(CKContext *Context) {
    if (!Context)
        return FALSE;

    const int size = Size();
    const int newSize = Context->m_ObjectManager->CheckIDArray(Begin(), size);
    Resize(newSize);
    return newSize != size;
}

CKBOOL XSObjectArray::AddIfNotHere(CK_ID id) {
    for (CK_ID *it = m_Begin; it != m_End; ++it) {
        if (*it == id)
            return FALSE;
    }
    PushBack(id);
    return TRUE;
}

CKBOOL XSObjectArray::AddIfNotHere(CKObject *obj) {
    if (!obj)
        return FALSE;

    CK_ID id = obj->GetID();
    for (CK_ID *it = m_Begin; it != m_End; ++it) {
        if (*it == id)
            return FALSE;
    }
    PushBack(id);
    return TRUE;
}

CKObject *XSObjectArray::GetObject(CKContext *Context, unsigned int i) const {
    if (!Context || i >= Size())
        return nullptr;
    return Context->GetObject(m_Begin[i]);
}

CKBOOL XSObjectArray::RemoveObject(CKObject *obj) {
    if (!obj)
        return FALSE;
    for (CK_ID *it = m_Begin; it != m_End; ++it) {
        if (*it == obj->GetID()) {
            RemoveAt(it - m_Begin);
            return TRUE;
        }
    }
    return FALSE;
}

CKBOOL XSObjectArray::FindObject(CKObject *obj) const {
    CK_ID id = obj ? obj->GetID() : 0;
    for (CK_ID *it = m_Begin; it != m_End; ++it) {
        if (*it == id)
            return TRUE;
    }
    return FALSE;
}

void XSObjectArray::Load(CKStateChunk *chunk) {
    const int count = chunk->StartReadSequence();
    if (count == 0)
        return;

    if (chunk->GetChunkVersion() >= 4) {
        Resize(count);
        for (int i = 0; i < count; ++i) {
            m_Begin[i] = chunk->ReadObjectID();
        }
    } else {
        chunk->Skip(sizeof(CKDWORD));

        const int objectCount = chunk->ReadInt();
        Resize(objectCount);
        for (int i = 0; i < objectCount; ++i) {
            m_Begin[i] = chunk->ReadInt();
        }
    }
}

void XSObjectArray::Save(CKStateChunk *chunk, CKContext *ctx) const {
    if (!chunk)
        return;

    chunk->StartObjectIDSequence(Size());
    if (ctx) {
        for (CK_ID *it = Begin(); it != End(); ++it) {
            chunk->WriteObjectSequence(ctx->GetObject(*it));
        }
    } else {
        for (CK_ID *it = Begin(); it != End(); ++it) {
            chunk->WriteObjectIDSequence(*it);
        }
    }
}

void XSObjectArray::Prepare(CKDependenciesContext &context) const {
    for (CK_ID *it = Begin(); it != End(); ++it) {
        CKObject *obj = context.m_CKContext->GetObject(*it);
        if (obj)
            obj->PrepareDependencies(context);
    }
}

void XSObjectArray::Remap(CKDependenciesContext &context) {
    for (CK_ID *it = Begin(); it != End(); ++it) {
        *it = context.RemapID(*it);
    }
}

void XObjectArray::ConvertFromObjects(const XObjectPointerArray &array) {
    Resize(array.Size());
    Resize(0);
    for (auto it = array.Begin(); it != array.End(); ++it) {
        if (it && *it)
            PushBack((*it)->GetID());
    }
}

CKBOOL XObjectArray::Check(CKContext *Context) {
    if (!Context)
        return FALSE;

    const int size = Size();
    const int newSize = Context->m_ObjectManager->CheckIDArray(Begin(), size);
    m_End = &m_Begin[newSize];
    return size != newSize;
}

CKBOOL XObjectArray::AddIfNotHere(CK_ID id) {
    for (CK_ID *it = m_Begin; it != m_End; ++it) {
        if (*it == id)
            return FALSE;
    }
    PushBack(id);
    return TRUE;
}

CKBOOL XObjectArray::AddIfNotHere(CKObject *obj) {
    if (!obj)
        return FALSE;

    CK_ID id = obj->GetID();
    for (CK_ID *it = m_Begin; it != m_End; ++it) {
        if (*it == id)
            return FALSE;
    }
    PushBack(id);
    return TRUE;
}

CKObject *XObjectArray::GetObject(CKContext *Context, unsigned int i) const {
    if (!Context || i >= Size())
        return nullptr;
    return Context->GetObject(m_Begin[i]);
}

CKBOOL XObjectArray::RemoveObject(CKObject *obj) {
    if (!obj)
        return FALSE;
    return Remove(obj->GetID()) != nullptr;
}

CKBOOL XObjectArray::FindObject(CKObject *obj) const {
    CK_ID id = obj ? obj->GetID() : 0;
    for (CK_ID *it = m_Begin; it != m_End; ++it) {
        if (*it == id)
            return TRUE;
    }
    return FALSE;
}

void XObjectArray::Load(CKStateChunk *chunk) {
    if (chunk) {
        *this = chunk->ReadXObjectArray();
    }
}

void XObjectArray::Save(CKStateChunk *chunk, CKContext *ctx) const {
    chunk->StartObjectIDSequence(Size());
    if (ctx) {
        for (CK_ID *it = Begin(); it != End(); ++it) {
            chunk->WriteObjectSequence(ctx->GetObject(*it));
        }
    } else {
        for (CK_ID *it = Begin(); it != End(); ++it) {
            chunk->WriteObjectIDSequence(*it);
        }
    }
}

void XObjectArray::Prepare(CKDependenciesContext &context) const {
    for (CK_ID *it = Begin(); it != End(); ++it) {
        CKObject *obj = context.m_CKContext->GetObject(*it);
        if (obj)
            obj->PrepareDependencies(context);
    }
}

void XObjectArray::Remap(CKDependenciesContext &context) {
    for (CK_ID *it = Begin(); it != End(); ++it) {
        *it = context.RemapID(*it);
    }
}

CKBOOL XSObjectPointerArray::Check() {
    CKObject **it = m_Begin;
    const int size = Size();
    int keptCount = 0;
    int i;
    for (i = 0; i < size; ++i) {
        CKObject *obj = *it++;
        if (obj->IsToBeDeleted())
            break;
        ++keptCount;
    }

    const int firstDeletedIndex = i + 1;
    if (firstDeletedIndex < size) {
        int remaining = size - firstDeletedIndex;
        do {
            if (!(*it)->IsToBeDeleted())
                m_Begin[keptCount++] = *it;
            ++it;
            --remaining;
        } while (remaining);
    }

    Resize(keptCount);
    return keptCount != size;
}

void XSObjectPointerArray::Load(CKContext *Context, CKStateChunk *chunk) {
    if (!Context || !chunk)
        return;

    const int count = chunk->StartReadSequence();
    if (count == 0)
        return;

    if (chunk->GetChunkVersion() >= 4) {
        for (int i = 0; i < count; ++i) {
            CKObject *obj = chunk->ReadObject(Context);
            if (obj)
                PushBack(obj);
        }
    } else {
        chunk->Skip(sizeof(CKDWORD));

        const int objectCount = chunk->ReadInt();
        for (int i = 0; i < objectCount; ++i) {
            CK_ID id = chunk->ReadDword();
            CKObject *obj = Context->GetObject(id);
            if (obj)
                PushBack(obj);
        }
    }
}

void XSObjectPointerArray::Save(CKStateChunk *chunk) const {
    if (!chunk)
        return;

    chunk->StartObjectIDSequence(Size());
    for (CKObject **it = Begin(); it != End(); ++it) {
        chunk->WriteObjectSequence(*it);
    }
}

void XSObjectPointerArray::Prepare(CKDependenciesContext &context) const {
    for (CKObject **it = m_Begin; it != m_End; ++it) {
        CKObject *obj = *it;
        if (obj)
            obj->PrepareDependencies(context);
    }
}

void XSObjectPointerArray::Remap(CKDependenciesContext &context) {
    for (CKObject **it = m_Begin; it != m_End; ++it) {
        if (*it)
            *it = context.Remap(*it);
    }
}
