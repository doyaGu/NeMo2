#include "CKMemoryPool.h"

CKMemoryPool::CKMemoryPool(CKContext *Context, size_t ByteCount) {
    m_Context = Context;
    m_Index = static_cast<size_t>(-1);
    if (Context) {
        m_Memory = m_Context->AllocateMemoryPool(ByteCount, m_Index);
    } else {
        m_Memory = nullptr;
    }
}

CKMemoryPool::~CKMemoryPool() {
    if (m_Context) {
        m_Context->ReleaseMemoryPool(m_Index);
    }
}

void *CKMemoryPool::Mem() const {
    return m_Memory;
}
