#include "RCKLayer.h"

#include "VxColor.h"
#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKContext.h"
#include "CKObject.h"
#include "CKGridManager.h"

/**
 * @brief RCKLayer constructor
 * @param Context The CKContext instance
 * @param name Name for the layer
 * @param owner Owner grid ID
 */
RCKLayer::RCKLayer(CKContext *Context, CKSTRING name, CK_ID owner)
    : CKLayer(Context, name),
      m_Grid(nullptr),
      m_Type(1),
      m_Format(0),
      m_Flags(1),
      m_SquareArray(nullptr) {
    // Get owner grid from context
    m_Grid = reinterpret_cast<CKGrid *>(m_Context->GetObject(owner));

    // Allocate square array if grid exists
    if (m_Grid) {
        const int width = m_Grid->GetWidth();
        const int length = m_Grid->GetLength();
        const int count = width * length;
        m_SquareArray = (count > 0) ? new CKSquare[count] : nullptr;
        if (m_SquareArray) {
            memset(m_SquareArray, 0, static_cast<size_t>(count) * sizeof(CKSquare));
        }
    }
}

/**
 * @brief RCKLayer destructor
 */
RCKLayer::~RCKLayer() {
    // Cleanup square array unconditionally (operator delete handles nullptr)
    delete[] m_SquareArray;
}

/**
 * @brief Get the class ID for RCKLayer
 * @return The class ID (51)
 */
CK_CLASSID RCKLayer::GetClassID() {
    return m_ClassID;
}

//=============================================================================
// CKLayer Virtual Methods
//=============================================================================

void RCKLayer::SetType(int Type) {
    m_Type = static_cast<CKDWORD>(Type);
}

int RCKLayer::GetType() {
    return static_cast<int>(m_Type);
}

void RCKLayer::SetFormat(int Format) {
    m_Format = static_cast<CKDWORD>(Format);
}

int RCKLayer::GetFormat() {
    return static_cast<int>(m_Format);
}

void RCKLayer::SetValue(int x, int y, void *val) {
    // Original directly accesses array without null checks
    const int width = m_Grid->GetWidth();
    const int idx = y * width + x;
    m_SquareArray[idx].ival = *static_cast<CKDWORD *>(val);
}

void RCKLayer::GetValue(int x, int y, void *val) {
    // Original directly accesses array without null checks
    const int width = m_Grid->GetWidth();
    const int idx = y * width + x;
    *static_cast<CKDWORD *>(val) = m_SquareArray[idx].ival;
}

CKBOOL RCKLayer::SetValue2(int x, int y, void *val) {
    const int width = m_Grid->GetWidth();
    const int length = m_Grid->GetLength();
    // Original checks bounds: x >= width || x < 0 || y >= length || y < 0
    if (x >= width || x < 0 || y >= length || y < 0)
        return FALSE;
    m_SquareArray[y * width + x].ival = *static_cast<CKDWORD *>(val);
    return TRUE;
}

CKBOOL RCKLayer::GetValue2(int x, int y, void *val) {
    const int width = m_Grid->GetWidth();
    const int length = m_Grid->GetLength();
    // Original checks bounds: x >= width || x < 0 || y >= length || y < 0
    if (x >= width || x < 0 || y >= length || y < 0) {
        *static_cast<CKDWORD *>(val) = 0;
        return FALSE;
    }
    *static_cast<CKDWORD *>(val) = m_SquareArray[y * width + x].ival;
    return TRUE;
}

CKSquare *RCKLayer::GetSquareArray() {
    return m_SquareArray;
}

void RCKLayer::SetSquareArray(CKSquare *sqarray) {
    // Original simply assigns without deleting old
    m_SquareArray = sqarray;
}

void RCKLayer::SetVisible(CKBOOL vis) {
    if (vis)
        m_Flags |= LAYER_STATE_VISIBLE;
    else
        m_Flags &= ~LAYER_STATE_VISIBLE;
}

CKBOOL RCKLayer::IsVisible() {
    return (m_Flags & LAYER_STATE_VISIBLE) != 0;
}

void RCKLayer::InitOwner(CK_ID owner) {
    // Get owner grid from context
    m_Grid = reinterpret_cast<CKGrid *>(m_Context->GetObject(owner));

    // Delete existing square array
    delete[] m_SquareArray;
    m_SquareArray = nullptr;

    // Allocate new square array if grid exists
    if (m_Grid) {
        const int width = m_Grid->GetWidth();
        const int length = m_Grid->GetLength();
        const int count = width * length;
        m_SquareArray = (count > 0) ? new CKSquare[count] : nullptr;
        if (m_SquareArray) {
            memset(m_SquareArray, 0, static_cast<size_t>(count) * sizeof(CKSquare));
        }
    }
}

void RCKLayer::SetOwner(CK_ID owner) {
    // Original simply sets m_Grid from context lookup without class validation
    m_Grid = reinterpret_cast<CKGrid *>(m_Context->GetObject(owner));
}

CK_ID RCKLayer::GetOwner() {
    return m_Grid ? m_Grid->GetID() : 0;
}

/**
 * @brief Save the layer data to a state chunk
 * @param file The CKFile to save to (can be nullptr for standalone saving)
 * @param flags Save flags
 * @return A new CKStateChunk containing the saved data
 */
CKStateChunk *RCKLayer::Save(CKFile *file, CKDWORD flags) {
    // Call base class Save first
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    // Create a new state chunk for layer data
    CKStateChunk *chunk = CreateCKStateChunk(m_ClassID, file);
    chunk->StartWrite();

    // Add the base class chunk
    chunk->AddChunkAndDelete(baseChunk);

    // Grid manager is used for file metadata (associated color/param).
    // When unavailable, we still write a structurally valid payload.
    CKGridManager *gridMgr = reinterpret_cast<CKGridManager *>(m_Context->GetManagerByGuid(GRID_MANAGER_GUID));

    // Only write layer payload if saving to file or flag 0x10 is set
    if (file || (flags & 0x10u) != 0) {
        chunk->WriteIdentifier(0x10u);
        chunk->WriteObject(reinterpret_cast<CKObject *>(m_Grid));

        // When saving to memory, write the raw type index
        if (!file) {
            chunk->WriteInt(static_cast<int>(m_Type));
        }

        chunk->WriteInt(static_cast<int>(m_Format));

        // When saving to a file, write version and metadata
        if (file) {
            chunk->WriteInt(3); // version

            VxColor color;
            color.Set(0.0f, 0.0f, 0.0f, 0.0f);
            if (gridMgr)
                gridMgr->GetAssociatedColor(m_Type, &color);
            chunk->WriteDword(color.GetRGBA());

            CKGUID paramGuid = CKGUID(0, 0);
            if (gridMgr)
                paramGuid = gridMgr->GetAssociatedParam(m_Type);
            chunk->WriteGuid(paramGuid);
        }

        chunk->WriteInt(static_cast<int>(m_Flags));

        // Write square array data if format is 0 and grid exists
        if (m_Format == 0 && m_Grid) {
            const int gridWidth = m_Grid->GetWidth();
            const int gridLength = m_Grid->GetLength();
            const int bufferSize = 4 * gridLength * gridWidth;
            chunk->WriteBuffer_LEndian(bufferSize, m_SquareArray);
        }

        // Mark type as used in file
        if (file) {
            // ManagerByGuid[11] + 4 * m_Type check from original
            // This updates tracking in the grid manager
        }
    }

    chunk->CloseChunk();
    return chunk;
}

/**
 * @brief Load layer data from a state chunk
 * @param chunk The CKStateChunk containing the saved data
 * @param file The CKFile being loaded from
 * @return CKERROR indicating success or failure
 */
CKERROR RCKLayer::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Call base class Load first
    CKObject::Load(chunk, file);

    // Load layer data if identifier is found
    if (chunk->SeekIdentifier(0x10u)) {
        CKGridManager *gridMgr = reinterpret_cast<CKGridManager *>(m_Context->GetManagerByGuid(GRID_MANAGER_GUID));
        if (!gridMgr)
            return CKERR_NOTINITIALIZED;

        // Determine layer type from this object's name (Virtools convention)
        const CKSTRING layerName = GetName();
        m_Type = gridMgr->GetTypeFromName(layerName);
        if (m_Type == 0)
            m_Type = gridMgr->RegisterType(layerName);

        // Load grid reference
        m_Grid = reinterpret_cast<CKGrid *>(chunk->ReadObject(m_Context));

        if (!file) {
            // Skip type index in runtime chunks (type is re-derived from layer name)
            chunk->ReadInt();
        }

        // Load format
        m_Format = chunk->ReadInt();

        // Handle file-specific loading
        if (file) {
            const int version = chunk->ReadInt();
            if (version >= 1) {
                const CKDWORD colorValue = chunk->ReadDword();
                VxColor layerColor(colorValue);
                gridMgr->SetAssociatedColor(static_cast<int>(m_Type), &layerColor);

                if (version < 2) {
                    m_Flags = 1; // LAYER_STATE_VISIBLE
                } else {
                    if (version < 3) {
                        // Historical default associated param GUID for older files
                        gridMgr->SetAssociatedParam(static_cast<int>(m_Type), CKGUID(0x5A6B0AFD, 0x44EB9DD7));
                    } else {
                        const CKGUID paramGuid = chunk->ReadGuid();
                        gridMgr->SetAssociatedParam(static_cast<int>(m_Type), paramGuid);
                    }
                    m_Flags = static_cast<CKDWORD>(chunk->ReadInt());
                }
            }
        } else {
            // Load flags directly for non-file context
            m_Flags = static_cast<CKDWORD>(chunk->ReadInt());
        }

        // Cleanup existing square array
        delete[] m_SquareArray;
        m_SquareArray = nullptr;

        // Load square array data if format is 0
        if (!m_Format) {
            void *raw = nullptr;
            const int bufferSize = chunk->ReadBuffer(&raw);
            if (raw && bufferSize > 0 && m_Grid) {
                const int width = m_Grid->GetWidth();
                const int length = m_Grid->GetLength();
                const int count = width * length;
                const int expectedBytes = count * static_cast<int>(sizeof(CKSquare));

                m_SquareArray = (count > 0) ? new CKSquare[count] : nullptr;
                if (m_SquareArray) {
                    memset(m_SquareArray, 0, static_cast<size_t>(count) * sizeof(CKSquare));
                    const int copyBytes = (bufferSize < expectedBytes) ? bufferSize : expectedBytes;
                    memcpy(m_SquareArray, raw, static_cast<size_t>(copyBytes));
                    CKConvertEndianArray32(reinterpret_cast<CKDWORD *>(m_SquareArray), expectedBytes >> 2);
                }
            }
            if (raw)
                CKDeletePointer(raw);
        }
    }

    return CK_OK;
}

CKERROR RCKLayer::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::PrepareDependencies(context);
    if (err != CK_OK)
        return err;

    if (m_Grid)
        m_Grid->PrepareDependencies(context);

    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR RCKLayer::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    m_Grid = static_cast<CKGrid *>(context.Remap(reinterpret_cast<CKObject *>(m_Grid)));
    return CK_OK;
}

/**
 * @brief Get memory occupation for the layer
 * @return Memory size in bytes
 */
int RCKLayer::GetMemoryOccupation() {
    return CKObject::GetMemoryOccupation() + (sizeof(RCKLayer) - sizeof(CKObject));
}

/**
 * @brief Copy layer data to another object
 * @param o The target object
 * @param context Dependencies context
 * @return CKERROR indicating success or failure
 */
CKERROR RCKLayer::Copy(CKObject &o, CKDependenciesContext &context) {
    // In Virtools, Copy is called on the destination object, with 'o' being the source.
    CKERROR err = CKObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    CKDWORD classDeps = context.GetClassDependencies(CKCID_LAYER);

    RCKLayer *src = static_cast<RCKLayer *>(&o);

    m_Grid = src->m_Grid;
    m_Type = src->m_Type;
    m_Format = src->m_Format;
    m_Flags = src->m_Flags;
    m_SquareArray = nullptr;

    // Original always allocates based on this->m_Grid after assignment
    if (m_Grid) {
        const int width = m_Grid->GetWidth();
        const int length = m_Grid->GetLength();
        const int count = width * length;
        m_SquareArray = new CKSquare[count];

        // Copy data if classDeps & 1, otherwise zero
        if (classDeps & 1)
            memcpy(m_SquareArray, src->m_SquareArray, static_cast<size_t>(count) * sizeof(CKSquare));
        else
            memset(m_SquareArray, 0, static_cast<size_t>(count) * sizeof(CKSquare));
    }

    return CK_OK;
}

// Static class registration methods
CKSTRING RCKLayer::GetClassName() {
    return (CKSTRING) "Layer";
}

int RCKLayer::GetDependenciesCount(int mode) {
    return 0; // No dependencies
}

CKSTRING RCKLayer::GetDependencies(int i, int mode) {
    return nullptr;
}

void RCKLayer::Register() {
    // Based on IDA decompilation
    CKCLASSNOTIFYFROMCID(RCKLayer, CKCID_GRID);
    CKCLASSDEFAULTOPTIONS(RCKLayer, CK_GENERALOPTIONS_NODUPLICATENAMECHECK);
    CKCLASSDEFAULTCOPYDEPENDENCIES(RCKLayer, CK_DEPENDENCIES_COPY);
}

CKLayer *RCKLayer::CreateInstance(CKContext *Context) {
    return reinterpret_cast<CKLayer *>(new RCKLayer(Context, nullptr, 0));
}

// Static class ID
CK_CLASSID RCKLayer::m_ClassID = CKCID_LAYER;
