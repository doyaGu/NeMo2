#include "CKDataArray.h"

#include "CKFile.h"
#include "CKGroup.h"
#include "CKParameter.h"
#include "CKParameterOut.h"
#include "CKParameterManager.h"

template <class T>
CKBOOL OpCompare(CK_COMPOPERATOR op, T a, T b) {
    switch (op) {
    case CKEQUAL:       return a == b;
    case CKNOTEQUAL:    return a != b;
    case CKLESSER:      return a < b;
    case CKLESSEREQUAL: return a <= b;
    case CKGREATER:     return a > b;
    case CKGREATEREQUAL:return a >= b;
    default:            return FALSE;
    }
}

int DataRowCompare(const void *elem1, const void *elem2) {
    CKDataRow *row1 = *(CKDataRow **) elem1;
    CKDataRow *row2 = *(CKDataRow **) elem2;
    return CKDataArray::g_SortFunction(row1, row2);
}

int ArrayIntComp(CKDataRow *row1, CKDataRow *row2) {
    int val1 = (*row1)[CKDataArray::g_ColumnIndex];
    int val2 = (*row2)[CKDataArray::g_ColumnIndex];
    if (CKDataArray::g_Order) return val1 - val2;
    return val2 - val1;
}

int ArrayFloatComp(CKDataRow *row1, CKDataRow *row2) {
    float val1 = *(float *) &((*row1)[CKDataArray::g_ColumnIndex]);
    float val2 = *(float *) &((*row2)[CKDataArray::g_ColumnIndex]);
    if (val1 < val2) return CKDataArray::g_Order ? -1 : 1;
    if (val1 > val2) return CKDataArray::g_Order ? 1 : -1;
    return 0;
}

int ArrayStringComp(CKDataRow *row1, CKDataRow *row2) {
    const char *s1 = (const char *) (*row1)[CKDataArray::g_ColumnIndex];
    const char *s2 = (const char *) (*row2)[CKDataArray::g_ColumnIndex];
    if (!s1) s1 = "";
    if (!s2) s2 = "";
    int diff = strcmp(s1, s2);
    return CKDataArray::g_Order ? diff : -diff;
}

int ArrayParameterComp(CKDataRow *row1, CKDataRow *row2) {
    CKParameter *p1 = (CKParameter *) (*row1)[CKDataArray::g_ColumnIndex];
    CKParameter *p2 = (CKParameter *) (*row2)[CKDataArray::g_ColumnIndex];
    if (p1 == p2) return 0;
    if (!p1 || !p2) return p1 ? 1 : -1;

    int size = XMin(p1->GetDataSize(), p2->GetDataSize());
    if (size == 0) return 0;

    int result = memcmp(p1->GetReadDataPtr(), p2->GetReadDataPtr(), size);
    return CKDataArray::g_Order ? result : -result;
}

CKBOOL ArrayIntEqual(CKDataRow *row) {
    int v1 = (*row)[CKDataArray::g_ColumnIndex];
    int v2 = (int) CKDataArray::g_Value;
    return OpCompare(CKDataArray::g_Operator, v1, v2);
}

CKBOOL ArrayFloatEqual(CKDataRow *row) {
    float v1 = *(float *) &(*row)[CKDataArray::g_ColumnIndex];
    float v2 = *(float *) &CKDataArray::g_Value;
    return OpCompare(CKDataArray::g_Operator, v1, v2);
}

CKBOOL ArrayStringEqual(CKDataRow *row) {
    const char *s1 = (const char *) (*row)[CKDataArray::g_ColumnIndex];
    const char *s2 = (const char *) CKDataArray::g_Value;
    if (!s1) s1 = "";
    if (!s2) s2 = "";
    return OpCompare(CKDataArray::g_Operator, strcmp(s1, s2), 0);
}

CKBOOL ArrayParameterEqual(CKDataRow *row) {
    CKParameter *p = (CKParameter *) (*row)[CKDataArray::g_ColumnIndex];
    if (!p) return FALSE;
    const int size = XMin(p->GetDataSize(), (int) CKDataArray::g_ValueSize);
    if (size == 0) return FALSE;
    const void *ptr = p->GetReadDataPtr();
    return OpCompare(CKDataArray::g_Operator, memcmp(ptr, (void *) CKDataArray::g_Value, size), 0);
}

CK_CLASSID CKDataArray::m_ClassID = CKCID_DATAARRAY;

int CKDataArray::g_ColumnIndex = 0;
CKBOOL CKDataArray::g_Order = TRUE;
CK_COMPOPERATOR CKDataArray::g_Operator = CKEQUAL;
CKDWORD CKDataArray::g_Value = 0;
CKDWORD CKDataArray::g_ValueSize = 0;
ArraySortFunction CKDataArray::g_SortFunction = nullptr;

void CKDataArray::InsertColumn(int cdest, CK_ARRAYTYPE type, char *name, CKGUID paramGuid) {
    if (!name) return;

    ColumnFormat *fmt = new ColumnFormat();
    fmt->m_Name = CKStrdup(name);
    fmt->m_Type = type;

    switch (type) {
    case CKARRAYTYPE_INT:
        fmt->m_SortFunction = ArrayIntComp;
        fmt->m_EqualFunction = ArrayIntEqual;
        break;
    case CKARRAYTYPE_FLOAT:
        fmt->m_SortFunction = ArrayFloatComp;
        fmt->m_EqualFunction = ArrayFloatEqual;
        break;
    case CKARRAYTYPE_STRING:
        fmt->m_SortFunction = ArrayStringComp;
        fmt->m_EqualFunction = ArrayStringEqual;
        break;
    case CKARRAYTYPE_OBJECT:
        fmt->m_SortFunction = ArrayIntComp;
        fmt->m_EqualFunction = ArrayIntEqual;
        break;
    case CKARRAYTYPE_PARAMETER:
        fmt->m_SortFunction = ArrayParameterComp;
        fmt->m_EqualFunction = ArrayParameterEqual;
        fmt->m_ParameterType = paramGuid;
        break;
    }

    if (cdest == -1 || cdest >= m_FormatArray.Size()) {
        m_FormatArray.PushBack(fmt);
    } else {
        m_FormatArray.Insert(cdest, fmt);
    }

    for (int i = 0; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *row = m_DataMatrix[i];

        if (type == CKARRAYTYPE_PARAMETER) {
            CKParameterOut *param = m_Context->CreateCKParameterOut(nullptr, fmt->m_ParameterType, IsDynamic());
            if (param) {
                param->SetOwner(this);
            }

            if (cdest == -1 || cdest >= row->Size()) {
                row->PushBack((CKDWORD) param);
            } else {
                row->Insert(cdest, (CKDWORD) param);
            }
        } else {
            if (cdest == -1 || cdest >= row->Size()) {
                row->PushBack(0);
            } else {
                row->Insert(cdest, 0);
            }
        }
    }
}

void CKDataArray::MoveColumn(int csrc, int cdest) {
    const int colCount = m_FormatArray.Size();

    if (csrc < 0 || csrc >= colCount)
        return;
    if (cdest < -1 || cdest > colCount)
        return;

    if (cdest == -1)
        cdest = colCount;

    if (csrc == cdest)
        return;

    ColumnFormat **srcFormat = m_FormatArray.Begin() + csrc;
    ColumnFormat **dstFormat = m_FormatArray.Begin() + cdest;

    m_FormatArray.Move(dstFormat, srcFormat);

    for (int i = 0; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *row = m_DataMatrix[i];
        CKDWORD *srcData = row->Begin() + csrc;
        CKDWORD *dstData = row->Begin() + cdest;
        row->Move(dstData, srcData);
    }
}

void CKDataArray::RemoveColumn(int c) {
    if (c < 0 || c >= m_FormatArray.Size()) return;

    // Update sorting state if needed
    if (m_Order) {
        if (m_ColumnIndex == c) {
            m_Order = FALSE; // Invalidate current sort
        } else if (m_ColumnIndex > c) {
            m_ColumnIndex--;
        }
    }

    // Update key column index if needed
    if (m_KeyColumn > c) {
        m_KeyColumn--;
    }

    // Remove column format
    ColumnFormat *format = m_FormatArray[c];
    CK_ARRAYTYPE colType = format->m_Type;

    delete[] format->m_Name;
    delete format;
    m_FormatArray.RemoveAt(c);

    // Remove column data from all rows
    for (int i = 0; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *row = m_DataMatrix[i];
        CKDWORD *element = &(*row)[c];

        switch (colType) {
        case CKARRAYTYPE_STRING:
            delete[] (char *) *element;
            break;
        case CKARRAYTYPE_PARAMETER: {
            CKParameter *param = (CKParameter *) *element;
            if (param && param->GetOwner() == this) {
                m_Context->DestroyObject(param);
            }
            break;
        }
        default:
            break;
        }

        row->RemoveAt(c);
    }
}

void CKDataArray::SetColumnName(int c, char *name) {
    if (c < 0 || c >= m_FormatArray.Size() || !name) return;
    ColumnFormat *fmt = m_FormatArray[c];
    if (!fmt) return;
    delete[] fmt->m_Name;
    fmt->m_Name = CKStrdup(name);
}

char *CKDataArray::GetColumnName(int c) {
    if (c < 0 || c >= m_FormatArray.Size()) return nullptr;
    ColumnFormat *fmt = m_FormatArray[c];
    return fmt ? fmt->m_Name : nullptr;
}

void CKDataArray::SetColumnType(int c, CK_ARRAYTYPE newType, CKGUID paramGuid) {
    if (c < 0 || c >= m_FormatArray.Size()) return;

    ColumnFormat *format = m_FormatArray[c];
    CK_ARRAYTYPE oldType = format->m_Type;
    if (oldType == newType && (oldType != CKARRAYTYPE_PARAMETER || format->m_ParameterType == paramGuid)) return;

    // Update column format
    format->m_Type = newType;
    format->m_ParameterType = paramGuid;

    // Set comparison functions
    switch (newType) {
    case CKARRAYTYPE_INT:
        format->m_SortFunction = ArrayIntComp;
        format->m_EqualFunction = ArrayIntEqual;
        break;
    case CKARRAYTYPE_FLOAT:
        format->m_SortFunction = ArrayFloatComp;
        format->m_EqualFunction = ArrayFloatEqual;
        break;
    case CKARRAYTYPE_STRING:
        format->m_SortFunction = ArrayStringComp;
        format->m_EqualFunction = ArrayStringEqual;
        break;
    case CKARRAYTYPE_OBJECT:
        format->m_SortFunction = ArrayIntComp;
        format->m_EqualFunction = ArrayIntEqual;
        break;
    case CKARRAYTYPE_PARAMETER:
        format->m_SortFunction = ArrayParameterComp;
        format->m_EqualFunction = ArrayParameterEqual;
        break;
    }

    // Convert existing data
    for (int i = 0; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *row = m_DataMatrix[i];
        CKDWORD &element = (*row)[c];

        switch (oldType) {
        case CKARRAYTYPE_INT: {
            int intValue = static_cast<int>(element);
            switch (newType) {
            case CKARRAYTYPE_FLOAT:
                element = static_cast<CKDWORD>(static_cast<float>(intValue));
                break;
            case CKARRAYTYPE_STRING: {
                char buf[32];
                sprintf(buf, "%d", intValue);
                element = reinterpret_cast<CKDWORD>(CKStrdup(buf));
                break;
            }
            case CKARRAYTYPE_PARAMETER: {
                CKParameterOut *param = m_Context->CreateCKParameterOut(nullptr, paramGuid, IsDynamic());
                if (param) {
                    param->SetOwner(this);
                    if (paramGuid == CKPGUID_INT) {
                        param->SetValue(&intValue, sizeof(int));
                    }
                    element = reinterpret_cast<CKDWORD>(param);
                } else {
                    element = 0;
                }
                break;
            }
            default:
                element = 0;
                break;
            }
            break;
        }

        case CKARRAYTYPE_FLOAT: {
            float floatValue = reinterpret_cast<float &>(element);
            switch (newType) {
            case CKARRAYTYPE_INT:
                element = static_cast<int>(floatValue);
                break;
            case CKARRAYTYPE_STRING: {
                char buf[32];
                sprintf(buf, "%.4f", floatValue);
                element = reinterpret_cast<CKDWORD>(CKStrdup(buf));
                break;
            }
            case CKARRAYTYPE_PARAMETER: {
                CKParameterOut *param = m_Context->CreateCKParameterOut(nullptr, paramGuid, IsDynamic());
                if (param) {
                    param->SetOwner(this);
                    if (paramGuid == CKPGUID_FLOAT) {
                        param->SetValue(&floatValue, sizeof(float));
                    }
                    element = reinterpret_cast<CKDWORD>(param);
                } else {
                    element = 0;
                }
                break;
            }
            default:
                element = 0;
                break;
            }
            break;
        }

        case CKARRAYTYPE_STRING: {
            char *strValue = reinterpret_cast<char *>(element);
            switch (newType) {
            case CKARRAYTYPE_INT: {
                int val = strValue ? atoi(strValue) : 0;
                delete[] strValue;
                element = val;
                break;
            }
            case CKARRAYTYPE_FLOAT: {
                float val = strValue ? static_cast<float>(atof(strValue)) : 0.0f;
                delete[] strValue;
                element = reinterpret_cast<CKDWORD &>(val);
                break;
            }
            case CKARRAYTYPE_PARAMETER: {
                CKParameterOut *param = m_Context->CreateCKParameterOut(nullptr, paramGuid, IsDynamic());
                if (param) {
                    param->SetOwner(this);
                    param->SetStringValue(strValue);
                    element = reinterpret_cast<CKDWORD>(param);
                } else {
                    element = 0;
                }
                delete[] strValue;
                break;
            }
            default:
                delete[] strValue;
                element = 0;
                break;
            }
            break;
        }

        case CKARRAYTYPE_OBJECT: {
            switch (newType) {
            case CKARRAYTYPE_INT:
            case CKARRAYTYPE_FLOAT:
                element = 0;
                break;
            case CKARRAYTYPE_PARAMETER: {
                CKParameterOut *param = m_Context->CreateCKParameterOut(nullptr, paramGuid, IsDynamic());
                if (param) {
                    param->SetOwner(this);
                    element = reinterpret_cast<CKDWORD>(param);
                } else {
                    element = 0;
                }
                break;
            }
            default:
                element = 0;
                break;
            }
            break;
        }

        case CKARRAYTYPE_PARAMETER: {
            CKParameter *oldParam = reinterpret_cast<CKParameter *>(element);
            CKParameter *ownedParam = (oldParam && oldParam->GetOwner() == this) ? oldParam : nullptr;

            switch (newType) {
            case CKARRAYTYPE_INT: {
                int val = 0;
                if (oldParam && oldParam->GetGUID() == CKPGUID_INT) {
                    oldParam->GetValue(&val, sizeof(int));
                }
                element = val;
                break;
            }
            case CKARRAYTYPE_FLOAT: {
                float val = 0.0f;
                if (oldParam && oldParam->GetGUID() == CKPGUID_FLOAT) {
                    oldParam->GetValue(&val, sizeof(float));
                }
                element = reinterpret_cast<CKDWORD &>(val);
                break;
            }
            case CKARRAYTYPE_STRING: {
                char *str = nullptr;
                if (oldParam && oldParam->GetGUID() == CKPGUID_STRING) {
                    str = CKStrdup(static_cast<char *>(oldParam->GetReadDataPtr()));
                } else {
                    str = CKStrdup(const_cast<char *>(""));
                }
                element = reinterpret_cast<CKDWORD>(str);
                break;
            }
            default:
                element = 0;
                break;
            }

            if (ownedParam) {
                m_Context->DestroyObject(ownedParam);
            }
            break;
        }
        }
    }
}

CK_ARRAYTYPE CKDataArray::GetColumnType(int c) {
    if (c < 0 || c >= m_FormatArray.Size()) return (CK_ARRAYTYPE) 0;
    ColumnFormat *fmt = m_FormatArray[c];
    return fmt ? fmt->m_Type : (CK_ARRAYTYPE) 0;
}

CKGUID CKDataArray::GetColumnParameterGuid(int c) {
    if (c < 0 || c >= m_FormatArray.Size()) return CKGUID();
    ColumnFormat *fmt = m_FormatArray[c];
    return fmt ? fmt->m_ParameterType : CKGUID();
}

int CKDataArray::GetKeyColumn() {
    return m_KeyColumn;
}

void CKDataArray::SetKeyColumn(int c) {
    if (c >= -1 && c < m_FormatArray.Size())
        m_KeyColumn = c;
}

int CKDataArray::GetColumnCount() {
    return m_FormatArray.Size();
}

CKDWORD *CKDataArray::GetElement(int i, int c) {
    if (i < 0 || i >= m_DataMatrix.Size())
        return nullptr;
    CKDataRow *row = m_DataMatrix[i];
    if (!row)
        return nullptr;
    if (c < 0 || c >= row->Size())
        return nullptr;
    return &(*row)[c];
}

CKBOOL CKDataArray::GetElementValue(int i, int c, void *value) {
    CKDWORD *element = GetElement(i, c);
    if (!element) return FALSE;
    if (GetColumnType(c) == CKARRAYTYPE_PARAMETER) {
        CKParameter *param = (CKParameter *) *element;
        param->GetValue(value);
    } else {
        *(CKDWORD *)value = *element;
    }
    return TRUE;
}

CKObject *CKDataArray::GetElementObject(int i, int c) {
    CKDWORD *element = GetElement(i, c);
    if (!element) return nullptr;
    return m_Context->GetObject(*element);
}

CKBOOL CKDataArray::SetElementValue(int i, int c, void *value, int size) {
    if (i < 0 || i >= m_DataMatrix.Size() || c < 0 || c >= m_FormatArray.Size())
        return FALSE;

    if (m_Order && c == m_ColumnIndex)
        m_Order = FALSE;

    CKDataRow *dataRow = m_DataMatrix[i];
    ColumnFormat *format = m_FormatArray[c];

    switch (format->m_Type) {
    case CKARRAYTYPE_STRING: {
        CKDWORD *element = &(*dataRow)[c];
        char *oldStr = reinterpret_cast<char *>(*element);
        delete[] oldStr;

        if (value) {
            *element = reinterpret_cast<CKDWORD>(CKStrdup(static_cast<char *>(value)));
        } else {
            *element = 0;
        }
        break;
    }

    case CKARRAYTYPE_PARAMETER: {
        CKParameterOut *param = reinterpret_cast<CKParameterOut *>((*dataRow)[c]);
        if (param) {
            param->SetValue(value, size);
        }
        break;
    }

    default: {
        // CKARRAYTYPE_INT, CKARRAYTYPE_FLOAT, CKARRAYTYPE_OBJECT
        CKDWORD *element = &(*dataRow)[c];
        *element = *static_cast<CKDWORD *>(value);
        break;
    }
    }

    return TRUE;
}

CKBOOL CKDataArray::SetElementValueFromParameter(int i, int c, CKParameter *pout) {
    if (!pout || i < 0 || i >= m_DataMatrix.Size() || c < 0 || c >= m_FormatArray.Size())
        return FALSE;

    if (m_Order && c == m_ColumnIndex)
        m_Order = FALSE;

    CKDataRow *dataRow = m_DataMatrix[i];
    ColumnFormat *format = m_FormatArray[c];
    CKDWORD &element = (*dataRow)[c];

    switch (format->m_Type) {
    case CKARRAYTYPE_STRING: {
        int strLen = pout->GetStringValue(nullptr, 0);
        if (strLen > 0) {
            char *newStr = new char[strLen];
            pout->GetStringValue(newStr, strLen);

            delete[] reinterpret_cast<char *>(element);
            element = reinterpret_cast<CKDWORD>(newStr);
        } else {
            delete[] reinterpret_cast<char *>(element);
            element = 0;
        }
        break;
    }

    case CKARRAYTYPE_PARAMETER: {
        CKParameterOut *existingParam = reinterpret_cast<CKParameterOut *>(element);
        if (existingParam) {
            existingParam->CopyValue(pout, TRUE);
        }
        break;
    }

    default: {
        const void *srcData = pout->GetReadDataPtr();
        if (srcData) {
            memcpy(&element, srcData, sizeof(CKDWORD));
        }
        break;
    }
    }

    return TRUE;
}

CKBOOL CKDataArray::SetElementObject(int i, int c, CKObject *object) {
    CK_ID id = object ? object->GetID() : 0;
    return SetElementValue(i, c, &id);
}

CKBOOL CKDataArray::PasteShortcut(int i, int c, CKParameter *pout) {
    if (i < 0 || i >= m_DataMatrix.Size() || c < 0 || c >= m_FormatArray.Size() || !pout)
        return FALSE;

    if (pout->GetOwner() == this)
        return FALSE;

    CKGUID columnGuid = GetColumnParameterGuid(c);
    CKGUID paramGuid = pout->GetGUID();

    CKParameterManager *pm = m_Context->GetParameterManager();
    if (!pm->IsDerivedFrom(paramGuid, columnGuid))
        return FALSE;

    CKDataRow *dataRow = m_DataMatrix[i];
    CKDWORD &element = (*dataRow)[c];

    CKParameter *param = reinterpret_cast<CKParameter *>(element);
    if (param && param->GetOwner() == this) {
        m_Context->DestroyObject(param);
    }

    element = reinterpret_cast<CKDWORD>(pout);
    return TRUE;
}

CKParameterOut *CKDataArray::RemoveShortcut(int i, int c) {
    CKDWORD *element = GetElement(i, c);
    if (!element)
        return nullptr;

    CKParameterOut *originalParam = reinterpret_cast<CKParameterOut *>(*element);

    if (!originalParam || originalParam->GetOwner() == this)
        return nullptr;

    CKParameterOut *newParam = (CKParameterOut *) m_Context->CopyObject(originalParam);
    if (newParam) {
        newParam->SetOwner(this);
        *element = reinterpret_cast<CKDWORD>(newParam);
    }

    return originalParam;
}

CKBOOL CKDataArray::SetElementStringValue(int i, int c, char *svalue) {
    if (!svalue || i < 0 || i >= m_DataMatrix.Size() || c < 0 || c >= m_FormatArray.Size())
        return FALSE;

    CKDataRow *dataRow = m_DataMatrix[i];
    ColumnFormat *format = m_FormatArray[c];
    CKDWORD &element = (*dataRow)[c];
    const CKBOOL isKeyColumn = (c == m_KeyColumn);

    switch (format->m_Type) {
    case CKARRAYTYPE_INT: {
        int intValue = atoi(svalue);

        if (isKeyColumn) {
            // Check for duplicate key
            CKDataRow *foundRow = FindRow(c, CKEQUAL, intValue);
            if (foundRow && foundRow != dataRow) {
                return FALSE;
            }
        }

        element = (CKDWORD) intValue;
        return TRUE;
    }

    case CKARRAYTYPE_FLOAT: {
        float floatValue = (float) atof(svalue);

        if (isKeyColumn) {
            // Check for duplicate key
            CKDataRow *foundRow = FindRow(c, CKEQUAL, (CKDWORD) floatValue);
            if (foundRow && foundRow != dataRow) {
                return FALSE;
            }
        }

        element = (CKDWORD) floatValue;
        return TRUE;
    }

    case CKARRAYTYPE_STRING: {
        char *oldString = (char *) element;
        delete[] oldString;

        if (isKeyColumn) {
            // Check for duplicate key
            CKDataRow *foundRow = FindRow(c, CKEQUAL, (CKDWORD) svalue);
            if (foundRow && foundRow != dataRow) {
                element = 0;
                return FALSE;
            }
        }

        element = (CKDWORD) CKStrdup(svalue);
        return TRUE;
    }

    case CKARRAYTYPE_OBJECT: {
        // Lookup object by name
        CKObject *obj = m_Context->GetObjectByName(svalue);
        element = obj ? obj->GetID() : 0;
        return TRUE;
    }

    case CKARRAYTYPE_PARAMETER: {
        // Update parameter value
        CKParameterOut *param = (CKParameterOut *) element;
        if (param) {
            param->SetStringValue(svalue);
        }
        return TRUE;
    }

    default:
        return FALSE;
    }
}

int CKDataArray::GetStringValue(CKDWORD key, int c, char *svalue) {
    if (c < 0 || c >= m_FormatArray.Size()) return 0;

    const ColumnFormat *format = m_FormatArray[c];
    char buffer[256];
    const char *result = nullptr;

    switch (format->m_Type) {
    case CKARRAYTYPE_INT:
        sprintf(buffer, "%d", (int) key);
        result = buffer;
        break;

    case CKARRAYTYPE_FLOAT:
        sprintf(buffer, "%.4f", (float) key);
        result = buffer;
        break;

    case CKARRAYTYPE_STRING:
        if (key) {
            result = (const char *) key;
        }
        break;

    case CKARRAYTYPE_OBJECT: {
        CKObject *obj = m_Context->GetObject(key);
        if (obj && obj->m_Name) {
            result = obj->m_Name;
        }
        break;
    }

    case CKARRAYTYPE_PARAMETER: {
        int length = 0;
        if (!key) {
            result = "Error : Parameter Deleted";
            break;
        }

        CKParameter *param = (CKParameter *) key;
        if (param->GetOwner() == this) {
            length = param->GetStringValue(svalue, 0);
            if (svalue) {
                param->GetStringValue(svalue, length);
            }
            return length + 1;
        } else {
            length = param->GetStringValue(nullptr, 0);
            if (length > 0) {
                char *temp = new char[length];
                param->GetStringValue(temp, length);

                if (svalue) {
                    sprintf(svalue, "(%s)", temp);
                }
                delete[] temp;
                return length + 2; // Account for parentheses
            } else {
                result = "(N/A)";
            }
        }
        break;
    }

    default:
        return 0;
    }

    if (!result)
        return 0;

    if (svalue) {
        strcpy(svalue, result);
    }

    return strlen(result) + 1;
}

int CKDataArray::GetElementStringValue(int i, int c, char *svalue) {
    CKDWORD *element = GetElement(i, c);
    if (!element) return 0;
    return GetStringValue(*element, c, svalue);
}

CKBOOL CKDataArray::LoadElements(CKSTRING filename, CKBOOL append, int column) {
    if (!filename)
        return FALSE;

    FILE *file = fopen(filename, "rb");
    if (!file)
        return FALSE;

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = new char[fileSize + 1];
    fread(buffer, 1, fileSize, file);
    buffer[fileSize] = '\0';
    fclose(file);

    char *current = buffer;
    int lineNum = 0;
    CKDataRow *row = nullptr;

    while (*current) {
        char *lineEnd = strchr(current, '\n');
        if (lineEnd) *lineEnd = '\0';

        if (!append || lineNum >= m_DataMatrix.Size()) {
            row = InsertRow(-1);
            if (!row)
                break;
        } else {
            row = GetRow(lineNum);
        }

        char *token = strtok(current, "\t");
        int col = column;

        while (token && col < m_FormatArray.Size()) {
            ColumnFormat *fmt = m_FormatArray[col];
            CKDWORD *element = &(*row)[col];

            switch (fmt->m_Type) {
            case CKARRAYTYPE_INT:
                *element = atoi(token);
                break;

            case CKARRAYTYPE_FLOAT:
                *reinterpret_cast<float *>(element) = static_cast<float>(atof(token));
                break;

            case CKARRAYTYPE_STRING: {
                char *newStr = CKStrdup(token);
                delete[] reinterpret_cast<char *>(*element);
                *element = reinterpret_cast<CKDWORD>(newStr);
                break;
            }

            case CKARRAYTYPE_OBJECT: {
                CKObject *obj = m_Context->GetObjectByName(token);
                *element = obj ? obj->m_ID : 0;
                break;
            }

            case CKARRAYTYPE_PARAMETER: {
                CKParameterOut *param = reinterpret_cast<CKParameterOut *>(*element);
                if (!param) {
                    param = m_Context->CreateCKParameterOut(nullptr, fmt->m_ParameterType, IsDynamic());
                    *element = reinterpret_cast<CKDWORD>(param);
                }
                if (param) {
                    param->SetStringValue(token);
                }
                break;
            }
            }

            token = strtok(nullptr, "\t");
            col++;
        }

        current = lineEnd ? lineEnd + 1 : current + strlen(current);
        lineNum++;
    }

    delete[] buffer;
    return TRUE;
}

CKBOOL CKDataArray::WriteElements(CKSTRING filename, int column, int number, CKBOOL append) {
    if (!filename || column < 0 || number <= 0)
        return FALSE;

    FILE *file = fopen(filename, append ? "at" : "wt");
    if (!file)
        return FALSE;

    for (CKDataMatrix::Iterator it = m_DataMatrix.Begin(); it != m_DataMatrix.End(); ++it) {
        CKDataRow *row = *it;
        if (!row)
            continue;

        for (int c = column; c < column + number; ++c) {
            if (c > column)
                fprintf(file, "\t");

            if (c >= m_FormatArray.Size()) {
                fprintf(file, "INVALID_COL");
                continue;
            }

            CKDWORD element = (*row)[c];
            ColumnFormat *fmt = m_FormatArray[c];

            switch (fmt->m_Type) {
            case CKARRAYTYPE_INT:
                fprintf(file, "%d", static_cast<int>(element));
                break;

            case CKARRAYTYPE_FLOAT:
                fprintf(file, "%g", reinterpret_cast<const float &>(element));
                break;

            case CKARRAYTYPE_STRING: {
                const char *str = reinterpret_cast<const char *>(element);
                fprintf(file, "%s", str ? str : "NULL");
                break;
            }

            case CKARRAYTYPE_OBJECT: {
                CKObject *obj = m_Context->GetObject(element);
                fprintf(file, "%s", obj ? obj->GetName() : "NULL_OBJ");
                break;
            }

            case CKARRAYTYPE_PARAMETER: {
                CKParameter *param = reinterpret_cast<CKParameter *>(element);
                if (param) {
                    char buffer[256];
                    param->GetStringValue(buffer, sizeof(buffer));
                    fprintf(file, "%s", buffer);
                } else {
                    fprintf(file, "NULL_PARAM");
                }
                break;
            }

            default:
                fprintf(file, "UNKNOWN_TYPE");
                break;
            }
        }

        fprintf(file, "\n");
    }

    fclose(file);
    return TRUE;
}

int CKDataArray::GetRowCount() {
    return m_DataMatrix.Size();
}

CKDataRow *CKDataArray::GetRow(int n) {
    if (n < 0 || n >= m_DataMatrix.Size())
        return nullptr;
    return m_DataMatrix[n];
}

void CKDataArray::AddRow() {
    if (m_Order) {
        m_Order = FALSE;
    }

    CKDataRow *newRow = new CKDataRow();
    newRow->Resize(m_FormatArray.Size());

    for (int i = 0; i < m_FormatArray.Size(); ++i) {
        ColumnFormat *fmt = m_FormatArray[i];
        if (fmt->m_Type == CKARRAYTYPE_PARAMETER) {
            CKParameterOut *param = m_Context->CreateCKParameterOut(nullptr, fmt->m_ParameterType, IsDynamic());
            if (param) {
                param->SetOwner(this);
                (*newRow)[i] = (CKDWORD) param;
            } else {
                (*newRow)[i] = 0;
            }
        } else {
            (*newRow)[i] = 0;
        }
    }

    m_DataMatrix.PushBack(newRow);
}

CKDataRow *CKDataArray::InsertRow(int n) {
    if (m_Order) {
        m_Order = FALSE;
    }

    if (n < -1 || n > m_DataMatrix.Size()) {
        return nullptr;
    }

    CKDataRow *newRow = new CKDataRow();
    newRow->Resize(m_FormatArray.Size());

    for (int i = 0; i < m_FormatArray.Size(); ++i) {
        ColumnFormat *fmt = m_FormatArray[i];

        if (fmt->m_Type == CKARRAYTYPE_PARAMETER) {
            CKParameterOut *param = m_Context->CreateCKParameterOut(nullptr, fmt->m_ParameterType, IsDynamic());
            if (param) {
                param->SetOwner(this);
                (*newRow)[i] = reinterpret_cast<CKDWORD>(param);
            } else {
                (*newRow)[i] = 0;
            }
        } else {
            (*newRow)[i] = 0;
        }
    }

    if (n == -1 || n == m_DataMatrix.Size()) {
        m_DataMatrix.PushBack(newRow);
    } else {
        m_DataMatrix.Insert(n, newRow);
    }

    return newRow;
}

CKBOOL CKDataArray::TestRow(int row, int c, CK_COMPOPERATOR op, CKDWORD key, int size) {
    if (c < 0 || c >= m_FormatArray.Size() || row < 0 || row >= m_DataMatrix.Size()) {
        return FALSE;
    }

    g_ColumnIndex = c;
    g_Value = key;
    g_Operator = op;
    g_ValueSize = size;

    CKDataRow *dataRow = m_DataMatrix[row];
    ArrayEqualFunction equalFunc = m_FormatArray[c]->m_EqualFunction;

    return equalFunc(dataRow);
}

int CKDataArray::FindRowIndex(int c, CK_COMPOPERATOR op, CKDWORD key, int size, int startIndex) {
    if (c < 0 || c >= m_FormatArray.Size() || startIndex < 0 || startIndex >= m_DataMatrix.Size()) {
        return -1;
    }

    g_ColumnIndex = c;
    g_Value = key;
    g_Operator = op;
    g_ValueSize = size;

    ArrayEqualFunction equalFunc = m_FormatArray[c]->m_EqualFunction;

    for (int i = startIndex; i < m_DataMatrix.Size(); ++i) {
        if (equalFunc(m_DataMatrix[i])) {
            return i;
        }
    }

    return -1;
}

CKDataRow *CKDataArray::FindRow(int c, CK_COMPOPERATOR op, CKDWORD key, int size, int startIndex) {
    if (c < 0 || c >= m_FormatArray.Size() || startIndex < 0 || startIndex >= m_DataMatrix.Size()) {
        return nullptr;
    }

    g_ColumnIndex = c;
    g_Value = key;
    g_Operator = op;
    g_ValueSize = size;

    ArrayEqualFunction equalFunc = m_FormatArray[c]->m_EqualFunction;

    for (int i = startIndex; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *currentRow = m_DataMatrix[i];
        if (equalFunc(currentRow)) {
            return currentRow;
        }
    }

    return nullptr;
}

void CKDataArray::RemoveRow(int n) {
    if (n < 0 || n >= m_DataMatrix.Size())
        return;

    CKDataRow *row = m_DataMatrix[n];

    for (int col = 0; col < m_FormatArray.Size(); ++col) {
        ColumnFormat *fmt = m_FormatArray[col];
        CKDWORD &element = (*row)[col];

        switch (fmt->m_Type) {
        case CKARRAYTYPE_STRING:
            delete[] reinterpret_cast<char *>(element);
            break;

        case CKARRAYTYPE_PARAMETER: {
            CKParameter *param = reinterpret_cast<CKParameter *>(element);
            if (param && param->GetOwner() == this) {
                m_Context->DestroyObject(param);
            }
            break;
        }

        case CKARRAYTYPE_INT:
        case CKARRAYTYPE_FLOAT:
        case CKARRAYTYPE_OBJECT:
            break;
        }
    }

    m_DataMatrix.RemoveAt(n);
    delete row;
}

void CKDataArray::MoveRow(int rsrc, int rdst) {
    const int rowCount = m_DataMatrix.Size();
    if (rsrc < 0 || rsrc >= rowCount)
        return;
    if (rdst < -1 || rdst >= rowCount)
        return;
    if (rsrc == rdst)
        return;

    CKDataRow **dstPtr;
    if (rdst == -1) {
        dstPtr = m_DataMatrix.End();
    } else {
        dstPtr = m_DataMatrix.Begin() + rdst;
        if (dstPtr < m_DataMatrix.Begin() || dstPtr >= m_DataMatrix.End())
            dstPtr = m_DataMatrix.End();
    }

    CKDataRow **srcPtr = m_DataMatrix.Begin() + rsrc;
    if (srcPtr < m_DataMatrix.Begin() || srcPtr >= m_DataMatrix.End())
        srcPtr = m_DataMatrix.End();

    m_DataMatrix.Move(dstPtr, srcPtr);
}

void CKDataArray::SwapRows(int i1, int i2) {
    const int rowCount = m_DataMatrix.Size();
    if (i1 < 0 || i1 >= rowCount)
        return;
    if (i2 < 0 || i2 >= rowCount)
        return;
    if (i1 == i2)
        return;

    CKDataRow *temp = m_DataMatrix[i1];
    m_DataMatrix[i1] = m_DataMatrix[i2];
    m_DataMatrix[i2] = temp;
}

void CKDataArray::Clear(CKBOOL Params) {
    XArray<CK_ID> paramsToDestroy;

    for (int i = m_DataMatrix.Size() - 1; i >= 0; --i) {
        CKDataRow *row = m_DataMatrix[i];

        for (int col = 0; col < m_FormatArray.Size(); ++col) {
            CKDWORD &element = (*row)[col];
            ColumnFormat *fmt = m_FormatArray[col];

            switch (fmt->m_Type) {
            case CKARRAYTYPE_STRING:
                delete[] (char *) element;
                element = 0;
                break;

            case CKARRAYTYPE_PARAMETER: {
                CKParameter *param = (CKParameter *) element;
                if (param) {
                    if (Params) {
                        if (param->GetOwner() == this) {
                            paramsToDestroy.PushBack(param->GetID());
                        }
                    }
                }
                element = 0;
                break;
            }

            case CKARRAYTYPE_INT:
            case CKARRAYTYPE_FLOAT:
            case CKARRAYTYPE_OBJECT:
                element = 0;
                break;
            }
        }

        delete row;
        m_DataMatrix.RemoveAt(i);
    }
    m_DataMatrix.Clear();

    if (Params && !paramsToDestroy.IsEmpty()) {
        m_Context->DestroyObjects(paramsToDestroy.Begin(), paramsToDestroy.Size());
    }
}

void CKDataArray::DataDelete(CKBOOL Params) {
    Clear(Params);
    for (int i = 0; i < m_FormatArray.Size(); ++i) {
        ColumnFormat *fmt = m_FormatArray[i];
        if (fmt) {
            delete[] fmt->m_Name;
            delete fmt;
        }
    }
    m_FormatArray.Clear();
}

CKBOOL CKDataArray::GetHighest(int c, int &row) {
    if (c < 0 || c >= m_FormatArray.Size() || m_DataMatrix.Size() == 0) {
        return FALSE;
    }

    if (m_Order && m_ColumnIndex == c) {
        row = m_Order ? m_DataMatrix.Size() - 1 : 0;
        return TRUE;
    }

    g_ColumnIndex = c;
    g_Order = TRUE;

    int maxIndex = 0;
    CKDataRow *maxRow = m_DataMatrix[0];
    ArraySortFunction comparator = m_FormatArray[c]->m_SortFunction;

    for (int i = 1; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *currentRow = m_DataMatrix[i];

        if (comparator(currentRow, maxRow) > 0) {
            maxRow = currentRow;
            maxIndex = i;
        }
    }

    row = maxIndex;
    return TRUE;
}

CKBOOL CKDataArray::GetLowest(int c, int &row) {
    if (c < 0 || c >= m_FormatArray.Size() || m_DataMatrix.Size() == 0) {
        return FALSE;
    }

    if (m_Order && m_ColumnIndex == c) {
        row = m_Order ? 0 : m_DataMatrix.Size() - 1;
        return TRUE;
    }

    g_ColumnIndex = c;
    g_Order = TRUE;

    int minIndex = 0;
    CKDataRow *minRow = m_DataMatrix[0];
    ArraySortFunction comparator = m_FormatArray[c]->m_SortFunction;

    for (int i = 1; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *currentRow = m_DataMatrix[i];

        if (comparator(currentRow, minRow) < 0) {
            minRow = currentRow;
            minIndex = i;
        }
    }

    row = minIndex;
    return TRUE;
}

CKBOOL CKDataArray::GetNearest(int c, void *value, int &row) {
    if (c < 0 || c >= m_FormatArray.Size() ||
        m_DataMatrix.Size() == 0 || !value) {
        return FALSE;
    }

    ColumnFormat *fmt = m_FormatArray[c];
    row = 0;

    switch (fmt->m_Type) {
    case CKARRAYTYPE_INT: {
        const int targetValue = *static_cast<int *>(value);
        int currentMin = 100000000;

        for (int i = 0; i < m_DataMatrix.Size(); ++i) {
            int rowValue = (int) (*m_DataMatrix[i])[c];
            int diff = abs(targetValue - rowValue);

            if (diff < currentMin) {
                currentMin = diff;
                row = i;
            }
        }
        break;
    }

    case CKARRAYTYPE_FLOAT: {
        const float targetValue = *static_cast<float *>(value);
        float currentMin = 100000000.0f;

        for (int i = 0; i < m_DataMatrix.Size(); ++i) {
            float rowValue = (float) (*m_DataMatrix[i])[c];
            float diff = (float) fabs(targetValue - rowValue);

            if (diff < currentMin) {
                currentMin = diff;
                row = i;
            }
        }
        break;
    }

    case CKARRAYTYPE_STRING: {
        const char *targetStr = static_cast<const char *>(value);
        int bestScore = 100000000;

        for (int i = 0; i < m_DataMatrix.Size(); ++i) {
            const char *rowStr = reinterpret_cast<const char *>((*m_DataMatrix[i])[c]);
            int score = 0;

            if (rowStr) {
                const char *t = targetStr;
                const char *r = rowStr;
                int matchCount = 0;

                while (*t && *r) {
                    if (*t != *r) {
                        score += abs(*t - *r);
                        break;
                    }
                    score -= 100; // Reward matching characters
                    t++;
                    r++;
                    matchCount++;
                }

                score += abs((int) strlen(t) - (int) strlen(r)) * 100;
            }

            if (score < bestScore) {
                bestScore = score;
                row = i;
            }
        }
        break;
    }

    default:
        return FALSE;
    }

    return TRUE;
}

void CKDataArray::ColumnTransform(int c, CK_BINARYOPERATOR op, CKDWORD value) {
    if (c < 0 || c >= m_FormatArray.Size())
        return;

    ColumnFormat *fmt = m_FormatArray[c];
    if (fmt->m_Type != CKARRAYTYPE_INT && fmt->m_Type != CKARRAYTYPE_FLOAT)
        return;

    const bool isFloat = (fmt->m_Type == CKARRAYTYPE_FLOAT);
    const float floatValue = reinterpret_cast<const float &>(value);

    for (int i = 0; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *row = m_DataMatrix[i];
        CKDWORD &element = (*row)[c];

        if (isFloat) {
            float current = reinterpret_cast<float &>(element);
            switch (op) {
            case CKADD: current += floatValue;
                break;
            case CKSUB: current -= floatValue;
                break;
            case CKMUL: current *= floatValue;
                break;
            case CKDIV: if (floatValue != 0.0f) current /= floatValue;
                break;
            }
            element = reinterpret_cast<CKDWORD &>(current);
        } else {
            int current = static_cast<int>(element);
            switch (op) {
            case CKADD: current += static_cast<int>(value);
                break;
            case CKSUB: current -= static_cast<int>(value);
                break;
            case CKMUL: current *= static_cast<int>(value);
                break;
            case CKDIV: if (value != 0) current /= static_cast<int>(value);
                break;
            }
            element = static_cast<CKDWORD>(current);
        }
    }
}

void CKDataArray::ColumnsOperate(int c1, CK_BINARYOPERATOR op, int c2, int cr) {
    if (c1 < 0 || c1 >= m_FormatArray.Size() ||
        c2 < 0 || c2 >= m_FormatArray.Size() ||
        cr < 0 || cr >= m_FormatArray.Size()) {
        return;
    }

    ColumnFormat *fmt1 = m_FormatArray[c1];
    ColumnFormat *fmt2 = m_FormatArray[c2];
    ColumnFormat *fmtResult = m_FormatArray[cr];

    if (fmt1->m_Type != fmt2->m_Type ||
        fmt1->m_Type != fmtResult->m_Type) {
        return;
    }

    const bool isFloat = (fmt1->m_Type == CKARRAYTYPE_FLOAT);

    for (int i = 0; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *row = m_DataMatrix[i];

        CKDWORD val1 = (*row)[c1];
        CKDWORD val2 = (*row)[c2];
        CKDWORD result = 0;

        if (isFloat) {
            float f1 = reinterpret_cast<const float &>(val1);
            float f2 = reinterpret_cast<const float &>(val2);
            float fres = 0.0f;

            switch (op) {
            case CKADD: fres = f1 + f2;
                break;
            case CKSUB: fres = f1 - f2;
                break;
            case CKMUL: fres = f1 * f2;
                break;
            case CKDIV:
                if (f2 != 0.0f) fres = f1 / f2;
                break;
            }
            result = reinterpret_cast<const CKDWORD &>(fres);
        } else {
            int i1 = static_cast<int>(val1);
            int i2 = static_cast<int>(val2);
            int ires = 0;

            switch (op) {
            case CKADD: ires = i1 + i2;
                break;
            case CKSUB: ires = i1 - i2;
                break;
            case CKMUL: ires = i1 * i2;
                break;
            case CKDIV:
                if (i2 != 0) ires = i1 / i2;
                break;
            }
            result = static_cast<CKDWORD>(ires);
        }

        (*row)[cr] = result;
    }
}

void CKDataArray::Sort(int c, CKBOOL ascending) {
    if (c < 0 || c >= m_FormatArray.Size())
        return;

    bool needsSort = !m_Order || (m_ColumnIndex != c) || (m_Order != ascending);
    if (!needsSort)
        return;

    g_ColumnIndex = c;
    g_Order = ascending;
    g_SortFunction = m_FormatArray[c]->m_SortFunction;

    m_DataMatrix.Sort(DataRowCompare);

    m_Order = ascending;
    m_ColumnIndex = c;
}

void CKDataArray::Unique(int c) {
    if (c < 0 || c >= m_FormatArray.Size())
        return;

    g_ColumnIndex = c;
    g_Operator = CKEQUAL;
    g_Order = TRUE;

    Sort(c, TRUE);

    ColumnFormat *fmt = m_FormatArray[c];
    ArrayEqualFunction equalFunc = fmt->m_EqualFunction;

    for (int i = m_DataMatrix.Size() - 1; i > 0; --i) {
        CKDataRow *current = m_DataMatrix[i];
        CKDataRow *previous = m_DataMatrix[i - 1];

        if (equalFunc(current) == equalFunc(previous)) {
            switch (fmt->m_Type) {
            case CKARRAYTYPE_STRING:
                delete[] (char *) (*current)[c];
                break;
            case CKARRAYTYPE_PARAMETER: {
                CKParameter *param = (CKParameter *) (*current)[c];
                if (param && param->GetOwner() == this) {
                    m_Context->DestroyObject(param);
                }
                break;
            }
            default:
                break;
            }

            delete current;
            m_DataMatrix.RemoveAt(i);
        }
    }
}

void CKDataArray::RandomShuffle() {
    const int rowCount = m_DataMatrix.Size();
    if (rowCount == 0)
        return;

    for (int i = 0; i < rowCount; ++i) {
        const int randomIndex = rand() % rowCount;
        SwapRows(i, randomIndex);
    }
}

void CKDataArray::Reverse() {
    const int count = m_DataMatrix.Size();
    int start = 0;
    int end = count - 1;

    while (start < end) {
        SwapRows(start++, end--);
    }
}

CKDWORD CKDataArray::Sum(int c) {
    if (c < 0 || c >= m_FormatArray.Size())
        return 0;

    ColumnFormat *fmt = m_FormatArray[c];

    if (fmt->m_Type == CKARRAYTYPE_INT) {
        CKDWORD sum = 0;
        for (CKDataMatrix::Iterator it = m_DataMatrix.Begin(); it != m_DataMatrix.End(); ++it) {
            sum += (**it)[c];
        }
        return sum;
    }

    if (fmt->m_Type == CKARRAYTYPE_FLOAT) {
        float floatSum = 0.0f;
        for (CKDataMatrix::Iterator it = m_DataMatrix.Begin(); it != m_DataMatrix.End(); ++it) {
            CKDataRow *row = *it;
            CKDWORD *begin = row->Begin();
            CKDWORD *end = row->End();

            if (c < (end - begin)) {
                float *valuePtr = reinterpret_cast<float *>(&begin[c]);
                floatSum += *valuePtr;
            }
        }
        return reinterpret_cast<CKDWORD &>(floatSum);
    }

    return 0;
}

CKDWORD CKDataArray::Product(int c) {
    if (c < 0 || c >= m_FormatArray.Size())
        return 0;

    ColumnFormat *fmt = m_FormatArray[c];
    g_ColumnIndex = c;

    if (fmt->m_Type == CKARRAYTYPE_INT) {
        CKDWORD product = 1;
        for (CKDataMatrix::Iterator it = m_DataMatrix.Begin(); it != m_DataMatrix.End(); ++it) {
            product *= (**it)[c];
        }
        return product;
    }

    if (fmt->m_Type == CKARRAYTYPE_FLOAT) {
        float floatProduct = 1.0f;
        for (CKDataMatrix::Iterator it = m_DataMatrix.Begin(); it != m_DataMatrix.End(); ++it) {
            CKDataRow *row = *it;
            CKDWORD *begin = row->Begin();
            CKDWORD *end = row->End();

            if (c < (end - begin)) {
                float *valuePtr = reinterpret_cast<float *>(&begin[c]);
                floatProduct *= *valuePtr;
            }
        }
        return reinterpret_cast<CKDWORD &>(floatProduct);
    }

    return 0;
}

int CKDataArray::GetCount(int c, CK_COMPOPERATOR op, CKDWORD key, int size) {
    if (c < 0 || c >= m_FormatArray.Size())
        return 0;

    g_ColumnIndex = c;
    g_Operator = op;
    g_Value = key;
    g_ValueSize = size;

    ArrayEqualFunction equalFunc = m_FormatArray[c]->m_EqualFunction;
    int count = 0;

    CKDataRow **current = m_DataMatrix.Begin();
    CKDataRow **end = m_DataMatrix.End();

    while (current != end) {
        if (equalFunc(*current)) {
            count++;
        }
        ++current;
    }

    return count;
}

void CKDataArray::CreateGroup(int mc, CK_COMPOPERATOR op, CKDWORD key, int size, CKGroup *group, int ec) {
    if (!group)
        return;

    group->Clear();

    if (ec < 0 || ec >= m_FormatArray.Size() ||
        mc < 0 || mc >= m_FormatArray.Size()) {
        return;
    }

    ColumnFormat *elementFormat = m_FormatArray[ec];
    if (elementFormat->m_Type != CKARRAYTYPE_OBJECT)
        return;

    g_ColumnIndex = mc;
    g_Operator = op;
    g_Value = key;
    g_ValueSize = size;

    ArrayEqualFunction equalFunc = m_FormatArray[mc]->m_EqualFunction;

    for (CKDataMatrix::Iterator it = m_DataMatrix.Begin(); it != m_DataMatrix.End(); ++it) {
        CKDataRow *row = *it;

        if (equalFunc(row)) {
            CKDWORD objID = (*row)[ec];
            CKObject *obj = m_Context->GetObject(objID);
            if (obj && CKIsChildClassOf(obj, CKCID_BEOBJECT)) {
                CKBeObject *beo = (CKBeObject *) obj;
                group->AddObject(beo);
            }
        }
    }
}

CKDataArray::CKDataArray(CKContext *Context, CKSTRING Name) : CKBeObject(Context, Name) {
    m_KeyColumn = -1;
    m_Order = FALSE;
    m_ColumnIndex = 0;
}

CKDataArray::~CKDataArray() {
    DataDelete(FALSE);
}

CK_CLASSID CKDataArray::GetClassID() {
    return m_ClassID;
}

void CKDataArray::PreSave(CKFile *file, CKDWORD flags) {
    CKBeObject::PreSave(file, flags);

    const int columnCount = m_FormatArray.Size();
    const int rowCount = m_DataMatrix.Size();
    if (columnCount == 0 || rowCount == 0)
        return;

    for (CKDataMatrix::Iterator it = m_DataMatrix.Begin(); it != m_DataMatrix.End(); ++it) {
        CKDataRow *row = *it;
        CKDWORD *rowData = row->Begin();

        for (int col = 0; col < columnCount; ++col) {
            ColumnFormat *fmt = m_FormatArray[col];
            if (fmt->m_Type == CKARRAYTYPE_PARAMETER) {
                CKParameter *param = (CKParameter *) rowData[col];
                if (param) {
                    file->SaveObject(param, -1);
                }
            }
        }
    }
}

CKStateChunk *CKDataArray::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKBeObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_DATAARRAY, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    chunk->WriteIdentifier(CK_STATESAVE_DATAARRAYFORMAT);
    chunk->WriteInt(m_FormatArray.Size());

    for (int i = 0; i < m_FormatArray.Size(); ++i) {
        ColumnFormat *fmt = m_FormatArray[i];
        chunk->WriteString(fmt->m_Name);
        chunk->WriteDword(fmt->m_Type);

        if (fmt->m_Type == CKARRAYTYPE_PARAMETER) {
            chunk->WriteGuid(fmt->m_ParameterType);
        }
    }

    chunk->WriteIdentifier(CK_STATESAVE_DATAARRAYDATA);
    chunk->WriteInt(m_DataMatrix.Size());

    for (int rowIdx = 0; rowIdx < m_DataMatrix.Size(); ++rowIdx) {
        CKDataRow *row = m_DataMatrix[rowIdx];

        for (int colIdx = 0; colIdx < m_FormatArray.Size(); ++colIdx) {
            ColumnFormat *fmt = m_FormatArray[colIdx];
            CKDWORD element = (*row)[colIdx];

            switch (fmt->m_Type) {
            case CKARRAYTYPE_INT:
                chunk->WriteInt((int) element);
                break;

            case CKARRAYTYPE_FLOAT:
                chunk->WriteFloat(*(float *) &element);
                break;

            case CKARRAYTYPE_STRING: {
                char *str = (char *) element;
                chunk->WriteString(str ? str : (char *) "");
                break;
            }

            case CKARRAYTYPE_OBJECT: {
                CKObject *obj = m_Context->GetObject(element);
                chunk->WriteObject(obj);
                break;
            }

            case CKARRAYTYPE_PARAMETER: {
                CKParameterOut *param = (CKParameterOut *) element;
                if (file) {
                    chunk->WriteObject(param);
                } else {
                    CKBOOL owned = (param->GetOwner() == this);
                    if (owned)
                        param->SetOwner(nullptr);

                    CKStateChunk *paramChunk = param->Save(nullptr, CK_STATESAVE_PARAMETEROUT_ALL);
                    chunk->WriteSubChunk(paramChunk);

                    if (paramChunk) {
                        delete paramChunk;
                    }
                    if (owned)
                        param->SetOwner(this);
                }
                break;
            }
            }
        }
    }

    chunk->WriteIdentifier(CK_STATESAVE_DATAARRAYMEMBERS);
    chunk->WriteInt(m_Order);
    chunk->WriteDword(m_ColumnIndex);
    chunk->WriteInt(m_KeyColumn);

    if (GetClassID() == CKCID_DATAARRAY) {
        chunk->CloseChunk();
    } else {
        chunk->UpdateDataSize();
    }
    return chunk;
}

CKERROR CKDataArray::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    CKERROR err = CKBeObject::Load(chunk, file);
    if (err != CK_OK)
        return err;

    // Load column formats
    if (chunk->SeekIdentifier(CK_STATESAVE_DATAARRAYFORMAT)) {
        DataDelete(TRUE); // Clear existing data

        int columnCount = chunk->ReadInt();
        m_FormatArray.Resize(columnCount);

        for (int i = 0; i < columnCount; ++i) {
            ColumnFormat *fmt = new ColumnFormat();

            // Read format properties
            chunk->ReadString(&fmt->m_Name);
            fmt->m_Type = (CK_ARRAYTYPE) chunk->ReadInt();

            // Set comparison functions
            switch (fmt->m_Type) {
            case CKARRAYTYPE_INT:
                fmt->m_SortFunction = ArrayIntComp;
                fmt->m_EqualFunction = ArrayIntEqual;
                break;
            case CKARRAYTYPE_FLOAT:
                fmt->m_SortFunction = ArrayFloatComp;
                fmt->m_EqualFunction = ArrayFloatEqual;
                break;
            case CKARRAYTYPE_STRING:
                fmt->m_SortFunction = ArrayStringComp;
                fmt->m_EqualFunction = ArrayStringEqual;
                break;
            case CKARRAYTYPE_OBJECT:
                fmt->m_SortFunction = ArrayIntComp;
                fmt->m_EqualFunction = ArrayIntEqual;
                break;
            case CKARRAYTYPE_PARAMETER:
                fmt->m_SortFunction = ArrayParameterComp;
                fmt->m_EqualFunction = ArrayParameterEqual;
                CKGUID guid = chunk->ReadGuid();
                if (guid == CKPGUID_OLDTIME)
                    guid = CKPGUID_TIME;
                fmt->m_ParameterType = guid;
                break;
            }

            m_FormatArray[i] = fmt;
        }
    }

    // Load data rows
    if (chunk->SeekIdentifier(CK_STATESAVE_DATAARRAYDATA)) {
        int rowCount = chunk->ReadInt();
        m_DataMatrix.Reserve(rowCount);

        for (int rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
            CKDataRow *newRow = new CKDataRow();
            newRow->Resize(m_FormatArray.Size());

            for (int colIdx = 0; colIdx < m_FormatArray.Size(); ++colIdx) {
                ColumnFormat *fmt = m_FormatArray[colIdx];
                CKDWORD &element = (*newRow)[colIdx];

                switch (fmt->m_Type) {
                case CKARRAYTYPE_INT:
                    element = chunk->ReadInt();
                    break;

                case CKARRAYTYPE_FLOAT: {
                    float f = chunk->ReadFloat();
                    element = *(CKDWORD *) &f;
                    break;
                }

                case CKARRAYTYPE_STRING: {
                    char *str = nullptr;
                    chunk->ReadString(&str);
                    element = (CKDWORD) str;
                    break;
                }

                case CKARRAYTYPE_OBJECT:
                    element = chunk->ReadObjectID();
                    break;

                case CKARRAYTYPE_PARAMETER: {
                    if (file) {
                        // Load parameter reference
                        CK_ID paramId = chunk->ReadObjectID();
                        element = (CKDWORD) m_Context->GetObject(paramId);
                    } else {
                        // Load parameter data
                        CKStateChunk *paramChunk = chunk->ReadSubChunk();
                        CKParameterOut *param = m_Context->CreateCKParameterOut(nullptr, fmt->m_ParameterType, IsDynamic());
                        if (param) {
                            param->Load(paramChunk, nullptr);
                            if (param->GetOwner() == nullptr)
                                param->SetOwner(this);
                            element = (CKDWORD) param;
                        }

                        delete paramChunk;
                    }
                    break;
                }
                }
            }
            m_DataMatrix.PushBack(newRow);
        }
    }

    // Load additional properties
    if (chunk->SeekIdentifier(CK_STATESAVE_DATAARRAYMEMBERS)) {
        m_Order = chunk->ReadInt();
        m_ColumnIndex = chunk->ReadInt();

        // Key column was added in version 5
        if (file || chunk->GetDataVersion() >= 5) {
            m_KeyColumn = chunk->ReadInt();
        }
    }

    return CK_OK;
}

void CKDataArray::PostLoad() {
    for (CKDataMatrix::Iterator it = m_DataMatrix.Begin(); it != m_DataMatrix.End(); ++it) {
        CKDataRow *row = *it;
        if (!row)
            continue;

        const int columnCount = m_FormatArray.Size();
        for (int col = 0; col < columnCount; ++col) {
            ColumnFormat *fmt = m_FormatArray[col];

            if (fmt->m_Type != CKARRAYTYPE_PARAMETER)
                continue;

            CKDWORD &element = (*row)[col];
            CKParameter *param = (CKParameter *) element;

            if (param) {
                if (!param->GetOwner()) {
                    param->SetOwner(this);
                }
            } else {
                CKParameterOut *newParam = m_Context->CreateCKParameterOut(nullptr, fmt->m_ParameterType);
                if (newParam) {
                    newParam->SetOwner(this);
                    element = (CKDWORD) newParam;
                }
            }
        }
    }

    CKObject::PostLoad();
}

void CKDataArray::CheckPreDeletion() {
    CKObject::CheckPreDeletion();

    XArray<int> paramColumns;
    for (int i = 0; i < m_FormatArray.Size(); ++i) {
        if (m_FormatArray[i]->m_Type == CKARRAYTYPE_PARAMETER) {
            paramColumns.PushBack(i);
        }
    }

    for (int rowIdx = 0; rowIdx < m_DataMatrix.Size(); ++rowIdx) {
        CKDataRow *row = m_DataMatrix[rowIdx];

        for (int colIdx = 0; colIdx < paramColumns.Size(); ++colIdx) {
            int paramCol = paramColumns[colIdx];
            CKDWORD &element = (*row)[paramCol];

            CKParameterOut *param = (CKParameterOut *) element;
            if (!param)
                continue;

            if (param->IsToBeDeleted() && param->GetOwner() != this) {
                CKParameterOut *newParam = m_Context->CreateCKParameterOut(nullptr, param->GetType(), IsDynamic());
                if (newParam) {
                    newParam->CopyValue(param, TRUE);
                    newParam->SetOwner(this);
                    element = (CKDWORD) newParam;
                }
            }
        }
    }
}

void CKDataArray::CheckPostDeletion() {
    CKObject::CheckPostDeletion();

    XArray<int> objectColumns;
    for (int i = 0; i < m_FormatArray.Size(); ++i) {
        if (m_FormatArray[i]->m_Type == CKARRAYTYPE_OBJECT) {
            objectColumns.PushBack(i);
        }
    }

    for (int rowIdx = 0; rowIdx < m_DataMatrix.Size(); ++rowIdx) {
        CKDataRow *row = m_DataMatrix[rowIdx];

        for (int colIdx = 0; colIdx < objectColumns.Size(); ++colIdx) {
            int objectCol = objectColumns[colIdx];
            CKDWORD &objectId = (*row)[objectCol];

            if (!m_Context->GetObject(objectId)) {
                objectId = 0; // Clear invalid reference
            }
        }
    }
}

int CKDataArray::GetMemoryOccupation() {
    int size = CKBeObject::GetMemoryOccupation() + (int) (sizeof(CKDataArray) - sizeof(CKBeObject));

    size += m_FormatArray.GetMemoryOccupation(FALSE);
    for (int i = 0; i < m_FormatArray.Size(); ++i) {
        ColumnFormat *fmt = m_FormatArray[i];
        if (!fmt)
            continue;
        size += (int) sizeof(*fmt);
        if (fmt->m_Name) {
            size += (int) (strlen(fmt->m_Name) + 1);
        }
    }

    size += m_DataMatrix.GetMemoryOccupation(FALSE);
    for (int i = 0; i < m_DataMatrix.Size(); ++i) {
        CKDataRow *row = m_DataMatrix[i];
        if (!row)
            continue;
        size += row->GetMemoryOccupation(TRUE);
    }

    return size;
}

int CKDataArray::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    CKParameterManager *pm = m_Context->GetParameterManager();
    CK_ID targetId = o ? o->GetID() : 0;

    // Check parameter columns that might reference objects
    XArray<int> paramObjectColumns;
    for (int col = 0; col < m_FormatArray.Size(); ++col) {
        ColumnFormat *fmt = m_FormatArray[col];
        if (fmt->m_Type == CKARRAYTYPE_PARAMETER) {
            CK_CLASSID paramClass = pm->GuidToClassID(fmt->m_ParameterType);
            if (paramClass != 0) {
                paramObjectColumns.PushBack(col);
            }
        }
    }

    // Search parameter columns
    for (int row = 0; row < m_DataMatrix.Size(); ++row) {
        CKDataRow *dataRow = m_DataMatrix[row];
        for (int pcol = 0; pcol < paramObjectColumns.Size(); ++pcol) {
            int col = paramObjectColumns[pcol];
            CKDWORD element = (*dataRow)[col];
            CKParameter *param = (CKParameter *) element;
            if (param) {
                CK_ID *idPtr = (CK_ID *) param->GetReadDataPtr();
                if (idPtr && *idPtr == targetId) {
                    return 1;
                }
            }
        }
    }

    // Check direct object columns
    XArray<int> objectColumns;
    for (int col = 0; col < m_FormatArray.Size(); ++col) {
        if (m_FormatArray[col]->m_Type == CKARRAYTYPE_OBJECT) {
            objectColumns.PushBack(col);
        }
    }

    // Search object columns
    for (int row = 0; row < m_DataMatrix.Size(); ++row) {
        CKDataRow *dataRow = m_DataMatrix[row];
        for (int ocol = 0; ocol < objectColumns.Size(); ++ocol) {
            int col = objectColumns[ocol];
            CKDWORD element = (*dataRow)[col];
            if (element == targetId) {
                return 1;
            }
        }
    }

    // Fallback to base class check
    return CKBeObject::IsObjectUsed(o, cid);
}

CKERROR CKDataArray::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKBeObject::PrepareDependencies(context);
    if (err != CK_OK)
        return err;

    CKDWORD classDeps = context.GetClassDependencies(m_ClassID);

    // DLL check: context.m_Mode != 1 || (classDeps & 2) != 0
    // Equivalent to: !IsInMode(CK_DEPENDENCIES_COPY) || (classDeps & 2)
    if (!context.IsInMode(CK_DEPENDENCIES_COPY) || (classDeps & 2)) {
        for (int rowIdx = 0; rowIdx < m_DataMatrix.Size(); ++rowIdx) {
            CKDataRow *row = m_DataMatrix[rowIdx];

            for (int colIdx = 0; colIdx < m_FormatArray.Size(); ++colIdx) {
                ColumnFormat *format = m_FormatArray[colIdx];
                CKDWORD &element = (*row)[colIdx];

                switch (format->m_Type) {
                case CKARRAYTYPE_OBJECT: {
                    if (classDeps & 1) {
                        CK_ID id = element;
                        CKObject *obj = m_Context->GetObject(id);
                        if (obj && !CKIsChildClassOf(obj, CKCID_LEVEL) && !CKIsChildClassOf(obj, CKCID_SCENE)) {
                            obj->PrepareDependencies(context);
                        }
                    }
                    break;
                }

                case CKARRAYTYPE_PARAMETER: {
                    CKParameter *param = reinterpret_cast<CKParameter *>(element);
                    // DLL: Check if param is owned by this (IsStateActive returns owner)
                    if (param && param->GetOwner() == this) {
                        param->PrepareDependencies(context);
                    }

                    if (classDeps & 1) {
                        if (param) {
                            CKParameterManager *pm = m_Context->GetParameterManager();
                            CK_CLASSID paramClass = pm->TypeToClassID(param->GetType());
                            if (paramClass != 0) {
                                CK_ID *idPtr = (CK_ID *) param->GetReadDataPtr();
                                if (idPtr) {
                                    CKObject *obj = m_Context->GetObject(*idPtr);
                                    if (obj && !CKIsChildClassOf(obj, CKCID_LEVEL) && !CKIsChildClassOf(obj, CKCID_SCENE)) {
                                        obj->PrepareDependencies(context);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                default:
                    break;
                }
            }
        }
    }

    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR CKDataArray::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKBeObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    if (context.GetClassDependencies(m_ClassID) & 2) {
        for (int rowIdx = 0; rowIdx < m_DataMatrix.Size(); ++rowIdx) {
            CKDataRow *row = m_DataMatrix[rowIdx];

            for (int colIdx = 0; colIdx < m_FormatArray.Size(); ++colIdx) {
                ColumnFormat *format = m_FormatArray[colIdx];
                CKDWORD &element = (*row)[colIdx];

                switch (format->m_Type) {
                case CKARRAYTYPE_OBJECT: {
                    // DLL: Uses direct hash table lookup on m_MapID for OBJECT columns
                    CK_ID oldId = element;
                    CK_ID newId = context.RemapID(oldId);
                    if (newId != 0) {
                        element = newId;
                    }
                    break;
                }

                case CKARRAYTYPE_PARAMETER: {
                    // DLL: Uses context.Remap for PARAMETER columns (remaps the pointer)
                    CKObject *param = (CKObject *) element;
                    CKObject *newParam = context.Remap(param);
                    if (newParam) {
                        element = (CKDWORD) newParam;
                    }
                    break;
                }
                default:
                    break;
                }
            }
        }
    }

    return CK_OK;
}

CKERROR CKDataArray::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKBeObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    CKDataArray *src = (CKDataArray *) &o;
    CKDWORD classDeps = context.GetClassDependencies(m_ClassID);

    // DLL: Resize format array to match source, then copy-construct each format
    m_FormatArray.Resize(src->m_FormatArray.Size());
    for (int i = 0; i < src->m_FormatArray.Size(); ++i) {
        ColumnFormat *newFormat = new ColumnFormat(*src->m_FormatArray[i]);
        m_FormatArray[i] = newFormat;
    }

    m_KeyColumn = src->m_KeyColumn;
    m_Order = src->m_Order;
    m_ColumnIndex = src->m_ColumnIndex;

    if (!(classDeps & 2))
        return CK_OK;

    // Copy data matrix
    int srcRowCount = src->m_DataMatrix.Size();
    int columnCount = m_FormatArray.Size();

    for (int rowIdx = 0; rowIdx < srcRowCount; ++rowIdx) {
        CKDataRow *srcRow = src->GetRow(rowIdx);
        CKDWORD *srcData = srcRow->Begin();

        CKDataRow *newRow = new CKDataRow();
        newRow->Resize(columnCount);

        for (int colIdx = 0; colIdx < columnCount; ++colIdx) {
            ColumnFormat *fmt = m_FormatArray[colIdx];
            CKDWORD element = srcData[colIdx];

            // DLL: Only STRING type is deep-copied, all others are raw DWORD copy
            if (fmt->m_Type == CKARRAYTYPE_STRING) {
                char *str = (char *) element;
                (*newRow)[colIdx] = (CKDWORD) CKStrdup(str);
            } else {
                (*newRow)[colIdx] = element;
            }
        }
        m_DataMatrix.PushBack(newRow);
    }

    return CK_OK;
}

CKSTRING CKDataArray::GetClassName() {
    return "Array";
}

int CKDataArray::GetDependenciesCount(int mode) {
    if (mode == 1) {
        return 2;
    }
    return mode == 2 ? 1 : 0;
}

CKSTRING CKDataArray::GetDependencies(int i, int mode) {
    if (i == 0) {
        return "Objects";
    }
    if (mode == 1 && i == 1) {
        return "Data";
    }
    return nullptr;
}

void CKDataArray::Register() {
    CKClassNeedNotificationFrom(m_ClassID, CKObject::m_ClassID);
    CKClassRegisterAssociatedParameter(m_ClassID, CKPGUID_DATAARRAY);
    CKClassRegisterDefaultDependencies(m_ClassID, 2, 1);
}

CKDataArray *CKDataArray::CreateInstance(CKContext *Context) {
    return new CKDataArray(Context);
}
