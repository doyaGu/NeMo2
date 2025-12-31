#ifndef CKDATAARRAY_H
#define CKDATAARRAY_H

#include "CKBeObject.h"

typedef XSArray<CKDWORD> CKDataRow;

typedef int (*ArraySortFunction)(CKDataRow *, CKDataRow *);

typedef CKBOOL (*ArrayEqualFunction)(CKDataRow *);

class ColumnFormat
{
public:
    ColumnFormat()
    {
        m_Name = NULL;
        m_Type = CKARRAYTYPE_INT;
        m_ParameterType = CKGUID(0, 0);
        m_SortFunction = NULL;
        m_EqualFunction = NULL;
    }
    ColumnFormat(const ColumnFormat &c)
    {
        m_Name = CKStrdup(c.m_Name);
        m_Type = c.m_Type;
        m_ParameterType = c.m_ParameterType;
        m_SortFunction = c.m_SortFunction;
        m_EqualFunction = c.m_EqualFunction;
    }

    // Column name
    char *m_Name;
    // Column Type
    CK_ARRAYTYPE m_Type;
    // Parameter Type
    CKGUID m_ParameterType;
    // Sort Function
    ArraySortFunction m_SortFunction;
    // Equal Function
    ArrayEqualFunction m_EqualFunction;
};

typedef XArray<CKDataRow *> CKDataMatrix;

typedef XSArray<ColumnFormat *> CKFormatArray;

/*************************************************
{filename:CKDataArray}
Name: CKDataArray

Summary: Array that contains a collection of data, each column with a unique type of data

Remarks:
+ There exist four basic types of elements you can put in a data array. These types are defined in CK_ARRAYTYPE.
+ The operations provided with the functions ColumnOperate and ColumnTransform only work with basic numeric types (CKARRAYTYPE_INT and CKARRAYTYPE_FLOAT) and not with a parameter of type "Integer" for example.
+ The comparison, as used by the functions Sort,FindLine, TestRow, etc... fully works with basic types (CKARRAYTYPE_INT, CKARRAYTYPE_FLOAT and CKARRAYTYPE_STRING) and works with comparaison type EQUAL or NOTEQUAL with object types (CKARRAYTYPE_OBJECT and CKARRAYTYPE_PARAMETER) (it compares the ID of the object and it does a memcmp on the content of the two compared parameter.)
+ The class id of CKDataArray is CKCID_DATAARRAY.


See also:
*************************************************/
class CKDataArray : public CKBeObject
{
public:
    // Column/Format Functions
    // Insert Column before column cdest (-1 means move at the end)
    DLL_EXPORT void InsertColumn(int cdest, CK_ARRAYTYPE type, char *name, CKGUID paramGuid = CKGUID(0, 0));
    // Move Column csrc before column cdest (-1 means move at the end)
    DLL_EXPORT void MoveColumn(int csrc, int cdest);
    // Remove Column
    DLL_EXPORT void RemoveColumn(int c);
    // Set Column Name
    DLL_EXPORT void SetColumnName(int c, char *name);
    // Get Column Name
    DLL_EXPORT char *GetColumnName(int c);
    // Get Column Format
    DLL_EXPORT void SetColumnType(int c, CK_ARRAYTYPE newType, CKGUID paramGuid = CKGUID(0, 0));
    // Get Column Format
    DLL_EXPORT CK_ARRAYTYPE GetColumnType(int c);
    // Get Column Format
    DLL_EXPORT CKGUID GetColumnParameterGuid(int c);
    // Get Key Column
    DLL_EXPORT int GetKeyColumn();
    // Set Key Column
    DLL_EXPORT void SetKeyColumn(int c);
    // Get Column Number
    DLL_EXPORT int GetColumnCount();

    // Elements Functions

    // Get the element pointer of the specified case
    DLL_EXPORT CKDWORD *GetElement(int i, int c);
    // Use to get an int, a float, a string or an object ID
    DLL_EXPORT CKBOOL GetElementValue(int i, int c, void *value);
    // Use to get an CKObject
    DLL_EXPORT CKObject *GetElementObject(int i, int c);

    // Use to set an int, a float ,an object ID, a string
    DLL_EXPORT CKBOOL SetElementValue(int i, int c, void *value, int size = 0);
    // Use to set a value from a parameter
    DLL_EXPORT CKBOOL SetElementValueFromParameter(int i, int c, CKParameter *pout);
    // Use to set an CKObject
    DLL_EXPORT CKBOOL SetElementObject(int i, int c, CKObject *object);

    // Parameters Shortcuts

    // Paste a shortcut of a parameter on an existing compatible parameter
    DLL_EXPORT CKBOOL PasteShortcut(int i, int c, CKParameter *pout);
    // Remove a shortcut parameter (if it exists)
    DLL_EXPORT CKParameterOut *RemoveShortcut(int i, int c);

    // String Value

    // Set the value of an existing element
    DLL_EXPORT CKBOOL SetElementStringValue(int i, int c, char *svalue);
    // Set the value of an existing element
    DLL_EXPORT int GetStringValue(CKDWORD key, int c, char *svalue);
    // Set the value of an existing element
    DLL_EXPORT int GetElementStringValue(int i, int c, char *svalue);

    // Load / Write

    // load elements into an array from a formatted file
    DLL_EXPORT CKBOOL LoadElements(CKSTRING filename, CKBOOL append, int column);
    // write elements from an array to a file
    DLL_EXPORT CKBOOL WriteElements(CKSTRING filename, int column, int number, CKBOOL append = FALSE);

    // Rows Functions
    // Get row Count
    DLL_EXPORT int GetRowCount();
    // Find the nth line
    DLL_EXPORT CKDataRow *GetRow(int n);
    // adds a row
    DLL_EXPORT void AddRow();
    // Insert a Row before another, -1 means after everything
    DLL_EXPORT CKDataRow *InsertRow(int n = -1);
    // Test a row on a column
    DLL_EXPORT CKBOOL TestRow(int row, int c, CK_COMPOPERATOR op, CKDWORD key, int size = 0);
    // Find a line with the current key : search in the column c and return the line index (-1) if none
    DLL_EXPORT int FindRowIndex(int c, CK_COMPOPERATOR op, CKDWORD key, int size = 0, int startIndex = 0);
    // Find a line with the current key : search in the column c and return the line itself (NULL if none)
    DLL_EXPORT CKDataRow *FindRow(int c, CK_COMPOPERATOR op, CKDWORD key, int size = 0, int startIndex = 0);
    // Remove the nth line
    DLL_EXPORT void RemoveRow(int n);
    // Move a row
    DLL_EXPORT void MoveRow(int rsrc, int rdst);
    // Swap 2 rows
    DLL_EXPORT void SwapRows(int i1, int i2);
    // Clear the entire array
    DLL_EXPORT void Clear(CKBOOL Params = TRUE);
    // Delete the data
    DLL_EXPORT void DataDelete(CKBOOL Params = TRUE);

    ///////////////////////////
    // Algorithm
    ///////////////////////////

    // Find the highest value in the given column
    DLL_EXPORT CKBOOL GetHighest(int c, int &row);
    // Find the lowest value in the given column
    DLL_EXPORT CKBOOL GetLowest(int c, int &row);
    // Find the nearest value in the given column
    DLL_EXPORT CKBOOL GetNearest(int c, void *value, int &row);
    // Transform the values by operating them with the given value
    DLL_EXPORT void ColumnTransform(int c, CK_BINARYOPERATOR op, CKDWORD value);
    // Operate two columns into a third
    DLL_EXPORT void ColumnsOperate(int c1, CK_BINARYOPERATOR op, int c2, int cr);

    // Sort the array on the column, ascending or descending
    DLL_EXPORT void Sort(int c, CKBOOL ascending);
    // Remove the elements identical in the array
    DLL_EXPORT void Unique(int c);
    // Shuffle the array
    DLL_EXPORT void RandomShuffle();
    // Reverse the array
    DLL_EXPORT void Reverse();

    // Get the sum of a specific column
    DLL_EXPORT CKDWORD Sum(int c);
    // Get The product of a specific column
    DLL_EXPORT CKDWORD Product(int c);

    // Get the count of elements verifying a condition (operator)
    DLL_EXPORT int GetCount(int c, CK_COMPOPERATOR op, CKDWORD key, int size = 0);
    // Create a group from elements matching a value
    DLL_EXPORT void CreateGroup(int mc, CK_COMPOPERATOR op, CKDWORD key, int size, CKGroup *group, int ec = 0);

    //-------------------------------------------------------------------------
    // Internal functions

    //-------------------------------------------------------
    // Virtual functions	{Secret}
    CKDataArray(CKContext *Context, CKSTRING Name = NULL);
    virtual ~CKDataArray();
    virtual CK_CLASSID GetClassID();

    virtual void PreSave(CKFile *file, CKDWORD flags);
    virtual CKStateChunk *Save(CKFile *file, CKDWORD flags);
    virtual CKERROR Load(CKStateChunk *chunk, CKFile *file);
    virtual void PostLoad();

    virtual void CheckPreDeletion();
    virtual void CheckPostDeletion();

    virtual size_t GetMemoryOccupation();
    virtual CKBOOL IsObjectUsed(CKObject *o, CK_CLASSID cid);

    //--------------------------------------------
    // Dependencies Functions	{Secret}
    virtual CKERROR PrepareDependencies(CKDependenciesContext &context);
    virtual CKERROR RemapDependencies(CKDependenciesContext &context);
    virtual CKERROR Copy(CKObject &o, CKDependenciesContext &context);

    //--------------------------------------------
    // Class Registering	{Secret}
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKDataArray *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

    // Dynamic Cast method (returns NULL if the object can't be cast)
    static CKDataArray *Cast(CKObject *iO)
    {
        return CKIsChildClassOf(iO, CKCID_DATAARRAY) ? (CKDataArray *)iO : NULL;
    }

    CKFormatArray m_FormatArray;
    CKDataMatrix m_DataMatrix;
    int m_KeyColumn;
    CKBOOL m_Order;
    int m_ColumnIndex;

    static int g_ColumnIndex;
    static CKBOOL g_Order;
    static CK_COMPOPERATOR g_Operator;
    static CKDWORD g_Value;
    static CKDWORD g_ValueSize;
    static ArraySortFunction g_SortFunction;
};

#endif // CKDATAARRAY_H
