#ifndef CKCALLBACKFUNCTIONS_H
#define CKCALLBACKFUNCTIONS_H

#include "CKDefines.h"

void CKInitializeParameterTypes(CKParameterManager *man);

void CK_ParameterCopier_SaveLoad(CKParameter *dest, CKParameter *src);
void CK_ParameterCopier_SetValue(CKParameter *dest, CKParameter *src);
void CK_ParameterCopier_Dword(CKParameter *dest, CKParameter *src);
void CK_ParameterCopier_Memcpy(CKParameter *dest, CKParameter *src);
void CK_ParameterCopier_BufferCompatibleWithDWORD(CKParameter *dest, CKParameter *src);
void CK_ParameterCopier_Nope(CKParameter *dest, CKParameter *src);

CKERROR CKParameterTypesEnumCreator(CKParameter *param);

CKERROR CKDependenciesCreator(CKParameter *param);
void CKDependenciesDestructor(CKParameter *param);
void CKDependenciesSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load);
void CKDependenciesCopier(CKParameter *dest, CKParameter *src);

CKERROR CKStateChunkCreator(CKParameter *param);
void CKStateChunkDestructor(CKParameter *param);
void CKStateChunkSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load);
void CKStateChunkCopier(CKParameter *dest, CKParameter *src);

CKERROR CKCollectionCreator(CKParameter *param);
void CKCollectionDestructor(CKParameter *param);
void CKCollectionSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load);
void CKCollectionCheck(CKParameter *param);
void CKCollectionCopy(CKParameter *dest, CKParameter *src);

CKERROR CKMatrixCreator(CKParameter *param);

CKERROR CKStructCreator(CKParameter *param);
void CKStructDestructor(CKParameter *param);
void CKStructSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load);
void CKStructCopier(CKParameter *dest, CKParameter *src);

CKERROR CK2dCurveCreator(CKParameter *param);
void CK2dCurveDestructor(CKParameter *param);
void CK2dCurveSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load);
void CK2dCurveCopier(CKParameter *dest, CKParameter *src);

void CKPointerSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load);

void CKObjectCheckFunc(CKParameter *param);

int CKMatrixStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKQuaternionStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKObjectStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKStringStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKAttributeStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKMessageStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKCollectionStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKStructStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKEnumStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKFlagsStringFunc(CKParameter *param, CKSTRING ValueString, CKBOOL ReadFromString);
int CKIntStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKAngleStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKEulerStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CK2DVectorStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKVectorStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKRectStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKBoxStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKColorStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKBoolStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKFloatStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKPercentageStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);
int CKTimeStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString);

CK_PARAMETERUICREATORFUNCTION GetUICreatorFunction(CKContext *context, CKParameterTypeDesc *desc);

#endif // CKCALLBACKFUNCTIONS_H