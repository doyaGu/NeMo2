#include "CKPluginManager.h"

#include "VxMath.h"
#include "CKGlobals.h"
#include "CKDirectoryParser.h"
#include "CKObjectDeclaration.h"
#include "CKParameterManager.h"
#include "CKAttributeManager.h"
#include "CKBehavior.h"
#include "CKTexture.h"
#include "CKSprite.h"
#include "CKWaveSound.h"
#include "CKBehavior.h"
#include "CKFile.h"

extern XArray<CKContext *> g_Contextes;
extern XObjDeclHashTable g_PrototypeDeclarationList;

CKPluginEntry *g_TheCurrentPluginEntry;

CKPluginEntry &CKPluginEntry::operator=(const CKPluginEntry &ent) {
    if (this == &ent) {
        return *this;
    }

    if (ent.m_ReadersInfo) {
        if (!m_ReadersInfo) {
            m_ReadersInfo = new CKPluginEntryReadersData;
        }
        memcpy(m_ReadersInfo, ent.m_ReadersInfo, sizeof(CKPluginEntryReadersData));
    } else {
        delete m_ReadersInfo;
        m_ReadersInfo = nullptr;
    }

    if (ent.m_BehaviorsInfo) {
        if (!m_BehaviorsInfo) {
            m_BehaviorsInfo = new CKPluginEntryBehaviorsData;
        }
        m_BehaviorsInfo->m_BehaviorsGUID = ent.m_BehaviorsInfo->m_BehaviorsGUID;
    } else {
        delete m_BehaviorsInfo;
        m_BehaviorsInfo = nullptr;
    }

    m_PluginDllIndex = ent.m_PluginDllIndex;
    m_PositionInDll = ent.m_PositionInDll;
    m_PluginInfo = ent.m_PluginInfo;
    m_Active = ent.m_Active;
    m_NeededByFile = ent.m_NeededByFile;
    m_IndexInCategory = ent.m_IndexInCategory;

    return *this;
}

CKPluginEntry::CKPluginEntry()
        : m_PluginDllIndex(0),
          m_PositionInDll(0),
          m_ReadersInfo(nullptr),
          m_BehaviorsInfo(nullptr),
          m_Active(FALSE),
          m_IndexInCategory(0),
          m_NeededByFile(FALSE) {}

CKPluginEntry::CKPluginEntry(const CKPluginEntry &ent) {
    m_PluginDllIndex = ent.m_PluginDllIndex;
    m_PositionInDll = ent.m_PositionInDll;
    m_PluginInfo = ent.m_PluginInfo;
    m_Active = ent.m_Active;
    m_NeededByFile = ent.m_NeededByFile;
    m_IndexInCategory = ent.m_IndexInCategory;
    if (ent.m_ReadersInfo) {
        m_ReadersInfo = new CKPluginEntryReadersData;
        memcpy(m_ReadersInfo, ent.m_ReadersInfo, sizeof(CKPluginEntryReadersData));
    } else {
        m_ReadersInfo = nullptr;
    }
    if (ent.m_BehaviorsInfo) {
        m_BehaviorsInfo = new CKPluginEntryBehaviorsData;
        m_BehaviorsInfo->m_BehaviorsGUID = ent.m_BehaviorsInfo->m_BehaviorsGUID;
    } else {
        m_BehaviorsInfo = nullptr;
    }
}

CKPluginEntry::~CKPluginEntry() {
    delete m_ReadersInfo;
    delete m_BehaviorsInfo;
}

CKPluginManager::CKPluginManager() {}

CKPluginManager::~CKPluginManager() {
    Clean();
    m_RunTimeDlls.Clear();
    m_PluginDlls.Clear();
    m_PluginCategories.Clear();
}

int CKPluginManager::ParsePlugins(CKSTRING Directory) {
    CKDirectoryParser parser(Directory, (CKSTRING) "*.dll", TRUE);
    VxAddLibrarySearchPath(Directory);
    int count = 0;
    for (char *plugin = parser.GetNextFile(); plugin != nullptr; plugin = parser.GetNextFile()) {
        if (RegisterPlugin(plugin) == CK_OK)
            ++count;
    }
    return count;
}

CKERROR CKPluginManager::RegisterPlugin(CKSTRING path) {
    if (!path)
        return CKERR_INVALIDPARAMETER;

    int idx = -1;
    CKPluginDll *plugin = GetPluginDllInfo(path, &idx);
    if (plugin) {
        if (!plugin->m_DllInstance)
            return ReLoadPluginDll(idx);
        return CKERR_ALREADYPRESENT;
    }

    VxSharedLibrary shl;
    auto *handle = shl.Load(path);

    CKPathSplitter ps(path);

    if (!handle)
        return CKERR_INVALIDPLUGIN;

    auto getPluginInfoCountFunc = (CKPluginGetInfoCountFunction) shl.GetFunctionPtr("CKGetPluginInfoCount");
    auto getPluginInfoFunc = (CKPluginGetInfoFunction) shl.GetFunctionPtr("CKGetPluginInfo");
    if (!getPluginInfoFunc) {
        shl.ReleaseLibrary();
        return CKERR_INVALIDPLUGIN;
    }

    XClassArray<CKPluginEntry> pluginEntries;

    CKPluginDll pluginDll;
    pluginDll.m_DllFileName = path;
    pluginDll.m_PluginInfoCount = getPluginInfoCountFunc ? getPluginInfoCountFunc() : 1;
    pluginDll.m_DllInstance = handle;

    CKERROR err = CKERR_INVALIDPLUGIN;
    for (int i = 0; i < pluginDll.m_PluginInfoCount; ++i) {
        CKPluginInfo *info = getPluginInfoFunc(i);
        if (!info)
            continue;

        CKPluginEntry *pEntry = new CKPluginEntry;
        CKPluginEntry &entry = *pEntry;
        entry.m_PluginDllIndex = m_PluginDlls.Size();
        entry.m_PositionInDll = i;
        entry.m_PluginInfo = *info;

        CK_PLUGIN_TYPE type = info->m_Type;
        if (type > m_PluginCategories.Size() - 1)
            m_PluginCategories.Resize(type + 1);
        entry.m_IndexInCategory = m_PluginCategories[type].m_Entries.Size();
        switch (type) {
            case CKPLUGIN_BITMAP_READER:
            case CKPLUGIN_SOUND_READER:
            case CKPLUGIN_MODEL_READER:
            case CKPLUGIN_MOVIE_READER: {
                entry.m_ReadersInfo = new CKPluginEntryReadersData;
                entry.m_ReadersInfo->m_GetReaderFct = (CKReaderGetReaderFunction) shl.GetFunctionPtr("CKGetReader");
                break;
            }
            case CKPLUGIN_BEHAVIOR_DLL: {
                entry.m_BehaviorsInfo = new CKPluginEntryBehaviorsData;
                auto registerBehaviorDeclarationsFunc = (CKDLL_OBJECTDECLARATIONFUNCTION) shl.GetFunctionPtr("RegisterBehaviorDeclarations");
                if (!registerBehaviorDeclarationsFunc)
                    registerBehaviorDeclarationsFunc = (CKDLL_OBJECTDECLARATIONFUNCTION) shl.GetFunctionPtr("RegisterNEMOExtensions");
                InitializeBehaviors(registerBehaviorDeclarationsFunc, entry);
                break;
            }
            default:
                break;
        }

        m_PluginCategories[type].m_Entries.PushBack(pEntry);
        if (!g_Contextes.IsEmpty()) {
            pluginEntries.PushBack(entry);
        }

        err = CK_OK;
    }

    if (err != CK_OK) {
        shl.ReleaseLibrary();
        return err;
    }

    m_PluginDlls.PushBack(pluginDll);
    if (!g_Contextes.IsEmpty()) {
        for (XArray<CKContext *>::Iterator cit = g_Contextes.Begin(); cit != g_Contextes.End(); ++cit) {
            for (XClassArray<CKPluginEntry>::Iterator eit = pluginEntries.Begin(); eit != pluginEntries.End(); ++eit) {
                InitInstancePluginEntry(&*eit, *cit);
            }
        }
    }

    return err;
}

CKPluginEntry *CKPluginManager::FindComponent(CKGUID Component, int catIdx) {
    if (catIdx < 0) {
        for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
            for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
                if ((*eit)->m_PluginInfo.m_GUID == Component) {
                    return *eit;
                }
            }
        }

        return nullptr;
    }

    if (catIdx >= m_PluginCategories.Size())
        return nullptr;

    CKPluginCategory &cat = m_PluginCategories[catIdx];
    for (XArray<CKPluginEntry *>::Iterator eit = cat.m_Entries.Begin(); eit != cat.m_Entries.End(); ++eit) {
        if ((*eit)->m_PluginInfo.m_GUID == Component) {
            return *eit;
        }
    }

    if (catIdx != CKPLUGIN_BEHAVIOR_DLL)
        return nullptr;

    CKObjectDeclaration *decl = CKGetObjectDeclarationFromGuid(Component);
    if (!decl)
        return nullptr;

    return GetPluginInfo(CKPLUGIN_BEHAVIOR_DLL, decl->m_PluginIndex);
}

int CKPluginManager::AddCategory(CKSTRING cat) {
    if (!cat || strlen(cat) == 0)
        return -1;

    int idx = GetCategoryIndex(cat);
    if (idx == -1) {
        CKPluginCategory category;
        category.m_Name = cat;
        m_PluginCategories.PushBack(category);
        idx = m_PluginCategories.Size() - 1;
    }

    return idx;
}

CKERROR CKPluginManager::RemoveCategory(int catIdx) {
    if (catIdx < 0 || catIdx >= m_PluginCategories.Size())
        return CKERR_INVALIDPARAMETER;

    CKPluginCategory &cat = m_PluginCategories[catIdx];
    if (!cat.m_Entries.IsEmpty()) {
        for (XArray<CKPluginEntry *>::Iterator eit = cat.m_Entries.Begin(); eit != cat.m_Entries.End(); ++eit) {
            if (*eit) {
                delete (*eit);
            }
        }
    }

    m_PluginCategories.RemoveAt(catIdx);
    return CK_OK;
}

int CKPluginManager::GetCategoryCount() {
    return m_PluginCategories.Size();
}

CKSTRING CKPluginManager::GetCategoryName(int catIdx) {
    if (catIdx < 0 || catIdx >= m_PluginCategories.Size())
        return nullptr;
    return m_PluginCategories[catIdx].m_Name.Str();
}

int CKPluginManager::GetCategoryIndex(CKSTRING cat) {
    if (m_PluginCategories.Size() == 0)
        return -1;

    for (int i = 0; i < m_PluginCategories.Size(); ++i) {
        if (m_PluginCategories[i].m_Name.Compare(cat) == 0)
            return i;
    }

    return -1;
}

CKERROR CKPluginManager::RenameCategory(int catIdx, CKSTRING newName) {
    if (catIdx < 0 || catIdx >= m_PluginCategories.Size())
        return CKERR_INVALIDPARAMETER;

    m_PluginCategories[catIdx].m_Name = newName;
    return CK_OK;
}

int CKPluginManager::GetPluginDllCount() {
    return m_PluginDlls.Size();
}

CKPluginDll *CKPluginManager::GetPluginDllInfo(int PluginDllIdx) {
    return &m_PluginDlls[PluginDllIdx];
}

CKPluginDll *CKPluginManager::GetPluginDllInfo(CKSTRING PluginName, int *idx) {
    XString path(PluginName);
    auto i = m_PluginDlls.Begin();
    for (; i != m_PluginDlls.End(); ++i) {
        if (path.Compare(i->m_DllFileName) == 0)
            break;
    }

    if (i == m_PluginDlls.End()) {
        return nullptr;
    }

    if (idx) {
        *idx = i - m_PluginDlls.Begin();
    }
    return i;
}

CKERROR CKPluginManager::UnLoadPluginDll(int PluginDllIdx) {
    CKPluginDll *pluginDll = GetPluginDllInfo(PluginDllIdx);
    if (!pluginDll || !pluginDll->m_DllInstance)
        return CKERR_INVALIDPARAMETER;

    for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
        int index = 0;
        for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
            CKPluginEntry *entry = *eit;
            if (entry->m_PluginDllIndex == PluginDllIdx) {
                auto &pluginInfo = entry->m_PluginInfo;
                if (pluginInfo.m_Type == CKPLUGIN_BEHAVIOR_DLL) {
                    RemoveBehaviors(index);
                }

                for (XArray<CKContext *>::Iterator tit = g_Contextes.Begin(); tit != g_Contextes.End(); ++tit) {
                    ExitInstancePluginEntry(entry, *tit);
                }

                pluginInfo.m_InitInstanceFct = nullptr;
                pluginInfo.m_ExitInstanceFct = nullptr;
            }

            ++index;
        }
    }

    VxSharedLibrary shl;
    shl.Attach(pluginDll->m_DllInstance);
    shl.ReleaseLibrary();
    pluginDll->m_DllInstance = nullptr;

    return CK_OK;
}

CKERROR CKPluginManager::ReLoadPluginDll(int PluginDllIdx) {
    CKPluginDll *pluginDll = GetPluginDllInfo(PluginDllIdx);
    if (!pluginDll)
        return CKERR_INVALIDPARAMETER;

    VxSharedLibrary shl;
    if (!pluginDll->m_DllInstance) {
        pluginDll->m_DllInstance = shl.Load(pluginDll->m_DllFileName.Str());
        if (!pluginDll->m_DllInstance)
            return CKERR_INVALIDPARAMETER;
    }

    auto *getPluginInfoCountFunc = (CKPluginGetInfoCountFunction) shl.GetFunctionPtr("CKGetPluginInfoCount");
    auto *getPluginInfoFunc = (CKPluginGetInfoFunction) shl.GetFunctionPtr("CKGetPluginInfo");
    if (!getPluginInfoFunc)
        return CKERR_INVALIDPARAMETER;

    int pluginInfoCount = (getPluginInfoCountFunc) ? getPluginInfoCountFunc() : 1;

    for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
        for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
            CKPluginEntry *entry = *eit;
            if (entry->m_PluginDllIndex != PluginDllIdx)
                continue;

            if (entry->m_PositionInDll >= pluginInfoCount)
                continue;

            CKPluginInfo *pluginInfo = getPluginInfoFunc(entry->m_PositionInDll);
            if (!pluginInfo)
                continue;

            entry->m_PluginInfo = *pluginInfo;

            switch (pluginInfo->m_Type) {
                case CKPLUGIN_BITMAP_READER:
                case CKPLUGIN_SOUND_READER:
                case CKPLUGIN_MODEL_READER:
                case CKPLUGIN_MOVIE_READER: {
                    if (entry->m_ReadersInfo) {
                        entry->m_ReadersInfo->m_GetReaderFct = (CKReaderGetReaderFunction) shl.GetFunctionPtr("CKGetReader");
                    }
                    break;
                }
                case CKPLUGIN_BEHAVIOR_DLL: {
                    auto registerBehaviorDeclarationsFunc = (CKDLL_OBJECTDECLARATIONFUNCTION) shl.GetFunctionPtr("RegisterBehaviorDeclarations");
                    if (!registerBehaviorDeclarationsFunc)
                        registerBehaviorDeclarationsFunc = (CKDLL_OBJECTDECLARATIONFUNCTION) shl.GetFunctionPtr("RegisterNEMOExtensions");
                    InitializeBehaviors(registerBehaviorDeclarationsFunc, *entry);
                    break;
                }
                default:
                    break;
            }
        }
    }

    return CK_OK;
}

int CKPluginManager::GetPluginCount(int catIdx) {
    if (catIdx < 0 || catIdx >= m_PluginCategories.Size())
        return 0;
    return m_PluginCategories[catIdx].m_Entries.Size();
}

CKPluginEntry *CKPluginManager::GetPluginInfo(int catIdx, int PluginIdx) {
    if (catIdx < 0 || catIdx >= m_PluginCategories.Size())
        return nullptr;
    CKPluginCategory &cat = m_PluginCategories[catIdx];
    if (PluginIdx < 0 || PluginIdx >= cat.m_Entries.Size())
        return nullptr;
    return cat.m_Entries[PluginIdx];
}

CKBOOL CKPluginManager::SetReaderOptionData(CKContext *context, void *memdata, CKParameterOut *Param, CKFileExtension ext, CKGUID *guid) {
    auto *ids = (CK_ID *) Param->GetReadDataPtr();
    CKParameterType type = Param->GetType();
    CKStructStruct *desc = context->m_ParameterManager->GetStructDescByType(type);
    if (!desc)
        return FALSE;

    char *mem = (char *)memdata;
    for (int i = 0; i < desc->NbData; ++i) {
        auto *param = (CKParameter *)context->GetObject(ids[i]);
        if (param) {
            int size = param->GetDataSize();
            void *data = param->GetReadDataPtr();
            memcpy(mem, data, size);
            mem += size;
        }
    }

    return TRUE;
}

CKParameterOut *CKPluginManager::GetReaderOptionData(CKContext *context, void *memdata, CKFileExtension ext, CKGUID *guid) {
    CKPluginEntry *entry = nullptr;

    if (guid) {
        for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
            for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
                if ((*eit)->m_PluginInfo.m_GUID == *guid) {
                    entry = *eit;
                    break;
                }
            }
        }
    } else {
        for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
            for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
                if (strcmpi((*eit)->m_PluginInfo.m_Extension, ext) == 0) {
                    entry = *eit;
                    break;
                }
            }
        }
    }

    if (!entry)
        return nullptr;

    if (!entry->m_ReadersInfo)
        return nullptr;

    CKGUID paramGuid = entry->m_ReadersInfo->m_SettingsParameterGuid;
    if (paramGuid == CKGUID())
        return nullptr;

    CKParameterOut *paramOut = context->CreateCKParameterOut("Temp", paramGuid);

    auto *ids = (CK_ID *) paramOut->GetReadDataPtr();
    CKParameterType type = paramOut->GetType();
    CKStructStruct *desc = context->m_ParameterManager->GetStructDescByType(type);
    if (!desc) {
        CKDestroyObject(paramOut);
        return nullptr;
    }

    char *mem = (char *)memdata;
    for (int i = 0; i < desc->NbData; ++i) {
        auto *param = (CKParameter *)context->GetObject(ids[i]);
        if (param) {
            int size = param->GetDataSize();
            void *dst = param->GetWriteDataPtr();
            memcpy(dst, mem, size);
            mem += size;
        }
    }

    return paramOut;
}

CKBitmapReader *CKPluginManager::GetBitmapReader(CKFileExtension &ext, CKGUID *preferedGUID) {
    if (!preferedGUID)
        return (CKBitmapReader *) EXTFindReader(ext, CKPLUGIN_BITMAP_READER);
    auto *reader = (CKBitmapReader *) GUIDFindReader(*preferedGUID, CKPLUGIN_BITMAP_READER);
    if (!reader)
        reader = (CKBitmapReader *) EXTFindReader(ext, CKPLUGIN_BITMAP_READER);
    return reader;
}

CKSoundReader *CKPluginManager::GetSoundReader(CKFileExtension &ext, CKGUID *preferedGUID) {
    if (!preferedGUID)
        return (CKSoundReader *) EXTFindReader(ext, CKPLUGIN_SOUND_READER);
    auto *reader = (CKSoundReader *) GUIDFindReader(*preferedGUID, CKPLUGIN_SOUND_READER);
    if (!reader)
        reader = (CKSoundReader *) EXTFindReader(ext, CKPLUGIN_SOUND_READER);
    return reader;
}

CKModelReader *CKPluginManager::GetModelReader(CKFileExtension &ext, CKGUID *preferedGUID) {
    if (!preferedGUID)
        return (CKModelReader *) EXTFindReader(ext, CKPLUGIN_MODEL_READER);
    auto *reader = (CKModelReader *) GUIDFindReader(*preferedGUID, CKPLUGIN_MODEL_READER);
    if (!reader)
        reader = (CKModelReader *) EXTFindReader(ext, CKPLUGIN_MODEL_READER);
    return reader;
}

CKMovieReader *CKPluginManager::GetMovieReader(CKFileExtension &ext, CKGUID *preferedGUID) {
    if (!preferedGUID)
        return (CKMovieReader *) EXTFindReader(ext, CKPLUGIN_MOVIE_READER);
    auto *reader = (CKMovieReader *) GUIDFindReader(*preferedGUID, CKPLUGIN_MOVIE_READER);
    if (!reader)
        reader = (CKMovieReader *) EXTFindReader(ext, CKPLUGIN_MOVIE_READER);
    return reader;
}

CKERROR CKPluginManager::Load(CKContext *context, CKSTRING FileName, CKObjectArray *liste, CK_LOAD_FLAGS LoadFlags, CKCharacter *carac, CKGUID *Readerguid) {
    if (!FileName || !context)
        return CKERR_INVALIDPARAMETER;

    context->m_GeneralRenameOption = 0;
    context->m_MatTexturesRenameOption = 0;

    CKPathSplitter ps(FileName);
    CKFileExtension ext(ps.GetExtension());

    CKERROR err = CK_OK;
    auto *reader = GetModelReader(ext, Readerguid);
    if (reader) {
        err = reader->Load(context, FileName, liste, LoadFlags, carac);
        reader->Release();
    } else {
        err = CKERR_NOLOADPLUGINS;
    }

    context->SetAutomaticLoadMode(CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID, CKLOAD_INVALID);
    context->SetUserLoadCallback(nullptr, nullptr);

    return err;
}

CKERROR CKPluginManager::Save(CKContext *context, CKSTRING FileName, CKObjectArray *liste, CKDWORD SaveFlags, CKGUID *Readerguid) {
    if (!FileName || !context)
        return CKERR_INVALIDPARAMETER;

    CKPathSplitter ps(FileName);
    CKFileExtension ext(ps.GetExtension());

    CKERROR err = CK_OK;
    auto *reader = GetModelReader(ext, Readerguid);
    if (reader && (reader->GetFlags() & CK_DATAREADER_FILESAVE) != 0) {
        err = reader->Save(context, FileName, liste, SaveFlags);
    } else {
        err = CKERR_NOSAVEPLUGINS;
    }

    if (reader)
        reader->Release();

    return err;
}

void CKPluginManager::ReleaseAllPlugins() {
    for (XClassArray<CKPluginDll>::Iterator it = m_PluginDlls.Begin(); it != m_PluginDlls.End(); ++it) {
        if (it->m_DllInstance) {
            VxSharedLibrary sl;
            sl.Attach(it->m_DllInstance);
            sl.ReleaseLibrary();
            it->m_DllInstance = nullptr;
        }
    }

    for (XClassArray<VxSharedLibrary *>::Iterator it = m_RunTimeDlls.Begin(); it != m_RunTimeDlls.End(); ++it) {
        (*it)->ReleaseLibrary();
        delete *it;
    }

    Clean();
}

void CKPluginManager::InitializePlugins(CKContext *context) {
    for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
        for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
            InitInstancePluginEntry(*eit, context);
        }
    }
}

void CKPluginManager::ComputeDependenciesList(CKFile *file) {
    if (!file)
        return;

    CKContext *context = file->m_Context;

    for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
        for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
            CKPluginEntry *entry = *eit;
            entry->m_NeededByFile = FALSE;
        }
    }

    for (auto it = file->m_IncludedFiles.Begin(); it != file->m_IncludedFiles.End(); ++it) {
        XString &includedFile = *it;
        CKPathSplitter ps(includedFile.Str());
        CKFileExtension ext(ps.GetExtension());
        CKPluginEntry *entry = EXTFindEntry(ext, CKPLUGIN_BITMAP_READER);
        if (entry) {
            entry->m_NeededByFile = TRUE;
            continue;
        }
        entry = EXTFindEntry(ext, CKPLUGIN_SOUND_READER);
        if (entry) {
            entry->m_NeededByFile = TRUE;
            continue;
        }
        entry = EXTFindEntry(ext, CKPLUGIN_MOVIE_READER);
        if (entry) {
            entry->m_NeededByFile = TRUE;
        }
    }

    CKBOOL useGlobal = FALSE;
    XIntArray &textures = file->m_IndexByClassId[CKCID_TEXTURE];
    for (auto it = textures.Begin(); it != textures.End(); ++it) {
        CKTexture *texture = (CKTexture *) file->m_FileObjects[*it].ObjPtr;
        if (texture) {
            CKMovieInfo *movieInfo = texture->m_MovieInfo;
            if (movieInfo) {
                CKMovieReader *movieReader = movieInfo->m_MovieReader;
                if (movieReader) {
                    CKPluginInfo *info = movieReader->GetReaderInfo();
                    if (info) {
                        MarkComponentAsNeeded(info->m_GUID, info->m_Type);
                    }
                } else {
                    if (texture->m_SaveOptions == CKTEXTURE_IMAGEFORMAT) {
                        CKBitmapProperties *prop = texture->m_SaveProperties;
                        if (prop) {
                            MarkComponentAsNeeded(prop->m_ReaderGuid, CKPLUGIN_BITMAP_READER);
                        }
                    } else if (texture->m_SaveOptions == CKTEXTURE_USEGLOBAL) {
                        useGlobal = TRUE;
                    }
                }
            }
        }
    }

    XIntArray &sprites = file->m_IndexByClassId[CKCID_SPRITE];
    for (auto it = sprites.Begin(); it != sprites.End(); ++it) {
        CKSprite *sprite = (CKSprite *) file->m_FileObjects[*it].ObjPtr;
        if (sprite) {
            CKMovieReader *movieReader = sprite->GetMovieReader();
            if (movieReader) {
                CKPluginInfo *info = movieReader->GetReaderInfo();
                if (info) {
                    MarkComponentAsNeeded(info->m_GUID, info->m_Type);
                }
            } else {
                if (sprite->GetSaveOptions() == CKTEXTURE_IMAGEFORMAT) {
                    CKBitmapProperties *prop = sprite->GetSaveFormat();
                    if (prop) {
                        MarkComponentAsNeeded(prop->m_ReaderGuid, CKPLUGIN_BITMAP_READER);
                    }
                } else if (sprite->GetSaveOptions() == CKTEXTURE_USEGLOBAL) {
                    useGlobal = TRUE;
                }
            }
        }
    }

    if (useGlobal && context->GetGlobalImagesSaveOptions() == CKTEXTURE_IMAGEFORMAT) {
        CKBitmapProperties *prop = context->GetGlobalImagesSaveFormat();
        if (prop) {
            MarkComponentAsNeeded(prop->m_ReaderGuid, CKPLUGIN_BITMAP_READER);
        }
    }

    XIntArray &sounds = file->m_IndexByClassId[CKCID_WAVESOUND];
    for (auto it = sounds.Begin(); it != sounds.End(); ++it) {
        CKWaveSound *sound = (CKWaveSound *) file->m_FileObjects[*it].ObjPtr;
        if (sound) {
            CKSoundReader *soundReader = sound->GetReader();
            if (soundReader) {
                CKPluginInfo *info = soundReader->GetReaderInfo();
                if (info) {
                    MarkComponentAsNeeded(info->m_GUID, info->m_Type);
                }
            }
        }
    }

    XIntArray &paramLocal = file->m_IndexByClassId[CKCID_PARAMETERLOCAL];
    for (auto it = paramLocal.Begin(); it != paramLocal.End(); ++it) {
        CKParameterLocal *param = (CKParameterLocal *) file->m_FileObjects[*it].ObjPtr;
        if (param) {
            CKParameterTypeDesc *desc = param->GetParameterType();
            if (desc && desc->CreatorDll) {
                desc->CreatorDll->m_NeededByFile = TRUE;
            }
        }
    }

    XIntArray &paramOut = file->m_IndexByClassId[CKCID_PARAMETEROUT];
    for (auto it = paramOut.Begin(); it != paramOut.End(); ++it) {
        CKParameterOut *param = (CKParameterOut *) file->m_FileObjects[*it].ObjPtr;
        if (param) {
            CKParameterTypeDesc *desc = param->GetParameterType();
            if (desc && desc->CreatorDll) {
                desc->CreatorDll->m_NeededByFile = TRUE;
            }
        }
    }

    XIntArray &param = file->m_IndexByClassId[CKCID_PARAMETER];
    for (auto it = param.Begin(); it != param.End(); ++it) {
        CKParameter *param = (CKParameter *) file->m_FileObjects[*it].ObjPtr;
        if (param) {
            CKParameterTypeDesc *desc = param->GetParameterType();
            if (desc && desc->CreatorDll) {
                desc->CreatorDll->m_NeededByFile = TRUE;
            }
        }
    }

    CKAttributeManager *am = context->GetAttributeManager();
    for (int i = 0; i < am->GetAttributeCount(); ++i) {
        CKPluginEntry *entry = am->GetCreatorDll(i);
        if (entry) {
            if (am->m_AttributeMask.IsSet(i)) {
                entry->m_NeededByFile = TRUE;
            }
        }
    }

    XHashTable<CKObjectDeclaration *, CKGUID> decls;
    XIntArray &behaviors = file->m_IndexByClassId[CKCID_BEHAVIOR];
    for (auto it = behaviors.Begin(); it != behaviors.End(); ++it) {
        CKBehavior *beh = (CKBehavior *) file->m_FileObjects[*it].ObjPtr;
        if (beh && beh->GetFlags() & CKBEHAVIOR_BUILDINGBLOCK) {
            CKGUID guid = beh->GetPrototypeGuid();
            CKObjectDeclaration *decl = CKGetObjectDeclarationFromGuid(guid);
            if (decl) {
                decls.InsertUnique(guid, decl);

                CKPluginEntry *entry = GetPluginInfo(CKPLUGIN_BEHAVIOR_DLL, decl->m_PluginIndex);
                if (entry) {
                    entry->m_NeededByFile = TRUE;
                }

                XArray<CKGUID> &managers = decl->m_ManagersGuid;
                for (XArray<CKGUID>::Iterator mit = managers.Begin(); mit != managers.End(); ++mit) {
                    MarkComponentAsNeeded(*mit, CKPLUGIN_MANAGER_DLL);
                }
            }
        }
    }

    XArray<CKFileManagerData> &managersData = file->m_ManagersData;
    for (XArray<CKFileManagerData>::Iterator mit = managersData.Begin(); mit != managersData.End(); ++mit) {
        MarkComponentAsNeeded(mit->Manager, CKPLUGIN_MANAGER_DLL);
    }

    file->m_PluginsDep.Clear();
    for (int i = 0; i < m_PluginCategories.Size(); ++i) {
        CKPluginCategory &cat = m_PluginCategories[i];

        CKFilePluginDependencies deps;
        deps.m_PluginCategory = i;

        for (XArray<CKPluginEntry *>::Iterator eit = cat.m_Entries.Begin(); eit != cat.m_Entries.End(); ++eit) {
            CKPluginEntry *entry = *eit;
            if (entry->m_NeededByFile) {
                deps.m_Guids.PushBack(entry->m_PluginInfo.m_GUID);
            }
        }

        int nameCmp = cat.m_Name.ICompare("BuildingBlocks");
        if (nameCmp == 0) {
            for (XHashTable<CKObjectDeclaration *, CKGUID>::Iterator dit = decls.Begin(); dit != decls.End(); ++dit) {
                deps.m_Guids.PushBack(dit.GetKey());
            }
        }

        if (deps.m_Guids.Size() > 0) {
            file->m_PluginsDep.PushBack(deps);
        }
    }
}

void CKPluginManager::MarkComponentAsNeeded(CKGUID Component, int catIdx) {
    if (catIdx >= 0) {
        auto &category = m_PluginCategories[catIdx];
        for (XArray<CKPluginEntry *>::Iterator it = category.m_Entries.Begin(); it != category.m_Entries.End(); ++it) {
            if ((*it)->m_PluginInfo.m_GUID == Component)
                (*it)->m_NeededByFile = TRUE;
        }
    } else {
        for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
            for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
                if ((*eit)->m_PluginInfo.m_GUID == Component)
                    (*eit)->m_NeededByFile = TRUE;
            }
        }
    }
}

void CKPluginManager::InitInstancePluginEntry(CKPluginEntry *entry, CKContext *context) {
    auto &pluginInfo = entry->m_PluginInfo;

    if ((context->m_StartOptions & CK_CONFIG_DISABLEDINPUT) != 0 && pluginInfo.m_GUID == INPUT_MANAGER_GUID)
        return;

    g_TheCurrentPluginEntry = entry;

    if (pluginInfo.m_InitInstanceFct &&
        (pluginInfo.m_Type != CKPLUGIN_RENDERENGINE_DLL ||
         context->m_SelectedRenderEngine == entry->m_IndexInCategory)) {
        pluginInfo.m_InitInstanceFct(context);
    }

    auto pluginType = pluginInfo.m_Type;
    if (pluginType == CKPLUGIN_MANAGER_DLL) {
        entry->m_Active = context->m_ManagerTable.IsHere(pluginInfo.m_GUID);
    } else if (pluginType == CKPLUGIN_RENDERENGINE_DLL) {
        entry->m_Active = entry->m_IndexInCategory == context->m_SelectedRenderEngine;
    } else {
        entry->m_Active = TRUE;
    }

    auto &readerInfo = entry->m_ReadersInfo;
    if (!readerInfo) {
        g_TheCurrentPluginEntry = nullptr;
        return;
    }

    auto *readerFct = readerInfo->m_GetReaderFct;
    if (!readerFct) {
        g_TheCurrentPluginEntry = nullptr;
        return;
    }

    auto *reader = readerFct(entry->m_PositionInDll);
    if (!reader) {
        g_TheCurrentPluginEntry = nullptr;
        return;
    }

    int optionCount = reader->GetOptionsCount();
    readerInfo->m_OptionCount = optionCount;
    readerInfo->m_ReaderFlags = reader->GetFlags();

    if (optionCount == 0) {
        g_TheCurrentPluginEntry = nullptr;
        reader->Release();
        return;
    }

    CKParameterManager *pm = context->m_ParameterManager;

    XArray<CKGUID> optionGuids(optionCount);

    size_t optionNamesLength = 0;
    auto *optionNames = new CKSTRING[optionCount];
    memset(optionNames, 0, sizeof(CKSTRING) * optionCount);

    for (int i = 0; i < optionCount; ++i) {
        CKSTRING desc = CKStrdup(reader->GetOptionDescription(i));
        char *name = strchr(desc, ':');
        if (!name)
            continue;

        *name = '\0';
        ++name;

        char *data = strchr(name, ':');
        if (data) {
            *data = '\0';
            ++data;
        }

        CKGUID guid;

        if (stricmp(desc, "Flags") == 0) {
            guid = context->GetSecureGuid();
            pm->RegisterNewFlags(guid, name, data);
        } else if (stricmp(desc, "Enum") == 0) {
            guid = context->GetSecureGuid();
            pm->RegisterNewEnum(guid, name, data);
            CKParameterTypeDesc *paramTypeDesc = pm->GetParameterTypeDescription(guid);
            if (paramTypeDesc)
                paramTypeDesc->dwFlags |= CKPARAMETERTYPE_HIDDEN;
        } else {
            guid = pm->ParameterNameToGuid(desc);
        }

        optionGuids.PushBack(guid);

        optionNames[i] = CKStrdup(name);
        optionNamesLength += strlen(name) + 2;

        delete[] desc;
    }

    auto *data = new char[optionNamesLength];
    strcpy(data, "");
    for (int i = 0; i < optionCount; ++i) {
        strcat(data, optionNames[i]);
        strcat(data, ",");
    }
    data[strlen(data) - 1] = '\0';

    XString name = pluginInfo.m_Description;
    name << " Options Parameter";
    readerInfo->m_SettingsParameterGuid = context->GetSecureGuid();

    pm->RegisterNewStructure(readerInfo->m_SettingsParameterGuid, name.Str(), data, optionGuids);
    CKParameterTypeDesc *paramTypeDesc = pm->GetParameterTypeDescription(readerInfo->m_SettingsParameterGuid);
    if (paramTypeDesc)
        paramTypeDesc->dwFlags |= CKPARAMETERTYPE_HIDDEN;

    for (int i = 0; i < optionCount; ++i) {
        delete[] optionNames[i];
    }
    memset(optionNames, 0, sizeof(CKSTRING) * optionCount);
    delete[] optionNames;
    delete[] data;

    g_TheCurrentPluginEntry = nullptr;
    reader->Release();
}

void CKPluginManager::ExitInstancePluginEntry(CKPluginEntry *entry, CKContext *context) {
    if (!entry || !context)
        return;

    auto &pluginInfo = entry->m_PluginInfo;
    if (pluginInfo.m_ExitInstanceFct)
        pluginInfo.m_ExitInstanceFct(context);

    auto &readerInfo = entry->m_ReadersInfo;
    if (!readerInfo || readerInfo->m_SettingsParameterGuid == CKGUID())
        return;

    CKParameterManager *pm = context->m_ParameterManager;
    CKParameterType type = pm->ParameterGuidToType(readerInfo->m_SettingsParameterGuid);
    CKStructStruct *structDesc = pm->GetStructDescByType(type);
    if (!structDesc)
        return;

    XArray<CKGUID> guids;
    guids.PushBack(readerInfo->m_SettingsParameterGuid);

    for (int i = 0; i < structDesc->NbData; ++i) {
        CKParameterTypeDesc *paramTypeDesc = pm->GetParameterTypeDescription(structDesc->Guids[i]);
        if ((paramTypeDesc->dwFlags & (CKPARAMETERTYPE_FLAGS | CKPARAMETERTYPE_ENUMS)) != 0) {
            guids.PushBack(structDesc->Guids[i]);
        }
    }

    for (XArray<CKGUID>::Iterator it = guids.Begin(); it != guids.End(); ++it) {
        pm->UnRegisterParameterType(*it);
    }

    readerInfo->m_SettingsParameterGuid = CKGUID();
}

void CKPluginManager::InitializeBehaviors(CKDLL_OBJECTDECLARATIONFUNCTION Fct, CKPluginEntry &entry) {
    if (!Fct)
        return;

    XObjectDeclarationArray array;
    Fct(&array);

    for (XObjectDeclarationArray::Iterator it = array.Begin(); it != array.End(); ++it) {
        CKObjectDeclaration *decl = *it;
        decl->m_PluginIndex = entry.m_IndexInCategory;
        if (decl->m_Type == CKDLL_BEHAVIORPROTOTYPE) {
            CKGUID guid = decl->GetGuid();
            auto pair = g_PrototypeDeclarationList.TestInsert(guid, decl);
            if (pair.m_New) {
                entry.m_BehaviorsInfo->m_BehaviorsGUID.PushBack(guid);
            } else {
                CKObjectDeclaration *decl2 = CKGetObjectDeclarationFromGuid(guid);
                if (decl2 && decl2 != decl) {
                    if (strcmp(decl->GetName(), decl2->GetName()) == 0 && decl->GetVersion() > decl2->GetVersion()) {
                        *decl2 = *decl;
                    }
                    delete decl;
                }
            }
        }
    }
}

void CKPluginManager::RemoveBehaviors(int PluginIndex) {
    XObjectDeclarationArray array;
    for (XObjDeclHashTable::Iterator it = g_PrototypeDeclarationList.Begin();
         it != g_PrototypeDeclarationList.End(); ++it) {
        CKObjectDeclaration *decl = *it;
        if (decl && decl->GetPluginIndex() == PluginIndex)
            array.PushBack(decl);
    }

    for (XObjectDeclarationArray::Iterator it = array.Begin(); it != array.End(); ++it) {
        CKRemovePrototypeDeclaration(*it);
    }
}

CKPluginEntry *CKPluginManager::EXTFindEntry(CKFileExtension &ext, int Category) {
    if (Category >= 0) {
        auto &category = m_PluginCategories[Category];
        for (XArray<CKPluginEntry *>::Iterator it = category.m_Entries.Begin(); it != category.m_Entries.End(); ++it) {
            CKPluginEntry *entry = *it;
            if (strcmpi(entry->m_PluginInfo.m_Extension, ext) == 0)
                return entry;
        }
    } else {
        for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
            for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
                CKPluginEntry *entry = *eit;
                if (strcmpi(entry->m_PluginInfo.m_Extension, ext) == 0)
                    return entry;
            }
        }
    }

    return nullptr;
}

CKDataReader *CKPluginManager::EXTFindReader(CKFileExtension &ext, int Category) {
    CKPluginEntry *entry = nullptr;

    if (Category >= 0) {
        auto &category = m_PluginCategories[Category];
        for (XArray<CKPluginEntry *>::Iterator it = category.m_Entries.Begin(); it != category.m_Entries.End(); ++it) {
            if (strcmpi((*it)->m_PluginInfo.m_Extension, ext) == 0) {
                entry = *it;
                break;
            }
        }
    } else {
        for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
            for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
                if (strcmpi((*eit)->m_PluginInfo.m_Extension, ext) == 0) {
                    entry = *eit;
                    break;
                }
            }
            if (entry)
                break;
        }
    }

    if (!entry)
        return nullptr;

    if (!entry->m_ReadersInfo)
        return nullptr;

    auto *fct = entry->m_ReadersInfo->m_GetReaderFct;
    return (fct) ? fct(entry->m_PositionInDll) : nullptr;
}

CKDataReader *CKPluginManager::GUIDFindReader(CKGUID &guid, int Category) {
    CKPluginEntry *entry = nullptr;

    if (Category >= 0) {
        auto &category = m_PluginCategories[Category];
        for (XArray<CKPluginEntry *>::Iterator it = category.m_Entries.Begin(); it != category.m_Entries.End(); ++it) {
            if ((*it)->m_PluginInfo.m_GUID == guid) {
                entry = *it;
                break;
            }
        }
    } else {
        for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
            for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
                if ((*eit)->m_PluginInfo.m_GUID == guid) {
                    entry = *eit;
                    break;
                }
            }
            if (entry)
                break;
        }
    }

    if (!entry)
        return nullptr;

    auto &pluginDll = m_PluginDlls[entry->m_PluginDllIndex];
    VxSharedLibrary shl;
    shl.Attach(pluginDll.m_DllInstance);
    auto *fct = (CKReaderGetReaderFunction) shl.GetFunctionPtr("CKGetReader");
    return (fct) ? fct(entry->m_PositionInDll) : nullptr;
}

void CKPluginManager::Clean() {
    for (XClassArray<CKPluginCategory>::Iterator cit = m_PluginCategories.Begin(); cit != m_PluginCategories.End(); ++cit) {
        for (XArray<CKPluginEntry *>::Iterator eit = cit->m_Entries.Begin(); eit != cit->m_Entries.End(); ++eit) {
            if (*eit) {
                delete *eit;
            }
        }
    }

    m_PluginCategories.Clear();
    m_PluginDlls.Clear();
    m_RunTimeDlls.Clear();
}