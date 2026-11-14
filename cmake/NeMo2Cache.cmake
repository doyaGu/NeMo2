function(nemo_set_cache_default var value type doc)
    if (NOT DEFINED ${var})
        set(${var} "${value}" CACHE ${type} "${doc}")
    endif ()
endfunction()

# A derived default is a normal variable, not persistent user intent. Child
# option() calls honor it through CMP0077 without seeding false cache overrides.
function(nemo_test_option var default enabled)
    if (DEFINED ${var})
        set(value "${${var}}")
    else ()
        set(value "${default}")
    endif ()
    if (NOT enabled)
        set(value OFF)
    endif ()
    set(${var} "${value}" PARENT_SCOPE)
endfunction()
