set(GNU_CLANG_FLAGS
    -finput-charset=UTF-8
    -fexec-charset=UTF-8
    -fexceptions
)

set(GNU_CLANG_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wcast-align
    -Wcast-qual
    -Wconversion
    -Wctor-dtor-privacy
    -Wenum-compare
    -Wfloat-equal
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Woverloaded-virtual
    -Wredundant-decls
    -Wsign-conversion
    -Wsign-promo
)

set(MSVC_FLAGS
    /utf-8
    /permissive-
    /EHsc
)

set(MSVC_WARNING_FLAGS
    /W4
    /w14242 /w14254 /w14263 /w14265 /w14287 /w14296
    /w14311 /w14545 /w14546 /w14547 /w14549 /w14555
    /w14619 /w14640 /w14826 /w14905 /w14906 /w14928
)

function(apply_compiler_flags)
    if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        add_compile_options(${GNU_CLANG_FLAGS})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(${MSVC_FLAGS})
    endif()
endfunction()

function(apply_warning_flags_to_targets)
    if(NOT MLP_CHECK_WARNINGS)
        return()
    endif()

    if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        set(WARNING_FLAGS ${GNU_CLANG_WARNING_FLAGS})
        if(MLP_WERROR)
            list(APPEND WARNING_FLAGS -Werror)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(WARNING_FLAGS ${MSVC_WARNING_FLAGS})
        if(MLP_WERROR)
            list(APPEND WARNING_FLAGS /WX)
        endif()
    else()
        return()
    endif()

    foreach(target IN LISTS ARGV)
        target_compile_options(${target} PRIVATE ${WARNING_FLAGS})
    endforeach()
endfunction()
