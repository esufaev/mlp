include(FetchContent)

set(CLI11_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(CLI11_INSTALL        OFF CACHE BOOL "" FORCE)
set(CLI11_PRECOMPILED    OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.7.2
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(CLI11)
