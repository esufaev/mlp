include(FetchContent)

set(MATPLOTPP_BUILD_EXAMPLES                  OFF CACHE BOOL "" FORCE)
set(MATPLOTPP_BUILD_TESTS                     OFF CACHE BOOL "" FORCE)
set(MATPLOTPP_BUILD_SHARED_LIBS               OFF CACHE BOOL "" FORCE)
set(MATPLOTPP_BUILD_HIGH_RESOLUTION_WORLD_MAP OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES                            OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING                             OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM              3.5 CACHE STRING "" FORCE)

FetchContent_Declare(
    matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
    GIT_TAG        v1.2.2
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(matplotplusplus)
