include(FetchContent)

set(EIGEN_BUILD_DOC              OFF CACHE BOOL   "" FORCE)
set(EIGEN_BUILD_TESTING          OFF CACHE BOOL   "" FORCE)
set(EIGEN_BUILD_PKGCONFIG        OFF CACHE BOOL   "" FORCE)
set(EIGEN_BUILD_BLAS             OFF CACHE BOOL   "" FORCE)
set(EIGEN_BUILD_LAPACK           OFF CACHE BOOL   "" FORCE)
set(EIGEN_BUILD_DEMOS            OFF CACHE BOOL   "" FORCE)
set(BUILD_TESTING                OFF CACHE BOOL   "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM  3.5 CACHE STRING "" FORCE)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
    SYSTEM
)

FetchContent_MakeAvailable(Eigen3)
