include(FetchContent)

FetchContent_Declare(
    rapidcsv
    GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git
    GIT_TAG        v9.07
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(rapidcsv)
