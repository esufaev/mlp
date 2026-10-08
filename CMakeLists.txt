cmake_minimum_required(VERSION 3.28)

project(
    mlp
    VERSION 0.1.0
    DESCRIPTION "A minimal from-scratch multilayer perceptron library"
    LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(MLP_BUILD_TESTS    "Enable unit testing"                     ON )
option(MLP_CHECK_WARNINGS "Strict compiler checks"                  ON )
option(MLP_WERROR         "Treat warnings as errors"                OFF)
option(MLP_SANITIZERS     "Build with address/undefined sanitizers" OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/CompilerFlags.cmake)
apply_compiler_flags()

if(MLP_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)
target_include_directories(${PROJECT_NAME}
    INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

find_package(fmt QUIET)
if(NOT fmt_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        11.0.2
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(fmt)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/Matplot.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/Eigen.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/RapidCsv.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CLI11.cmake)

target_link_libraries(${PROJECT_NAME} INTERFACE fmt::fmt Matplot++::matplot rapidcsv Eigen3::Eigen)

add_executable(${PROJECT_NAME}_cli apps/main.cpp)
set_target_properties(${PROJECT_NAME}_cli PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_cli PRIVATE ${PROJECT_NAME} CLI11::CLI11)
apply_warning_flags_to_targets(${PROJECT_NAME}_cli)

install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_cli
    RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(MLP_BUILD_TESTS AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(test)
endif()
