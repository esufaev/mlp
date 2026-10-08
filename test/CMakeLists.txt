include(${CMAKE_SOURCE_DIR}/cmake/Catch2.cmake)

set(TEST_SOURCES
    test_random.cpp
    test_csv.cpp
    test_preprocessing.cpp
    test_autograd.cpp
    test_activations.cpp
    test_dense.cpp
    test_loss.cpp
    test_metrics.cpp
    test_network.cpp
    test_model_io.cpp
    test_architectures.cpp
)

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE mlp Catch2::Catch2WithMain)
target_compile_features(tests PRIVATE cxx_std_23)
target_compile_definitions(tests PRIVATE MLP_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

set_target_properties(tests
    PROPERTIES
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
)

include(CTest)
include(Catch)
catch_discover_tests(tests)

set(CLI_SCRATCH ${CMAKE_CURRENT_BINARY_DIR}/cli)

function(add_cli_test name)
    add_test(NAME "cli: ${name}"
             COMMAND $<TARGET_FILE:mlp_cli> ${ARGN}
             WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endfunction()

add_cli_test("help exits cleanly" help)
add_cli_test("no arguments prints the usage and fails")
add_cli_test("an unknown command is rejected" bogus)
add_cli_test("a required option cannot be omitted" describe)
add_cli_test("a typo in an option name is caught, not ignored"
             train --dataset data/data.csv --epocs 84)
add_cli_test("malformed input is rejected"
             train --dataset data/data.csv --epochs abc)
add_cli_test("a negative count is rejected"
             train --dataset data/data.csv --epochs -5)
add_cli_test("a missing dataset is reported, not crashed on"
             describe --dataset no/such.csv)
add_cli_test("describe reads the assignment's dataset" describe --dataset data/data.csv)
add_cli_test("the assignment's invocation parses"
             train --dataset data/data.csv --layer 24 24 24 --epochs 1 --batch_size 8
             --learning_rate 0.0314 --loss binaryCrossentropy --seed 42 --quiet
             --out ${CLI_SCRATCH}/model.mlp --plot-dir ${CLI_SCRATCH}/plots)
add_cli_test("negative numbers are values, not options"
             train --dataset data/data.csv --learning_rate -0.5 --epochs 1
             --out ${CLI_SCRATCH}/unused.mlp --plot-dir ${CLI_SCRATCH}/unused)

set_tests_properties(
    "cli: no arguments prints the usage and fails"
    "cli: an unknown command is rejected"
    "cli: a required option cannot be omitted"
    "cli: a typo in an option name is caught, not ignored"
    "cli: malformed input is rejected"
    "cli: a negative count is rejected"
    "cli: a missing dataset is reported, not crashed on"
    PROPERTIES WILL_FAIL TRUE)

set_tests_properties("cli: describe reads the assignment's dataset"
    PROPERTIES PASS_REGULAR_EXPRESSION "samples: 569, features: 30")

set_tests_properties("cli: the assignment's invocation parses"
    PROPERTIES PASS_REGULAR_EXPRESSION "total trainable parameters +1994")

set_tests_properties("cli: negative numbers are values, not options"
    PROPERTIES PASS_REGULAR_EXPRESSION "learning rate -0\\.5 must be positive")
