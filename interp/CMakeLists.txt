cmake_minimum_required(VERSION 3.16)
project(tfi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB TFI_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(tfi STATIC ${TFI_SOURCES})
target_include_directories(tfi PUBLIC include)
target_compile_options(tfi PRIVATE -Wall -Wextra)
find_package(Threads REQUIRED)
target_link_libraries(tfi PUBLIC Threads::Threads)

if(APPLE)
  find_library(ACCELERATE Accelerate)
  if(ACCELERATE)
    target_compile_definitions(tfi PRIVATE TFI_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(tfi PUBLIC ${ACCELERATE})
  endif()
endif()

add_executable(tfi_cli tools/tfi.cpp)
set_target_properties(tfi_cli PROPERTIES OUTPUT_NAME tfi)
target_link_libraries(tfi_cli PRIVATE tfi)

add_executable(tfi_golden tests/golden.cpp)
target_link_libraries(tfi_golden PRIVATE tfi)

add_executable(tfi_tests tests/unit.cpp)
target_link_libraries(tfi_tests PRIVATE tfi)

enable_testing()
add_test(NAME unit COMMAND tfi_tests)
