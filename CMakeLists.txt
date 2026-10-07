cmake_minimum_required(VERSION 3.16)
project(yue2-eagle3-train LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

option(GGML_VULKAN "Build the ggml Vulkan backend" ON)

# ggml + libllama come from the llama.cpp submodule (engival/llama.cpp, branch
# feat/eagle3-train: CONCAT backward + F32 MUL_MAT backward, see README).
# llama.cpp's common library is only for yue2mtp-roundtrip.
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(Y2_LLAMA ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)
set(Y2_TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/tools)

# YuE2's semantic-phase sampler distribution, shared by the dumper and its test.
add_library(y2m_shaped STATIC ${Y2_TOOLS}/shaped_dist.cpp)
target_include_directories(y2m_shaped PUBLIC ${Y2_TOOLS})

add_subdirectory(train)

# The dumper checks its sampler port against yue2.cpp's own sample_step, cut out
# of a yue2.cpp checkout at configure time (dump/CMakeLists.txt).
option(YUE2_EAGLE3_BUILD_DUMP "Build yue2mtp-dump (needs a yue2.cpp checkout, YUE2_CPP_DIR)" ON)
set(YUE2_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../yue2.cpp" CACHE PATH "yue2.cpp source checkout")
if(YUE2_EAGLE3_BUILD_DUMP)
	add_subdirectory(dump)
endif()
