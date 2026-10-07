# Feature dumper + its sampler unit test (included from the top level).

# The reference for the sample_step unit test: yue2.cpp's own sample_step (with
# SampleScratch, scan_allowed, by_score_*), cut verbatim out of stage_ar.cpp at
# configure time. A change to that file re-runs cmake.
set(Y2M_STAGE_AR ${YUE2_CPP_DIR}/src/stage_ar.cpp)
if(NOT EXISTS ${Y2M_STAGE_AR})
	message(FATAL_ERROR "yue2mtp-dump needs a yue2.cpp checkout: ${Y2M_STAGE_AR} not found. "
		"Run git submodule update --init yue2.cpp, pass -DYUE2_CPP_DIR=/path/to/yue2.cpp, or -DYUE2_EAGLE3_BUILD_DUMP=OFF to build the trainer only.")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${Y2M_STAGE_AR})
file(READ ${Y2M_STAGE_AR} Y2M_AR_SRC)
string(FIND "${Y2M_AR_SRC}" "// Sampling order for the candidate list" Y2M_A)
string(FIND "${Y2M_AR_SRC}" "// LLAMA_BUILD_COMMON is OFF" Y2M_B)
if(Y2M_A EQUAL -1 OR Y2M_B EQUAL -1 OR NOT Y2M_B GREATER Y2M_A)
	message(FATAL_ERROR "cannot find sample_step in ${Y2M_STAGE_AR} (markers moved?)")
endif()
math(EXPR Y2M_LEN "${Y2M_B} - ${Y2M_A}")
string(SUBSTRING "${Y2M_AR_SRC}" ${Y2M_A} ${Y2M_LEN} Y2M_SAMPLER)
string(REPLACE "#include \"sampler_ref.hpp\"" "" Y2M_SAMPLER "${Y2M_SAMPLER}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/y2_sample_step.inc "${Y2M_SAMPLER}")

add_library(y2m_ref STATIC ref_sample_step.cpp)
target_include_directories(y2m_ref PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)

set(Y2M_WARN -Wall -Wextra)

add_executable(yue2mtp-dump yue2mtp-dump.cpp)
target_include_directories(yue2mtp-dump PRIVATE ${Y2_LLAMA}/src)
target_link_libraries(yue2mtp-dump PRIVATE llama ggml y2m_shaped y2m_ref vendor::hash vendor::nlohmann pthread)
target_compile_options(yue2mtp-dump PRIVATE ${Y2M_WARN})
target_compile_definitions(yue2mtp-dump PRIVATE Y2M_TOOLS_DIR="${Y2_TOOLS}")

add_executable(shaped-dist-test shaped_dist_test.cpp)
target_link_libraries(shaped-dist-test PRIVATE y2m_shaped y2m_ref)
target_compile_options(shaped-dist-test PRIVATE ${Y2M_WARN})
