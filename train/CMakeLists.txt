# Trainer, validation tool and GGUF round-trip check (included from the top level).

set(Y2T_WARN -Wall -Wextra -Wno-unused-parameter)

add_library(y2t STATIC model.cpp graph.cpp cache.cpp)
target_include_directories(y2t PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(y2t PUBLIC ggml vendor::nlohmann pthread)
target_compile_options(y2t PRIVATE ${Y2T_WARN})
if(GGML_VULKAN)
	target_compile_definitions(y2t PUBLIC Y2T_VULKAN)
endif()

add_executable(yue2mtp-train train.cpp)
target_link_libraries(yue2mtp-train PRIVATE y2t)
target_compile_options(yue2mtp-train PRIVATE ${Y2T_WARN})

add_executable(yue2mtp-check check.cpp)
target_link_libraries(yue2mtp-check PRIVATE y2t)
target_compile_options(yue2mtp-check PRIVATE ${Y2T_WARN})

add_executable(yue2mtp-roundtrip roundtrip.cpp)
target_include_directories(yue2mtp-roundtrip PRIVATE ${Y2_LLAMA}/src)
target_link_libraries(yue2mtp-roundtrip PRIVATE y2t llama llama-common)
target_compile_options(yue2mtp-roundtrip PRIVATE ${Y2T_WARN})
