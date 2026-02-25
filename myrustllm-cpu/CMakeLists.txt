cmake_minimum_required(VERSION 3.20)
project(myrustllm-cpu VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED)

# Collect src
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")

# Create shared lib
add_library(${PROJECT_NAME} SHARED
    ${CPP_SOURCES}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
    PREFIX "lib"
)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

# Install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)