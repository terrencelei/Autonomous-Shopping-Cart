cmake_minimum_required(VERSION 3.16)
project(cart_tracker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# ── Darknet ──────────────────────────────────────────────────────────────────
set(DARKNET_ROOT "$ENV{HOME}/Desktop/darknet" CACHE PATH "Darknet source root")
set(DARKNET_BUILD "${DARKNET_ROOT}/build/src-lib")

add_library(darknet SHARED IMPORTED)
set_target_properties(darknet PROPERTIES
    IMPORTED_LOCATION "${DARKNET_BUILD}/libdarknet.dylib"
    INTERFACE_INCLUDE_DIRECTORIES "${DARKNET_ROOT}/src-lib"
)

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED HINTS /opt/homebrew/opt/opencv/lib/cmake/opencv4)

# ── Target ───────────────────────────────────────────────────────────────────
add_executable(cart_tracker cart_tracker.cpp)

target_link_libraries(cart_tracker PRIVATE
    darknet
    ${OpenCV_LIBS}
)

# Embed the darknet dylib rpath so the binary finds it without env vars
set_target_properties(cart_tracker PROPERTIES
    INSTALL_RPATH "${DARKNET_ROOT}/build/src-lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
