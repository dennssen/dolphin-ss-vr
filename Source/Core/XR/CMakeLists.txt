add_library(xr STATIC
  XRSession.cpp
  XRSession.h
)

target_link_libraries(xr PUBLIC
  common
  OpenXR::openxr_loader
  videocommon
)

target_include_directories(xr
PUBLIC
  ${CMAKE_SOURCE_DIR}/Externals/Vulkan-Headers/include
)
