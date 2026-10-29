#pragma once

#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>

#include "vulkan/vulkan_core.h"

namespace XR
{
  bool Init();
  void Shutdown();
  bool IsActive();

  VkResult CreateVulkanInstance(const VkInstanceCreateInfo* info,
                                PFN_vkGetInstanceProcAddr get_proc, VkInstance* out_instance);
  VkPhysicalDevice GetVulkanPhysicalDevice(VkInstance instance);
  VkResult CreateVulkanDevice(VkPhysicalDevice physical_device, const VkDeviceCreateInfo* info,
                              PFN_vkGetInstanceProcAddr get_proc, VkDevice* out_device);
  bool CreateSession(VkInstance vk_instance, VkPhysicalDevice physical_device, VkDevice device,
                     uint32_t queue_family_index, uint32_t queue_index);
  void DestroySession();
  void PollEvents();
  bool BeginFrame();
  VkImage GetCurrentImage();
  VkImage AcquireUIImage(uint32_t width, uint32_t height);
  void EndFrame();
  bool CreateSwapchain(uint32_t width, uint32_t height);
  uint32_t GetSwapchainWidth();
  uint32_t GetSwapchainHeight();

  struct EyeData
  {
    std::array<float, 16> view; // row-major: game camera space -> this eye's space, in game units
    float tan_left, tan_right, tan_up, tan_down;
  };

  bool GetEyeData(EyeData out[2]);
  void Recenter();
  uint64_t GetPoseGeneration();
  void LatchPose();

  void SetCurrentDrawIsUi(bool is_ui);
  bool CurrentDrawIsUi();
  int CurrentDrawTexLayer();
  bool IsUiDraw(bool is_ortho);
  void SetUIContentSize(uint32_t width, uint32_t height);
  // The part of the UI texture that holds content, as fractions (0..1) of its size.
  bool GetUIContentFraction(float* out_u, float* out_v);
  void EndFrameStructure();
  uint64_t FrameNo();
  bool LastFrameWas2D();
  void SetPresentedFrameIs2D(bool is_2d);

  void NoteDrawTextures(bool uses_efb_copy, bool copies_all_flat, bool copies_all_old);
  bool TakeCopyContentIsFlat();
}
