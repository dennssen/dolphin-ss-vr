#include "XR/XRSession.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <atomic>
#include <string>

#include "XRSession.h"
#include "vulkan/vulkan_core.h"

#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include "Common/Logging/Log.h"

namespace
{
  XrInstance s_instance = XR_NULL_HANDLE;
  XrSystemId s_system_id = XR_NULL_SYSTEM_ID;
  XrSession s_session = XR_NULL_HANDLE;
  XrSessionState s_state = XR_SESSION_STATE_UNKNOWN;
  bool s_running = false;
  XrFrameState s_frame_state{XR_TYPE_FRAME_STATE};
  XrSwapchain s_swapchain = XR_NULL_HANDLE;
  std::vector<XrSwapchainImageVulkan2KHR> s_swapchain_images;
  XrSwapchain s_ui_swapchain = XR_NULL_HANDLE;
  std::vector<XrSwapchainImageVulkan2KHR> s_ui_images;
  bool s_ui_acquired = false;
  uint32_t s_ui_index = 0;
  uint32_t s_ui_width = 0;
  uint32_t s_ui_height = 0;
  uint32_t s_ui_content_w = 0;
  uint32_t s_ui_content_h = 0;
  XrSpace s_space = XR_NULL_HANDLE;
  std::atomic<uint64_t> s_pose_generation{0};
  bool s_draw_uses_efb_copy = false;
  bool s_seen_perspective = false;
  bool s_prev_frame_had_perspective = false;
  uint32_t s_draw_index = 0;
  uint32_t s_ui_start = ~0u; // from the previous frame; ~0 means nothing counts as UI
  uint32_t s_ui_start_next = ~0u; // being collected for the next frame
  std::string s_trace;
  char s_trace_last = 0;

  struct Vec3
  {
    float x, y, z;
  };

  Vec3 Rotate(const XrQuaternionf& q, const Vec3& v)
  {
    const Vec3 t{2.0f * (q.y * v.z - q.z * v.y), 2.0f * (q.z * v.x - q.x * v.z),
                 2.0f * (q.x * v.y - q.y * v.x)};
    return {v.x + q.w * t.x + (q.y * t.z - q.z * t.y), v.y + q.w * t.y + (q.z * t.x - q.x * t.z),
        v.z + q.w * t.z + (q.x * t.y - q.y * t.x)};
  }

  XrQuaternionf Conjugate(const XrQuaternionf& q)
  {
    return {-q.x, -q.y, -q.z, q.w};
  }

  XrQuaternionf Multiply(const XrQuaternionf& a, const XrQuaternionf& b)
  {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
  }

  struct ViewSnapshot
  {
    bool valid = false;
    XrPosef pose[2]{};
    XrFovf fov[2]{};
  };

  ViewSnapshot s_latest;
  ViewSnapshot s_draw; // frozen copy used for drawing and for submitting the frame
  ViewSnapshot s_presented; // pose the most recently finished frame was drawn with
  XrPosef s_ref{};
  bool s_have_ref = false;
  std::mutex s_view_mutex;

  float UnitsPerMeter()
  {
    static const float s = [] {
      const char* v = std::getenv("DOLPHIN_SS_VR_SCALE");
      return v ? static_cast<float>(std::atof(v)) : 100.0f;
    }();
    return s;
  }

  bool s_frame_begun = false;
  bool s_image_acquired = false;
  uint32_t s_image_index = 0;
  uint32_t s_eye_width = 0;
  uint32_t s_eye_height = 0;

  XrDebugUtilsMessengerEXT s_messenger = XR_NULL_HANDLE;
  XrBool32 XRAPI_PTR OnXrMessage(XrDebugUtilsMessageSeverityFlagsEXT severity,
                               XrDebugUtilsMessageTypeFlagsEXT types,
                               const XrDebugUtilsMessengerCallbackDataEXT* data, void* user_data)
  {
    WARN_LOG_FMT(VIDEO, "XR runtime message: [{}] {}", data->functionName, data->message);
    return XR_FALSE;
  }

  bool IsExtensionSupported(const char* name)
  {
    uint32_t count = 0;
    xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr);
    std::vector<XrExtensionProperties> props(count, {XR_TYPE_EXTENSION_PROPERTIES});
    xrEnumerateInstanceExtensionProperties(nullptr, count, &count, props.data());
    return std::any_of(props.begin(), props.end(), [&](const XrExtensionProperties& p) {
      return std::strcmp(p.extensionName, name) == 0;
    });
  }
}

namespace XR
{

  bool Init()
  {
    XrInstanceCreateInfo create_info = {};
    create_info.type = XR_TYPE_INSTANCE_CREATE_INFO;
    std::strncpy(create_info.applicationInfo.applicationName, "Dolphin SS VR",
                 XR_MAX_APPLICATION_NAME_SIZE - 1);
    create_info.applicationInfo.applicationVersion = 1;
    create_info.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);

    std::vector<const char*> extensions = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
    const bool has_debug_utils = IsExtensionSupported(XR_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (has_debug_utils)
      extensions.push_back(XR_EXT_DEBUG_UTILS_EXTENSION_NAME);
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.enabledExtensionNames = extensions.data();

    XrInstance instance = XR_NULL_HANDLE;
    const XrResult result = xrCreateInstance(&create_info, &instance);
    if (XR_FAILED(result))
    {
      WARN_LOG_FMT(VIDEO, "XR: xrCreateInstance failed with code {}", static_cast<int>(result));
      return false;
    }

    if (has_debug_utils)
    {
      PFN_xrVoidFunction create_fn = nullptr;
      xrGetInstanceProcAddr(instance, "xrCreateDebugUtilsMessengerEXT", &create_fn);
      if (create_fn)
      {
        XrDebugUtilsMessengerCreateInfoEXT messenger_info{
            XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        messenger_info.messageSeverities = XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messenger_info.messageTypes = XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                      XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                      XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
                                      XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT;
        messenger_info.userCallback = OnXrMessage;
        reinterpret_cast<PFN_xrCreateDebugUtilsMessengerEXT>(create_fn)(instance, &messenger_info,
                                                                        &s_messenger);
      }
    }
    INFO_LOG_FMT(VIDEO, "XR: debug messenger active: {}", s_messenger != XR_NULL_HANDLE);

    XrInstanceProperties instance_props = {};
    instance_props.type = XR_TYPE_INSTANCE_PROPERTIES;
    xrGetInstanceProperties(instance, &instance_props);
    INFO_LOG_FMT(VIDEO, "XR: runtime is {}", instance_props.runtimeName);

    XrSystemGetInfo system_info = {};
    system_info.type = XR_TYPE_SYSTEM_GET_INFO;
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    XrSystemId system_id = XR_NULL_SYSTEM_ID;
    if (XR_FAILED(xrGetSystem(instance, &system_info, &system_id)))
    {
      WARN_LOG_FMT(VIDEO, "XR: runtime found, but no headset available");
      xrDestroyInstance(instance);
      return false;
    }

    PFN_xrVoidFunction fn = nullptr;
    xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsRequirements2KHR", &fn);
    PFN_xrGetVulkanGraphicsRequirements2KHR get_reqs = reinterpret_cast<PFN_xrGetVulkanGraphicsRequirements2KHR>(fn);

    if (get_reqs)
    {
      XrGraphicsRequirementsVulkan2KHR reqs = {};
      reqs.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR;
      const XrResult reqs_result = get_reqs(instance, system_id, &reqs);
      if (XR_SUCCEEDED(reqs_result))
      {
        INFO_LOG_FMT(VIDEO, "XR: Vulkan {}.{} to {}.{} required",
                     XR_VERSION_MAJOR(reqs.minApiVersionSupported),
                     XR_VERSION_MINOR(reqs.minApiVersionSupported),
                     XR_VERSION_MAJOR(reqs.maxApiVersionSupported),
                     XR_VERSION_MINOR(reqs.maxApiVersionSupported));
      }
      else
      {
        WARN_LOG_FMT(VIDEO, "XR: requirements query failed with code {}", static_cast<int>(reqs_result));
      }
    }

    XrSystemProperties system_props = {};
    system_props.type = XR_TYPE_SYSTEM_PROPERTIES;
    xrGetSystemProperties(instance, system_id, &system_props);
    INFO_LOG_FMT(VIDEO, "XR: headset is {}", system_props.systemName);

    s_instance = instance;
    s_system_id = system_id;
    return true;
  }

  void Shutdown()
  {
    if (!IsActive())
    {
      return;
    }

    DestroySession();

    if (s_messenger != XR_NULL_HANDLE)
    {
      PFN_xrVoidFunction destroy_fn = nullptr;
      xrGetInstanceProcAddr(s_instance, "xrDestroyDebugUtilsMessengerEXT", &destroy_fn);
      if (destroy_fn)
        reinterpret_cast<PFN_xrDestroyDebugUtilsMessengerEXT>(destroy_fn)(s_messenger);
      s_messenger = XR_NULL_HANDLE;
    }

    xrDestroyInstance(s_instance);

    s_instance = XR_NULL_HANDLE;
    s_system_id = XR_NULL_SYSTEM_ID;
  }

  bool IsActive()
  {
    return s_instance != XR_NULL_HANDLE;
  }

  VkResult CreateVulkanInstance(const VkInstanceCreateInfo* info,
                                PFN_vkGetInstanceProcAddr get_proc, VkInstance* out_instance)
  {
    PFN_xrVoidFunction fn = nullptr;
    xrGetInstanceProcAddr(s_instance, "xrCreateVulkanInstanceKHR", &fn);
    if (!fn)
    {
      WARN_LOG_FMT(VIDEO, "XR: xrCreateVulkanInstanceKHR not available");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    const auto create = reinterpret_cast<PFN_xrCreateVulkanInstanceKHR>(fn);

    XrVulkanInstanceCreateInfoKHR create_info = {};
    create_info.type = XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR;
    create_info.systemId = s_system_id;
    create_info.pfnGetInstanceProcAddr = get_proc;
    create_info.vulkanCreateInfo = info;

    VkResult vk_result = VK_SUCCESS;
    const XrResult xr_result = create(s_instance, &create_info, out_instance, &vk_result);
    INFO_LOG_FMT(VIDEO, "XR: xrCreateVulkanInstanceKHR xr={} vk={}", static_cast<int>(xr_result),
                 static_cast<int>(vk_result));
    if (XR_FAILED(xr_result))
      return VK_ERROR_INITIALIZATION_FAILED;

    return vk_result;
  }

  VkPhysicalDevice GetVulkanPhysicalDevice(VkInstance instance)
  {
    PFN_xrVoidFunction fn = nullptr;
    xrGetInstanceProcAddr(s_instance, "xrGetVulkanGraphicsDevice2KHR", &fn);
    if (!fn)
    {
      WARN_LOG_FMT(VIDEO, "XR: xrGetVulkanGraphicsDevice2KHR not available");
      return VK_NULL_HANDLE;
    }
    const auto get_device = reinterpret_cast<PFN_xrGetVulkanGraphicsDevice2KHR>(fn);

    XrVulkanGraphicsDeviceGetInfoKHR get_info = {};
    get_info.type = XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR;
    get_info.systemId = s_system_id;
    get_info.vulkanInstance = instance;

    VkPhysicalDevice device = VK_NULL_HANDLE;
    const XrResult result = get_device(s_instance, &get_info, &device);
    if (XR_FAILED(result))
    {
      WARN_LOG_FMT(VIDEO, "XR: xrGetVulkanGraphicsDevice2KHR failed with code {}",
                   static_cast<int>(result));
      return VK_NULL_HANDLE;
    }

    return device;
  }

  VkResult CreateVulkanDevice(VkPhysicalDevice physical_device, const VkDeviceCreateInfo* info,
                              PFN_vkGetInstanceProcAddr get_proc, VkDevice* out_device)
  {
    PFN_xrVoidFunction fn = nullptr;
    xrGetInstanceProcAddr(s_instance, "xrCreateVulkanDeviceKHR", &fn);
    if (!fn)
    {
      WARN_LOG_FMT(VIDEO, "XR: xrCreateVulkanDeviceKHR not available");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    const auto create = reinterpret_cast<PFN_xrCreateVulkanDeviceKHR>(fn);

    XrVulkanDeviceCreateInfoKHR create_info = {};
    create_info.type = XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR;
    create_info.systemId = s_system_id;
    create_info.pfnGetInstanceProcAddr = get_proc;
    create_info.vulkanPhysicalDevice = physical_device;
    create_info.vulkanCreateInfo = info;

    VkResult vk_result = VK_SUCCESS;
    const XrResult xr_result = create(s_instance, &create_info, out_device, &vk_result);
    INFO_LOG_FMT(VIDEO, "XR: xrCreateVulkanDeviceKHR xr={} vk={}", static_cast<int>(xr_result),
                 static_cast<int>(vk_result));
    if (XR_FAILED(xr_result))
      return VK_ERROR_INITIALIZATION_FAILED;
    return vk_result;
  }

  bool CreateSession(VkInstance vk_instance, VkPhysicalDevice physical_device, VkDevice device,
                     uint32_t queue_family_index, uint32_t queue_index)
  {
    XrGraphicsBindingVulkan2KHR binding = {};
    binding.type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR;
    binding.instance = vk_instance;
    binding.physicalDevice = physical_device;
    binding.device = device;
    binding.queueFamilyIndex = queue_family_index;
    binding.queueIndex = queue_index;

    XrSessionCreateInfo session_info = {};
    session_info.type = XR_TYPE_SESSION_CREATE_INFO;
    session_info.next = &binding;
    session_info.systemId = s_system_id;

    const XrResult result = xrCreateSession(s_instance, &session_info, &s_session);
    if (XR_FAILED(result))
    {
      WARN_LOG_FMT(VIDEO, "XR: xrCreateSession failed with code {}", static_cast<int>(result));
      return false;
    }
    INFO_LOG_FMT(VIDEO, "XR: session created");

    uint32_t view_count = 0;
    xrEnumerateViewConfigurationViews(s_instance, s_system_id,
                                      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &view_count,
                                      nullptr);
    std::vector<XrViewConfigurationView> config_views(view_count,
                                                      {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    xrEnumerateViewConfigurationViews(s_instance, s_system_id,
                                      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, view_count,
                                      &view_count, config_views.data());
    if (view_count < 2)
      return false;
    s_eye_width = config_views[0].recommendedImageRectWidth;
    s_eye_height = config_views[0].recommendedImageRectHeight;
    INFO_LOG_FMT(VIDEO, "XR: recommended eye size {}x{}", s_eye_width, s_eye_height);

    // One swapchain, both eyes side by side.
    if (!CreateSwapchain(s_eye_width * 2, s_eye_height))
      return false;

    XrReferenceSpaceCreateInfo space_info{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    space_info.poseInReferenceSpace.orientation.w = 1.0f;
    if (XR_FAILED(xrCreateReferenceSpace(s_session, &space_info, &s_space)))
    {
      WARN_LOG_FMT(VIDEO, "XR: xrCreateReferenceSpace failed");
      return false;
    }

    return true;
  }

  void DestroySession()
  {
    if (s_session == XR_NULL_HANDLE)
    {
      return;
    }
    if (s_running) {
      xrEndSession(s_session);
      s_running = false;
    }

    if (s_space != XR_NULL_HANDLE)
    {
      xrDestroySpace(s_space);
      s_space = XR_NULL_HANDLE;
    }

    if (s_ui_swapchain != XR_NULL_HANDLE)
    {
      xrDestroySwapchain(s_ui_swapchain);
      s_ui_swapchain = XR_NULL_HANDLE;
      s_ui_images.clear();
    }

    if (s_swapchain != XR_NULL_HANDLE)
    {
      xrDestroySwapchain(s_swapchain);
      s_swapchain = XR_NULL_HANDLE;
      s_swapchain_images.clear();
    }

    xrDestroySession(s_session);
  }

  void PollEvents()
  {
    if (s_instance == XR_NULL_HANDLE)
      return;

    XrEventDataBuffer event = {};
    event.type = XR_TYPE_EVENT_DATA_BUFFER;
    while (xrPollEvent(s_instance, &event) == XR_SUCCESS)
    {
      if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED)
      {
        const auto& changed = *reinterpret_cast<const XrEventDataSessionStateChanged*>(&event);
        s_state = changed.state;
        INFO_LOG_FMT(VIDEO, "XR_ session state {}", static_cast<int>(s_state));

        if (s_state == XR_SESSION_STATE_READY && !s_running)
        {
          XrSessionBeginInfo begin_info = {};
          begin_info.type = XR_TYPE_SESSION_BEGIN_INFO;
          begin_info.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
          const XrResult result = xrBeginSession(s_session, &begin_info);
          s_running = XR_SUCCEEDED(result);
          INFO_LOG_FMT(VIDEO, "XR: xrBeginSession result {}", static_cast<int>(result));
        }
        else if (s_state == XR_SESSION_STATE_STOPPING && s_running)
        {
          xrEndSession(s_session);
          s_running = false;
          INFO_LOG_FMT(VIDEO, "XR: session ended");
        }
      }
      event = XrEventDataBuffer{XR_TYPE_EVENT_DATA_BUFFER};
    }
  }

  bool BeginFrame()
  {
    if (!s_running)
    {
      return false;
    }

    XrFrameWaitInfo wait_info{XR_TYPE_FRAME_WAIT_INFO};
    s_frame_state = XrFrameState{XR_TYPE_FRAME_STATE};
    if (XR_FAILED(xrWaitFrame(s_session, &wait_info, &s_frame_state)))
    {
      return false;
    }

    XrFrameBeginInfo begin_info{XR_TYPE_FRAME_BEGIN_INFO};
    if (XR_FAILED(xrBeginFrame(s_session, &begin_info)))
    {
      return false;
    }
    s_frame_begun = true;

    std::lock_guard<std::mutex> lock(s_view_mutex);
    XrViewLocateInfo locate_info{XR_TYPE_VIEW_LOCATE_INFO};
    locate_info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    locate_info.displayTime = s_frame_state.predictedDisplayTime;
    locate_info.space = s_space;
    XrViewState view_state{XR_TYPE_VIEW_STATE};
    XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    uint32_t located = 0;
    const XrResult r = xrLocateViews(s_session, &locate_info, &view_state, 2, &located, views);
    const XrViewStateFlags needed =
        XR_VIEW_STATE_ORIENTATION_VALID_BIT | XR_VIEW_STATE_POSITION_VALID_BIT;
    s_latest.valid = XR_SUCCEEDED(r) && located == 2 && (view_state.viewStateFlags & needed) == needed;
    if (s_latest.valid)
    {
      for (int i = 0; i < 2; i++)
      {
        s_latest.pose[i] = views[i].pose;
        s_latest.fov[i] = views[i].fov;
      }
      if (!s_draw.valid)
      {
        s_draw = s_latest;
        s_presented = s_latest;
        s_pose_generation++;
      }
      if (!s_have_ref)
      {
        s_ref.orientation = views[0].pose.orientation;
        s_ref.position = {(views[0].pose.position.x + views[1].pose.position.x) * 0.5f,
                          (views[0].pose.position.y + views[1].pose.position.y) * 0.5f,
                          (views[0].pose.position.z + views[1].pose.position.z) * 0.5f};
        s_have_ref = true;
      }

      static int s_view_frames = 0;
      if (s_view_frames++ % 120 == 0)
      {
        const XrPosef& p = s_latest.pose[0];
        INFO_LOG_FMT(VIDEO,
                    "XR: eye0 pos {:.3f} {:.3f} {:.3f} quat {:.3f} {:.3f} {:.3f} {:.3f} fov "
                    "{:.2f} {:.2f} {:.2f} {:.2f}",
                    p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y,
                    p.orientation.z, p.orientation.w, s_latest.fov[0].angleLeft,
                    s_latest.fov[0].angleRight, s_latest.fov[0].angleUp,
                    s_latest.fov[0].angleDown);
      }
    }

    if (!s_frame_state.shouldRender)
    {
      return false;
    }

    XrSwapchainImageAcquireInfo acquire_info{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    XrSwapchainImageWaitInfo image_wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    image_wait.timeout = XR_INFINITE_DURATION;
    if (XR_SUCCEEDED(xrAcquireSwapchainImage(s_swapchain, &acquire_info, &s_image_index)) && XR_SUCCEEDED(xrWaitSwapchainImage(s_swapchain, &image_wait)))
    {
      s_image_acquired = true;
    }

    return s_image_acquired;
  }

  static bool CreateUISwapchain(uint32_t width, uint32_t height)
  {
    uint32_t format_count = 0;
    xrEnumerateSwapchainFormats(s_session, 0, &format_count, nullptr);
    std::vector<int64_t> formats(format_count);
    xrEnumerateSwapchainFormats(s_session, format_count, &format_count, formats.data());

    for (const int64_t f : formats)
      INFO_LOG_FMT(VIDEO, "XR: runtime offers swapchain format {}", f);

    // RGBA SRGB only: the UI is raw-copied from an RGBA8 texture, so channel order must match.
    const int64_t preferred[] = {VK_FORMAT_R8G8B8A8_SRGB};
    int64_t chosen = 0;
    for (const int64_t format : preferred)
    {
      if (std::find(formats.begin(), formats.end(), format) != formats.end())
      {
        chosen = format;
        break;
      }
    }
    if (chosen == 0)
    {
      WARN_LOG_FMT(VIDEO, "XR: no usable UI swapchain format");
      return false;
    }

    XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
                      XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    info.format = chosen;
    info.sampleCount = 1;
    info.width = width;
    info.height = height;
    info.faceCount = 1;
    info.arraySize = 1;
    info.mipCount = 1;
    if (XR_FAILED(xrCreateSwapchain(s_session, &info, &s_ui_swapchain)))
    {
      WARN_LOG_FMT(VIDEO, "XR: UI xrCreateSwapchain failed");
      s_ui_swapchain = XR_NULL_HANDLE;
      return false;
    }

    uint32_t image_count = 0;
    xrEnumerateSwapchainImages(s_ui_swapchain, 0, &image_count, nullptr);
    s_ui_images.assign(image_count,
                       XrSwapchainImageVulkan2KHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
    xrEnumerateSwapchainImages(s_ui_swapchain, image_count, &image_count,
                            reinterpret_cast<XrSwapchainImageBaseHeader*>(s_ui_images.data()));
    s_ui_width = width;
    s_ui_height = height;
    INFO_LOG_FMT(VIDEO, "XR: UI swapchain created, format {}, {} images, {}x{}",
                 static_cast<int>(chosen), image_count, width, height);
    return true;
  }

  VkImage AcquireUIImage(uint32_t width, uint32_t height)
  {
    if (!s_frame_begun)
    {
      return VK_NULL_HANDLE;
    }

    if (s_ui_swapchain == XR_NULL_HANDLE && !CreateUISwapchain(width, height))
    {
      return VK_NULL_HANDLE;
    }

    XrSwapchainImageAcquireInfo acquire_info{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    XrSwapchainImageWaitInfo wait_info{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait_info.timeout = XR_INFINITE_DURATION;
    if (XR_FAILED(xrAcquireSwapchainImage(s_ui_swapchain, &acquire_info, &s_ui_index)) || XR_FAILED(xrWaitSwapchainImage(s_ui_swapchain, &wait_info)))
    {
      return VK_NULL_HANDLE;
    }

    s_ui_acquired = true;
    return s_ui_images[s_ui_index].image;
  }

  VkImage GetCurrentImage()
  {
    return s_image_acquired ? s_swapchain_images[s_image_index].image : VK_NULL_HANDLE;
  }

  void EndFrame()
  {
    if (!s_frame_begun)
      return;

    XrCompositionLayerProjectionView proj_views[2] = {};
    XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    const XrCompositionLayerBaseHeader* layers[2] = {};
    uint32_t layer_count = 0;

    if (s_image_acquired)
    {
      XrSwapchainImageReleaseInfo release_info{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
      xrReleaseSwapchainImage(s_swapchain, &release_info);

      ViewSnapshot snap;
      {
        std::lock_guard<std::mutex> lock(s_view_mutex);
        snap = s_presented;
      }

      if (snap.valid)
      {
        for (uint32_t i = 0; i < 2; i++)
        {
          proj_views[i].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
          proj_views[i].pose = snap.pose[i];
          proj_views[i].fov = snap.fov[i];
          proj_views[i].subImage.swapchain = s_swapchain;
          proj_views[i].subImage.imageRect.offset = {static_cast<int32_t>(i * s_eye_width), 0};
          proj_views[i].subImage.imageRect.extent = {static_cast<int32_t>(s_eye_width),
                                                    static_cast<int32_t>(s_eye_height)};
          proj_views[i].subImage.imageArrayIndex = 0;
        }
        projection.space = s_space;
        projection.viewCount = 2;
        projection.views = proj_views;

        layers[0] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection);
        layer_count = 1;
      }
    }

    XrCompositionLayerQuad quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
    if (s_ui_acquired)
    {
      XrSwapchainImageReleaseInfo ui_release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
      xrReleaseSwapchainImage(s_ui_swapchain, &ui_release);
      s_ui_acquired = false;

      quad.layerFlags = 0;  // opaque for now; alpha blending comes with the clear-alpha step
      quad.space = s_space;
      quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
      quad.subImage.swapchain = s_ui_swapchain;
      quad.subImage.imageRect.offset = {0, 0};
      const uint32_t crop_w = (s_ui_content_w > 0) ? std::min(s_ui_content_w, s_ui_width) : s_ui_width;
      const uint32_t crop_h = (s_ui_content_h > 0) ? std::min(s_ui_content_h, s_ui_height) : s_ui_height;
      quad.subImage.imageRect.extent = {static_cast<int32_t>(crop_w), static_cast<int32_t>(crop_h)};
      quad.subImage.imageArrayIndex = 0;
      // Place the panel 1.5 m in front of the game's reference pose, using only its yaw
      // so the panel stays upright and level.
      XrPosef ref;
      bool have_ref;
      {
        std::lock_guard<std::mutex> lock(s_view_mutex);
        ref = s_ref;
        have_ref = s_have_ref;
      }
      if (have_ref)
      {
        const XrQuaternionf& q = ref.orientation;
        const float yaw = std::atan2(2.0f * (q.x * q.z + q.w * q.y), 1.0f - 2.0f * (q.x * q.x + q.y * q.y));
        quad.pose.orientation = {0.0f, std::sin(yaw * 0.5f), 0.0f, std::cos(yaw * 0.5f)};
        quad.pose.position = {ref.position.x - std::sin(yaw) * 1.5f, ref.position.y,
                              ref.position.z - std::cos(yaw) * 1.5f};
      }
      else
      {
        quad.pose.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
        quad.pose.position = {0.0f, 0.0f, -1.5f};
      }
      quad.size = {2.0f, 2.0f * 9.0f / 16.0f};
      layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quad);
    }

    XrFrameEndInfo end_info{XR_TYPE_FRAME_END_INFO};
    end_info.displayTime = s_frame_state.predictedDisplayTime;
    end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    end_info.layerCount = layer_count;
    end_info.layers = layers;
    const XrResult end_result = xrEndFrame(s_session, &end_info);
    if (XR_FAILED(end_result))
      WARN_LOG_FMT(VIDEO, "XR: xrEndFrame failed with code {}", static_cast<int>(end_result));

    s_frame_begun = false;
    s_image_acquired = false;
  }

  bool CreateSwapchain(uint32_t width, uint32_t height)
  {
    uint32_t format_count = 0;
    xrEnumerateSwapchainFormats(s_session, 0, &format_count, nullptr);
    std::vector<int64_t> formats(format_count);
    xrEnumerateSwapchainFormats(s_session, format_count, &format_count, formats.data());

    const int64_t preferred[] = {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB,
                               VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM};
    int64_t chosen = 0;
    for (const int64_t format : preferred)
    {
      if (std::find(formats.begin(), formats.end(), format) != formats.end())
      {
        chosen = format;
        break;
      }
    }
    if (chosen == 0)
    {
      WARN_LOG_FMT(VIDEO, "XR: no usable swapchain format");
      return false;
    }

    XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT |
                  XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    info.format = chosen;
    info.sampleCount = 1;
    info.width = width;
    info.height = height;
    info.faceCount = 1;
    info.arraySize = 1;
    info.mipCount = 1;

    const XrResult result = xrCreateSwapchain(s_session, &info, &s_swapchain);
    if (XR_FAILED(result))
    {
      WARN_LOG_FMT(VIDEO, "XR: xrCreateSwapchain failed with code {}", static_cast<int>(result));
      return false;
    }

    uint32_t image_count = 0;
    xrEnumerateSwapchainImages(s_swapchain, 0, &image_count, nullptr);
    s_swapchain_images.assign(image_count,
                              XrSwapchainImageVulkan2KHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
    xrEnumerateSwapchainImages(s_swapchain, image_count, &image_count, reinterpret_cast<XrSwapchainImageBaseHeader*>(s_swapchain_images.data()));
    INFO_LOG_FMT(VIDEO, "XR: swapchain created, format {}, {} images, {}x{}",
                 static_cast<int>(chosen), image_count, width, height);

    return true;
  }

  uint32_t GetSwapchainWidth()
  {
    return s_eye_width * 2;
  }

  uint32_t GetSwapchainHeight()
  {
    return s_eye_height;
  }

  void Recenter()
  {
    std::lock_guard<std::mutex> lock(s_view_mutex);
    s_have_ref = false;
  }

  bool GetEyeData(EyeData out[2])
  {
    std::lock_guard<std::mutex> lock(s_view_mutex);
    if (!s_draw.valid || !s_have_ref)
    {
      return false;
    }

    const float scale = UnitsPerMeter();
    const XrQuaternionf ref_inv = Conjugate(s_ref.orientation);
    for (int i = 0; i < 2; i++)
    {
      const XrPosef& p = s_draw.pose[i];
      const XrQuaternionf rel = Multiply(ref_inv, p.orientation);
      Vec3 t = Rotate(ref_inv, {p.position.x - s_ref.position.x, p.position.y - s_ref.position.y,
                                p.position.z - s_ref.position.z});
      t = {t.x * scale, t.y * scale, t.z * scale};

      const Vec3 cx = Rotate(rel, {1, 0, 0});
      const Vec3 cy = Rotate(rel, {0, 1, 0});
      const Vec3 cz = Rotate(rel, {0, 0, 1});
      out[i].view = {cx.x, cx.y, cx.z, -(cx.x * t.x + cx.y * t.y + cx.z * t.z),
                   cy.x, cy.y, cy.z, -(cy.x * t.x + cy.y * t.y + cy.z * t.z),
                   cz.x, cz.y, cz.z, -(cz.x * t.x + cz.y * t.y + cz.z * t.z),
                   0.0f, 0.0f, 0.0f, 1.0f};
      out[i].tan_left = std::tan(s_draw.fov[i].angleLeft);
      out[i].tan_right = std::tan(s_draw.fov[i].angleRight);
      out[i].tan_up = std::tan(s_draw.fov[i].angleUp);
      out[i].tan_down = std::tan(s_draw.fov[i].angleDown);
    }

    return true;
  }

  uint64_t GetPoseGeneration()
  {
    return s_pose_generation.load();
  }

  void LatchPose()
  {
    std::lock_guard<std::mutex> lock(s_view_mutex);
    if (!s_latest.valid)
    {
      return;
    }

    s_presented = s_draw; // the frame that just finished was drawn with this pose
    s_draw = s_latest; // draws of the next frame use the newest pose
    s_pose_generation++;
  }

  bool IsUiDraw(bool is_ortho)
  {
    if (!is_ortho)
    {
      s_seen_perspective = true;
      return false;
    }
    if (s_draw_uses_efb_copy)
    {
      return false;
    }
    // After the first 3D draw of a frame, or in a frame following one with no 3D at all.
    return s_seen_perspective || !s_prev_frame_had_perspective;
  }

  void SetUIContentSize(uint32_t width, uint32_t height)
  {
    s_ui_content_w = width;
    s_ui_content_h = height;
  }

  void NoteCopy()
  {
    s_ui_start_next = s_draw_index;
  }

  void NoteDrawTextures(bool uses_efb_copy)
  {
    s_draw_uses_efb_copy = uses_efb_copy;
  }

  void EndFrameStructure()
  {
    s_prev_frame_had_perspective = s_seen_perspective;
    s_seen_perspective = false;
    s_draw_index = 0;
  }

  void TraceEvent(char c)
  {
    if (s_draw_uses_efb_copy)
    {
      if (c == 'O')
      {
        c = 'E';
      }
      else if (c == 'U')
      {
        c = 'V';
      }
    }

    if (c == s_trace_last || s_trace.size() >= 200)
    {
      return;
    }
    s_trace_last = c;
    s_trace.push_back(c);
  }

  void FlushTrace()
  {
    static int s_3d_frames = 0;
    static std::string s_last_2d;
    if (s_trace.find('P') != std::string::npos)
    {
      if ((s_3d_frames++ % 30) == 0)
        INFO_LOG_FMT(VIDEO, "XR: pass trace {}", s_trace);
    }
    else if (s_trace != s_last_2d)
    {
      s_last_2d = s_trace;
      INFO_LOG_FMT(VIDEO, "XR: 2D pass trace {}", s_trace);
    }
    s_trace.clear();
    s_trace_last = 0;
  }
}
