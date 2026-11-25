#include "Vst2Abi.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "DpdfDenoiser.h"
#include "embedded_model.h"

#include <windows.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef DPDFNET_LOCAL_ONNXRUNTIME
#include <mutex>
#include <type_traits>
#endif

namespace {

using namespace vst2;
using dpdfvst::DpdfDenoiser;

constexpr Int kBypass = 0;
constexpr Int kSliceSize = 128;
constexpr Int kModelRate = 48000;
constexpr Int kModelWindow = 960;
constexpr Int kVersion = 1000;
constexpr int kEditorResource = 103;
constexpr int kBypassControl = 1001;
constexpr int kEngineControl = 1003;
constexpr UINT_PTR kEditorTimer = 1;

struct EditorRect {
  std::int16_t top, left, bottom, right;
};

enum class Engine { Waiting, NativeInt8, Onnx, Unavailable };

const wchar_t* engineText(Engine engine) {
  switch (engine) {
    case Engine::NativeInt8: return L"Engine: Native INT8";
    case Engine::Onnx: return L"Engine: ONNX Runtime";
    case Engine::Unavailable: return L"Engine: unavailable";
    default: return L"Engine: waiting for audio";
  }
}

std::filesystem::path pluginDirectory() {
  HMODULE pluginModule = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&pluginDirectory), &pluginModule)) {
    throw std::runtime_error("Cannot locate DPDFNet.dll");
  }
  std::wstring modulePath(32768, L'\0');
  const DWORD length = GetModuleFileNameW(pluginModule, modulePath.data(),
                                         static_cast<DWORD>(modulePath.size()));
  if (!length || length == modulePath.size()) {
    throw std::runtime_error("Cannot resolve the DPDFNet.dll path");
  }
  modulePath.resize(length);
  return std::filesystem::path(modulePath).parent_path();
}

#ifdef DPDFNET_LOCAL_ONNXRUNTIME
using RuntimeModule = std::shared_ptr<std::remove_pointer_t<HMODULE>>;

RuntimeModule acquireRuntime(const std::filesystem::path& directory) {
  static std::mutex mutex;
  static std::weak_ptr<RuntimeModule::element_type> cachedRuntime;
  const std::lock_guard lock(mutex);
  if (auto runtime = cachedRuntime.lock()) return runtime;

  const auto runtimePath = directory / L"onnxruntime.dll";
  const RuntimeModule runtime(
      LoadLibraryExW(runtimePath.c_str(), nullptr,
                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32),
      [](HMODULE module) { if (module) FreeLibrary(module); });
  if (!runtime) throw std::runtime_error("Cannot load ONNX Runtime beside DPDFNet.dll");

  const auto getApiBase = reinterpret_cast<decltype(&OrtGetApiBase)>(
      GetProcAddress(runtime.get(), "OrtGetApiBase"));
  const auto* apiBase = getApiBase ? getApiBase() : nullptr;
  const auto* api = apiBase ? apiBase->GetApi(ORT_API_VERSION) : nullptr;
  if (!api) throw std::runtime_error("ONNX Runtime does not support the required API version");
  Ort::InitApi(api);
  cachedRuntime = runtime;
  return runtime;
}
#endif

Value writeString(void* destination, const char* source, std::size_t capacity) {
  if (!destination) return 0;
  const auto length = std::min(std::strlen(source), capacity - 1);
  std::memcpy(destination, source, length);
  static_cast<char*>(destination)[length] = '\0';
  return 1;
}

class Plugin {
public:
  explicit Plugin(Dispatch host) : host_(host) {
    effect_.magic = kEffectMagic;
    effect_.dispatcher = dispatch;
    effect_.process = processAccumulating;
    effect_.setParameter = setParameter;
    effect_.getParameter = getParameter;
    effect_.numPrograms = 1;
    effect_.numParams = 1;
    effect_.numInputs = 2;
    effect_.numOutputs = 2;
    effect_.flags = kCanReplacing | kHasEditor;
    effect_.initialDelay = kModelWindow;
    effect_.ioRatio = 1.f;
    effect_.object = this;
    effect_.uniqueId = kPluginId;
    effect_.version = kVersion;
    effect_.processReplacing = processReplacing;
  }

  Effect* effect() { return &effect_; }

private:
  ~Plugin() { closeEditor(); }

  void updateEditor(HWND window) {
    CheckDlgButton(window, kBypassControl,
                   getParameter(&effect_, kBypass) ? BST_CHECKED : BST_UNCHECKED);
    const auto engine = engine_.load(std::memory_order_relaxed);
    if (engine != displayedEngine_) {
      SetDlgItemTextW(window, kEngineControl, engineText(engine));
      displayedEngine_ = engine;
    }
  }

  void editParameter(Int index, float value) {
    setParameter(&effect_, index, value);
    host_(&effect_, kHostAutomate, index, 0, nullptr, value);
  }

  static INT_PTR CALLBACK editorProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    if (message == WM_INITDIALOG) {
      auto* plugin = reinterpret_cast<Plugin*>(lParam);
      SetWindowLongPtrW(window, DWLP_USER, lParam);
      plugin->displayedEngine_ = Engine::Waiting;
      plugin->updateEditor(window);
      SetTimer(window, kEditorTimer, 100, nullptr);
      return TRUE;
    }

    auto* plugin = reinterpret_cast<Plugin*>(GetWindowLongPtrW(window, DWLP_USER));
    if (!plugin) return FALSE;
    if (message == WM_TIMER && wParam == kEditorTimer) {
      plugin->updateEditor(window);
      return TRUE;
    }
    if (message == WM_COMMAND && lParam) {
      if (LOWORD(wParam) == kBypassControl && HIWORD(wParam) == BN_CLICKED) {
        plugin->editParameter(kBypass,
                              IsDlgButtonChecked(window, kBypassControl) == BST_CHECKED ? 1.f : 0.f);
        return TRUE;
      }
    }
    if (message == WM_NCDESTROY) {
      KillTimer(window, kEditorTimer);
      plugin->editor_ = nullptr;
    }
    return FALSE;
  }

  Value getEditorRect(void* ptr) {
    if (!ptr) return 0;
    const DWORD units = GetDialogBaseUnits();
    editorRect_ = {0, 0, static_cast<std::int16_t>(MulDiv(43, HIWORD(units), 8)),
                        static_cast<std::int16_t>(MulDiv(160, LOWORD(units), 4))};
    *static_cast<EditorRect**>(ptr) = &editorRect_;
    return 1;
  }

  bool openEditor(HWND parent) {
    if (!parent) return false;
    if (editor_ && GetParent(editor_) == parent) return true;
    closeEditor();
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&editorProc), &module)) return false;
    editor_ = CreateDialogParamW(module, MAKEINTRESOURCEW(kEditorResource), parent,
                                 editorProc, reinterpret_cast<LPARAM>(this));
    if (!editor_) return false;
    ShowWindow(editor_, SW_SHOWNOACTIVATE);
    return true;
  }

  void closeEditor() {
    if (editor_) DestroyWindow(editor_);
  }

  static Plugin& from(Effect* effect) {
    return *static_cast<Plugin*>(effect->object);
  }

  static Value __cdecl dispatch(Effect* effect, Int opcode, Int index,
                                Value value, void* ptr, float option) {
    auto& plugin = from(effect);
    try {
      switch (opcode) {
        case kOpen: return 1;
        case kClose: delete &plugin; return 1;
        case kSetProgram: return value == 0;
        case kGetProgram: return 0;
        case kSetProgramName:
          return ptr ? writeString(plugin.programName_, static_cast<const char*>(ptr), 24) : 0;
        case kGetProgramName:
          return writeString(ptr, plugin.programName_, 24);
        case kGetProgramNameIndexed:
          return index == 0 ? writeString(ptr, plugin.programName_, 24) : 0;
        case kGetParamName:
          if (index == kBypass) return writeString(ptr, "Bypass", 8);
          return 0;
        case kGetParamLabel:
          return index == kBypass ? writeString(ptr, "", 8) : 0;
        case kGetParamDisplay:
          if (index == kBypass) return writeString(ptr, getParameter(effect, index) ? "On" : "Off", 8);
          return 0;
        case kCanBeAutomated: return index == kBypass;
        case kSetSampleRate: return plugin.setSampleRate(option);
        case kSetBlockSize: return 1;
        case kMainsChanged:
          if (value) plugin.resume();
          else plugin.active_ = false;
          return 1;
        case kSetBypass:
          plugin.hostBypass_.store(value != 0, std::memory_order_relaxed);
          return 1;
        case kEditGetRect: return plugin.getEditorRect(ptr);
        case kEditOpen: return plugin.openEditor(static_cast<HWND>(ptr));
        case kEditClose: plugin.closeEditor(); return 1;
        case kGetPlugCategory: return 1; // Effect
        case kGetEffectName: return writeString(ptr, "DPDFNet", 32);
        case kGetProductString: return writeString(ptr, "DPDFNet", 64);
        case kGetVendorString: return writeString(ptr, "Outsidepro Arts", 64);
        case kGetVendorVersion: return kVersion;
        case kGetVstVersion: return 2400;
        case kStartProcess:
        case kStopProcess: return 1;
        case kCanDo:
          return ptr && std::strcmp(static_cast<const char*>(ptr), "bypass") == 0;
        default: return 0;
      }
    } catch (...) {
      plugin.active_ = false;
      return 0;
    }
  }

  static void __cdecl setParameter(Effect* effect, Int index, float value) {
    auto& plugin = from(effect);
    if (index == kBypass) plugin.bypass_.store(value >= 0.5f, std::memory_order_relaxed);
  }

  static float __cdecl getParameter(Effect* effect, Int index) {
    auto& plugin = from(effect);
    if (index == kBypass) return plugin.bypass_.load(std::memory_order_relaxed) ? 1.f : 0.f;
    return 0.f;
  }

  bool setSampleRate(float rate) {
    if (!std::isfinite(rate) || rate < 1000.f || rate > 768000.f) return false;
    const auto sampleRate = static_cast<Int>(std::lround(rate));
    if (sampleRate == sampleRate_) return true;
    const double ratio = double(sampleRate) / kModelRate;
    Int delay = static_cast<Int>(std::ceil(kModelWindow * ratio));
    if (sampleRate != kModelRate) {
      // Allow both resamplers to collect their filter lookahead.
      delay += static_cast<Int>(std::ceil(16.0 * std::max(1.0, ratio)));
    }
    sampleRate_ = sampleRate;
    effect_.initialDelay = delay;
    if (active_) resume();
    host_(&effect_, kHostIoChanged, 0, 0, nullptr, 0.f);
    return true;
  }

  void resume() {
    active_ = false;
    if (!denoiser_) {
      const auto directory = pluginDirectory();
      auto denoiser = std::make_unique<DpdfDenoiser>(directory);
      if (!denoiser->isReady()) {
        try {
#ifdef DPDFNET_LOCAL_ONNXRUNTIME
          if (!runtime_) runtime_ = acquireRuntime(directory);
#endif
          std::size_t size = 0;
          const auto* data = dpdfvst::GetEmbeddedModelData(&size);
          if (data) denoiser = std::make_unique<DpdfDenoiser>(data, size);
        } catch (...) {}
      }
      if (denoiser->isReady()) denoiser_ = std::move(denoiser);
    }
    if (denoiser_) {
      denoiser_->setHostSampleRate(sampleRate_);
      denoiser_->reset();
    }
    engine_.store(denoiser_ ? (denoiser_->usesNative() ? Engine::NativeInt8 : Engine::Onnx)
                            : Engine::Unavailable, std::memory_order_relaxed);
    dryDelay_.assign(effect_.initialDelay, {});
    dryCursor_ = 0;
    wet_.reserve(effect_.initialDelay + kSliceSize + 2 * effect_.initialDelay);
    wet_.assign(effect_.initialDelay, 0.f);
    active_ = true;
  }

  static void __cdecl processReplacing(Effect* effect, float** inputs,
                                       float** outputs, Int frames) {
    from(effect).process(inputs, outputs, frames, false);
  }

  static void __cdecl processAccumulating(Effect* effect, float** inputs,
                                          float** outputs, Int frames) {
    from(effect).process(inputs, outputs, frames, true);
  }

  void process(float** inputs, float** outputs, Int frames, bool accumulate) noexcept {
    if (!outputs || frames <= 0) return;
    const auto* left = inputs ? inputs[0] : nullptr;
    const auto* right = inputs ? inputs[1] : nullptr;
    const bool bypass = bypass_.load(std::memory_order_relaxed)
                       || hostBypass_.load(std::memory_order_relaxed);
    for (Int offset = 0; offset < frames;) {
      const Int count = std::min(kSliceSize, frames - offset);
      if (active_ && denoiser_) {
        try {
          for (Int i = 0; i < count; ++i) {
            const Int frame = offset + i;
            mono_[i] = left && right ? 0.5f * (left[frame] + right[frame])
                                    : left ? left[frame] : right ? right[frame] : 0.f;
          }
          denoiser_->process(mono_.data(), count, wet_);
        } catch (...) {
          active_ = false;
        }
      }
      const auto wetCount = std::min<std::size_t>(count, wet_.size());
      for (Int i = 0; i < count; ++i) {
        const Int frame = offset + i;
        std::array<float, 2> samples{left ? left[frame] : 0.f, right ? right[frame] : 0.f};
        if (active_) {
          std::swap(samples, dryDelay_[dryCursor_]);
          if (++dryCursor_ == dryDelay_.size()) dryCursor_ = 0;
          if (!bypass && denoiser_) {
            const float wet = static_cast<std::size_t>(i) < wetCount ? wet_[i] : 0.f;
            samples = {wet, wet};
          }
        }
        for (int channel = 0; channel < 2; ++channel) {
          if (!outputs[channel]) continue;
          if (accumulate) outputs[channel][frame] += samples[channel];
          else outputs[channel][frame] = samples[channel];
        }
      }
      if (active_) wet_.erase(wet_.begin(), wet_.begin() + wetCount);
      offset += count;
    }
  }

#ifdef DPDFNET_LOCAL_ONNXRUNTIME
  // Keep the API loaded until after the denoiser has been destroyed.
  RuntimeModule runtime_;
#endif
  Effect effect_{};
  Dispatch host_;
  HWND editor_ = nullptr;
  EditorRect editorRect_{};
  Engine displayedEngine_ = Engine::Waiting;
  std::atomic<Engine> engine_{Engine::Waiting};
  std::atomic<bool> bypass_{false};
  std::atomic<bool> hostBypass_{false};
  char programName_[24] = "Default";
  Int sampleRate_ = kModelRate;
  bool active_ = false;
  std::unique_ptr<DpdfDenoiser> denoiser_;
  std::array<float, kSliceSize> mono_{};
  std::vector<float> wet_;
  std::vector<std::array<float, 2>> dryDelay_;
  std::size_t dryCursor_ = 0;
};

} // namespace

extern "C" __declspec(dllexport) vst2::Effect* __cdecl VSTPluginMain(vst2::Dispatch host) {
  if (!host || !host(nullptr, vst2::kHostVersion, 0, 0, nullptr, 0.f)) return nullptr;
  try {
    return (new Plugin(host))->effect();
  } catch (...) {
    return nullptr;
  }
}
