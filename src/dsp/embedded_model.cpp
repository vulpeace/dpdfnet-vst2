#include "embedded_model.h"

#include <windows.h>

namespace dpdfvst {

const unsigned char* GetEmbeddedModelData(std::size_t* outSize) {
  if (!outSize) return nullptr;
  *outSize = 0;
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&GetEmbeddedModelData), &module)) {
    return nullptr;
  }
  const auto resource = FindResourceW(module, MAKEINTRESOURCEW(101), RT_RCDATA);
  if (!resource) return nullptr;
  const auto loaded = LoadResource(module, resource);
  if (!loaded) return nullptr;
  const auto* data = static_cast<const unsigned char*>(LockResource(loaded));
  const auto size = SizeofResource(module, resource);
  if (!data || !size) return nullptr;
  *outSize = size;
  return data;
}

} // namespace dpdfvst
