#pragma once
#include <cstddef>

namespace dpdfvst {

// Returns the embedded ONNX model and its size, or nullptr if unavailable.
const unsigned char* GetEmbeddedModelData(std::size_t* outSize);

} // namespace dpdfvst
