cmake_minimum_required(VERSION 3.21)
project(DPDFNetVST2 LANGUAGES C CXX)

if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "DPDFNet VST2 requires a 64-bit Windows toolchain")
endif()

enable_language(RC)
set(ONNXRUNTIME_NUGET_ROOT "" CACHE PATH "Extracted Microsoft.ML.OnnxRuntime NuGet package")
if(ONNXRUNTIME_NUGET_ROOT)
  set(ORT_HEADERS "${ONNXRUNTIME_NUGET_ROOT}/build/native/include")
  set(ORT_NATIVE "${ONNXRUNTIME_NUGET_ROOT}/runtimes/win-x64/native")
  foreach(required_file IN ITEMS
      "${ORT_HEADERS}/onnxruntime_cxx_api.h"
      "${ORT_NATIVE}/onnxruntime.lib"
      "${ORT_NATIVE}/onnxruntime.dll")
    if(NOT EXISTS "${required_file}")
      message(FATAL_ERROR "Missing ONNX Runtime NuGet file: ${required_file}")
    endif()
  endforeach()

  # The shared DSP headers include ONNX Runtime headers under onnxruntime/.
  set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ort-include")
  file(COPY "${ORT_HEADERS}/" DESTINATION "${ORT_INCLUDE_DIR}/onnxruntime")
  add_library(onnxruntime::onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ORT_NATIVE}/onnxruntime.lib"
    IMPORTED_LOCATION "${ORT_NATIVE}/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")
else()
  find_package(onnxruntime CONFIG REQUIRED)
endif()

set(DSP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/dsp")
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets/models")
set(DPDF8_MODEL "${MODEL_DIR}/dpdfnet8_48khz_hr.onnx")
if(NOT EXISTS "${DPDF8_MODEL}")
  message(FATAL_ERROR "Missing model: ${DPDF8_MODEL}")
endif()

configure_file(src/models.rc.in models.rc @ONLY)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/models.rc"
  PROPERTIES OBJECT_DEPENDS "${DPDF8_MODEL}")

set(NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dpdf-native")
set(NATIVE_SIMD
  "${NATIVE_DIR}/runtime/avx2.c"
  "${NATIVE_DIR}/runtime/int8.c"
  "${NATIVE_DIR}/runtime/fp16.c")
add_library(dpdf_native STATIC
  "${NATIVE_DIR}/runtime/dpdf_dprnn.c"
  "${NATIVE_DIR}/runtime/full_ops.c"
  "${NATIVE_DIR}/runtime/extended_ops.c"
  "${NATIVE_DIR}/models/dpdfnet8_48khz_hr/generated_model.c"
  ${NATIVE_SIMD})
target_compile_features(dpdf_native PRIVATE c_std_11)
target_compile_definitions(dpdf_native PRIVATE DPDF_X86_DISPATCH)
target_include_directories(dpdf_native PUBLIC "${NATIVE_DIR}/runtime")
if(MSVC)
  set_source_files_properties(${NATIVE_SIMD} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  target_compile_options(dpdf_native PRIVATE -ffp-contract=off)
  set_source_files_properties(${NATIVE_SIMD} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()

add_library(DPDFNet MODULE
  src/Plugin.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/models.rc"
  "${DSP_DIR}/DpdfDenoiser.cpp"
  "${DSP_DIR}/DpdfModel.cpp"
  "${DSP_DIR}/StftProcessor.cpp"
  "${DSP_DIR}/embedded_model.cpp"
  "${DSP_DIR}/resample.cc"
  "${DSP_DIR}/rfft.cc"
)
target_compile_features(DPDFNet PRIVATE cxx_std_17)
target_compile_definitions(DPDFNet PRIVATE NOMINMAX UNICODE _UNICODE)
if(ONNXRUNTIME_NUGET_ROOT)
  target_compile_definitions(DPDFNet PRIVATE DPDFNET_LOCAL_ONNXRUNTIME ORT_API_MANUAL_INIT)
endif()
target_include_directories(DPDFNet PRIVATE "${DSP_DIR}" "${NATIVE_DIR}/models")
target_link_libraries(DPDFNet PRIVATE onnxruntime::onnxruntime dpdf_native bcrypt user32)
set_target_properties(DPDFNet PROPERTIES PREFIX "")
if(ONNXRUNTIME_NUGET_ROOT)
  file(GLOB ORT_RUNTIME_DLLS CONFIGURE_DEPENDS "${ORT_NATIVE}/*.dll")
  add_custom_command(TARGET DPDFNet POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      ${ORT_RUNTIME_DLLS} "$<TARGET_FILE_DIR:DPDFNet>"
    VERBATIM)
endif()
