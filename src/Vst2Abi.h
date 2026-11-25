#pragma once

#include <cstddef>
#include <cstdint>

namespace vst2 {

struct Effect;
using Int = std::int32_t;
using Value = std::intptr_t;
using Dispatch = Value (__cdecl*)(Effect*, Int, Int, Value, void*, float);
using Process = void (__cdecl*)(Effect*, float**, float**, Int);
using SetParameter = void (__cdecl*)(Effect*, Int, float);
using GetParameter = float (__cdecl*)(Effect*, Int);
using ProcessDouble = void (__cdecl*)(Effect*, double**, double**, Int);

struct Effect {
  Int magic;
  Dispatch dispatcher;
  Process process;
  SetParameter setParameter;
  GetParameter getParameter;
  Int numPrograms;
  Int numParams;
  Int numInputs;
  Int numOutputs;
  Int flags;
  Value reserved1;
  Value reserved2;
  Int initialDelay;
  Int realQualities;
  Int offQualities;
  float ioRatio;
  void* object;
  void* user;
  Int uniqueId;
  Int version;
  Process processReplacing;
  ProcessDouble processDoubleReplacing;
  char future[56];
};

static_assert(sizeof(Effect) == 192);
static_assert(offsetof(Effect, initialDelay) == 80);
static_assert(offsetof(Effect, processReplacing) == 120);

constexpr Int kEffectMagic = 0x56737450; // VstP
constexpr Int kPluginId = 0x44504432;
constexpr Int kCanReplacing = 1 << 4;
constexpr Int kHasEditor = 1;

enum Opcode : Int {
  kOpen = 0,
  kClose = 1,
  kSetProgram = 2,
  kGetProgram = 3,
  kSetProgramName = 4,
  kGetProgramName = 5,
  kGetParamLabel = 6,
  kGetParamDisplay = 7,
  kGetParamName = 8,
  kSetSampleRate = 10,
  kSetBlockSize = 11,
  kMainsChanged = 12,
  kEditGetRect = 13,
  kEditOpen = 14,
  kEditClose = 15,
  kCanBeAutomated = 26,
  kGetProgramNameIndexed = 29,
  kGetPlugCategory = 35,
  kSetBypass = 44,
  kGetEffectName = 45,
  kGetVendorString = 47,
  kGetProductString = 48,
  kGetVendorVersion = 49,
  kCanDo = 51,
  kGetVstVersion = 58,
  kStartProcess = 71,
  kStopProcess = 72
};

constexpr Int kHostVersion = 1;
constexpr Int kHostAutomate = 0;
constexpr Int kHostIoChanged = 13;

} // namespace vst2
