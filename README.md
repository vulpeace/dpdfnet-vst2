# DPDFNet VST2

DPDFNet is a 64-bit Windows VST2.4 noise suppression plug-in. It mixes stereo input to mono, denoises the mix with DPDFNet8 48 kHz HR, and sends the same signal to both outputs. The editor shows the active inference engine and offers **Bypass**, which preserves stereo and compensates for the processing delay.

## Build on Windows

Install Visual Studio with **Desktop development with C++** and CMake. The examples use the Visual Studio 2026 generator. For Visual Studio 2022, use `Visual Studio 17 2022` in both configure commands. Run the commands from the repository root in PowerShell.

### Microsoft NuGet runtime

This build uses the prebuilt x64 ONNX Runtime package and avoids compiling ONNX Runtime from source.

```powershell
$package = Join-Path $PWD 'build-nuget/onnxruntime-package'
$zip = "$package.zip"
New-Item -ItemType Directory -Force (Split-Path $package) | Out-Null
curl.exe --fail --location --output "$zip" `
  'https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/1.32.0/microsoft.ml.onnxruntime.1.32.0.nupkg'
if ($LASTEXITCODE -ne 0) { throw 'ONNX Runtime download failed.' }
Expand-Archive -LiteralPath $zip -DestinationPath $package -Force -ErrorAction Stop

cmake -S . -B build-nuget -G 'Visual Studio 18 2026' -A x64 "-DONNXRUNTIME_NUGET_ROOT=$package"
if ($LASTEXITCODE -eq 0) { cmake --build build-nuget --config Release }
```

Copy all three DLLs from `build-nuget/Release` into the host's VST2 plug-in directory and rescan. Keep `onnxruntime.dll` and `onnxruntime_providers_shared.dll` beside `DPDFNet.dll`.

### Static ONNX Runtime with vcpkg

Set `VCPKG_ROOT` to your bootstrapped vcpkg installation. This build requires access to the source archives used by the pinned vcpkg ports.

```powershell
cmake -S . -B build -G 'Visual Studio 18 2026' -A x64 `
  "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows-static-md
if ($LASTEXITCODE -eq 0) { cmake --build build --config Release }
```

Copy `build/Release/DPDFNet.dll` into the host's VST2 plug-in directory and rescan. Both builds embed the model and use the Microsoft Visual C++ runtime.

### Native INT8 inference

Both builds automatically prefer native selective INT8 inference on x64 CPUs with AVX2 and FMA when the matching weight file is beside `DPDFNet.dll`. Run this after building with NuGet, then copy the `.weights.f32` file along with the DLLs.

```powershell
.\scripts\fetch-native-weights.ps1 -Destination build-nuget/Release
```

For the static build, use `build/Release` as the destination. The plug-in checks the size and SHA-256 of the weight file. ONNX Runtime handles inference when the native engine is unavailable.

## Licenses

The plug-in and DSP adapted from Outsidepro Arts' VST3 project are covered by the [MIT license](LICENSE). The bundled [Ceva-IP DPDFNet8 model](https://github.com/ceva-ip/DPDFNet), [native runtime](third_party/dpdf-native/README.md), and the resampler components with Apache notices use the [Apache License 2.0](licenses/Apache-2.0.txt). ONNX Runtime is supplied separately and carries its own license and third-party notices.
