// RUN: triton-opt %s -split-input-file --tritonamdgpu-ws-lowering | FileCheck %s

// CHECK-NOT: triton_nvidia_gpu.create_token
// CHECK-LABEL: foo1
// CHECK-NOT: create_token
//

#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @foo1() {
    %c0_i32_5 = arith.constant 0 : i32
    %false = arith.constant false
    %2 = triton_nvidia_gpu.create_token {num = 1 : i32} : tensor<1x!triton_nvidia_gpu.token>
    triton_nvidia_gpu.producer_acquire %2, %c0_i32_5, %false : tensor<1x!triton_nvidia_gpu.token>, i32, i1
    triton_nvidia_gpu.producer_commit %2, %c0_i32_5 : tensor<1x!triton_nvidia_gpu.token>, i32
    tt.return
  }
}
