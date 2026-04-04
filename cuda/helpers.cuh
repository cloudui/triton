#pragma once

__device__ __forceinline__ float sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
