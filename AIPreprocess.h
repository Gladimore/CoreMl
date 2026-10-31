// AIPreprocess.h — frame preprocessing that reproduces the training pipeline.
//
// Plain C with no Apple dependencies, so the exact same code is unit-tested
// against PyTorch on Linux (tools/test_preprocess.py) and compiled into the dylib.
//
// Training (build_dataset_*_gpu.py) did, per decoded RGB frame:
//   1. torchvision resize to 128x128, bilinear + antialias, on float32,
//      then round-half-even back to uint8
//   2. torchvision rgb_to_grayscale: 0.2989 R + 0.587 G + 0.114 B in float32,
//      then truncated to uint8 (so pure white maps to 254, not 255)
// AIResizeToGray() does exactly that.

#ifndef AI_PREPROCESS_H
#define AI_PREPROCESS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Precomputed antialiased-bilinear weights for one axis (in -> out).
typedef struct {
    int    inSize, outSize, maxTaps;
    int   *start;    // [outSize] first input index
    int   *count;    // [outSize] taps used
    float *weights;  // [outSize * maxTaps], normalised
} AIAxisWeights;

// Precomputed resampler for one source size. Rebuild when the source size changes.
typedef struct {
    AIAxisWeights x, y;
    float *rowTmp;   // [inHeight * outWidth * 3] horizontal pass result
} AIResampler;

// Returns 0 on success. outW/outH are the model input size (128).
int  AIResamplerInit(AIResampler *r, int inW, int inH, int outW, int outH);
void AIResamplerFree(AIResampler *r);

// src: 4-byte pixels, `rowBytes` per row. rIdx/gIdx/bIdx give the byte offset of
// each channel inside a pixel (RGBA: 0,1,2; BGRA: 2,1,0).
// dst: outW*outH uint8 grayscale, identical to the training dataset's frames.bin.
void AIResizeToGray(const AIResampler *r, const uint8_t *src, size_t rowBytes,
                    int rIdx, int gIdx, int bIdx, uint8_t *dst);

// Causal model only: diff channel used by the causal dataset builder.
// clamp(floorDiv((cur - prev) * 127, 255) + 127, 0, 255)
uint8_t AIDiffValue(uint8_t cur, uint8_t prev);

#ifdef __cplusplus
}
#endif

#endif
