// AIPreprocess.c — see AIPreprocess.h for the contract.

#include "AIPreprocess.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// PyTorch's antialiased bilinear ("triangle") filter.
static float AITriangle(float x) {
    if (x < 0.0f) x = -x;
    return x < 1.0f ? 1.0f - x : 0.0f;
}

// Mirrors ATen's _compute_indices_min_size_weights_aa (align_corners=False,
// no explicit scale factor), which both the CPU and CUDA kernels use.
static int AIAxisInit(AIAxisWeights *a, int inSize, int outSize) {
    memset(a, 0, sizeof *a);
    if (inSize <= 0 || outSize <= 0) return -1;

    const float scale    = (float)inSize / (float)outSize;
    const float support  = scale >= 1.0f ? scale : 1.0f;     // interp_size(2) * 0.5 * scale
    const float invScale = scale >= 1.0f ? 1.0f / scale : 1.0f;

    a->inSize  = inSize;
    a->outSize = outSize;
    a->maxTaps = (int)ceilf(support) * 2 + 1;
    a->start   = (int *)calloc((size_t)outSize, sizeof(int));
    a->count   = (int *)calloc((size_t)outSize, sizeof(int));
    a->weights = (float *)calloc((size_t)outSize * (size_t)a->maxTaps, sizeof(float));
    if (!a->start || !a->count || !a->weights) return -1;

    for (int i = 0; i < outSize; i++) {
        const float center = scale * ((float)i + 0.5f);
        int lo = (int)(center - support + 0.5f);
        if (lo < 0) lo = 0;
        int hi = (int)(center + support + 0.5f);
        if (hi > inSize) hi = inSize;
        int n = hi - lo;
        if (n > a->maxTaps) n = a->maxTaps;

        float *w = a->weights + (size_t)i * (size_t)a->maxTaps;
        float total = 0.0f;
        for (int j = 0; j < n; j++) {
            w[j] = AITriangle(((float)(j + lo) - center + 0.5f) * invScale);
            total += w[j];
        }
        if (total != 0.0f) for (int j = 0; j < n; j++) w[j] /= total;
        a->start[i] = lo;
        a->count[i] = n;
    }
    return 0;
}

static void AIAxisFree(AIAxisWeights *a) {
    free(a->start);
    free(a->count);
    free(a->weights);
    memset(a, 0, sizeof *a);
}

int AIResamplerInit(AIResampler *r, int inW, int inH, int outW, int outH) {
    memset(r, 0, sizeof *r);
    if (AIAxisInit(&r->x, inW, outW) != 0 || AIAxisInit(&r->y, inH, outH) != 0) {
        AIResamplerFree(r);
        return -1;
    }
    r->rowTmp = (float *)malloc((size_t)inH * (size_t)outW * 3 * sizeof(float));
    if (!r->rowTmp) {
        AIResamplerFree(r);
        return -1;
    }
    return 0;
}

void AIResamplerFree(AIResampler *r) {
    AIAxisFree(&r->x);
    AIAxisFree(&r->y);
    free(r->rowTmp);
    r->rowTmp = NULL;
}

void AIResizeToGray(const AIResampler *r, const uint8_t *src, size_t rowBytes,
                    int rIdx, int gIdx, int bIdx, uint8_t *dst) {
    const int inH = r->y.inSize, outW = r->x.outSize, outH = r->y.outSize;
    const int tapsX = r->x.maxTaps, tapsY = r->y.maxTaps;

    // Horizontal pass: every source row -> outW RGB floats.
    for (int y = 0; y < inH; y++) {
        const uint8_t *row = src + (size_t)y * rowBytes;
        float *out = r->rowTmp + (size_t)y * (size_t)outW * 3;
        for (int i = 0; i < outW; i++) {
            const float *w = r->x.weights + (size_t)i * (size_t)tapsX;
            const uint8_t *p = row + (size_t)r->x.start[i] * 4;
            float sr = 0.0f, sg = 0.0f, sb = 0.0f;
            for (int j = 0, n = r->x.count[i]; j < n; j++, p += 4) {
                sr += w[j] * (float)p[rIdx];
                sg += w[j] * (float)p[gIdx];
                sb += w[j] * (float)p[bIdx];
            }
            out[i * 3 + 0] = sr;
            out[i * 3 + 1] = sg;
            out[i * 3 + 2] = sb;
        }
    }

    // Vertical pass, uint8 rounding, then torchvision's grayscale.
    for (int o = 0; o < outH; o++) {
        const float *w = r->y.weights + (size_t)o * (size_t)tapsY;
        const int y0 = r->y.start[o], n = r->y.count[o];
        for (int i = 0; i < outW; i++) {
            float sr = 0.0f, sg = 0.0f, sb = 0.0f;
            for (int j = 0; j < n; j++) {
                const float *p = r->rowTmp + ((size_t)(y0 + j) * (size_t)outW + (size_t)i) * 3;
                sr += w[j] * p[0];
                sg += w[j] * p[1];
                sb += w[j] * p[2];
            }
            // resize: round-half-even, back to uint8 (weights are >= 0 and sum to 1)
            float R = nearbyintf(sr), G = nearbyintf(sg), B = nearbyintf(sb);
            R = R < 0 ? 0 : (R > 255 ? 255 : R);
            G = G < 0 ? 0 : (G > 255 ? 255 : G);
            B = B < 0 ? 0 : (B > 255 ? 255 : B);
            // rgb_to_grayscale: float32 maths, truncating cast
            float l = R * 0.2989f;
            l = l + G * 0.587f;
            l = l + B * 0.114f;
            dst[(size_t)o * (size_t)outW + (size_t)i] = (uint8_t)(l < 0 ? 0 : (l > 255 ? 255 : l));
        }
    }
}

uint8_t AIDiffValue(uint8_t cur, uint8_t prev) {
    int a = ((int)cur - (int)prev) * 127;
    int q = a / 255;
    if ((a % 255) != 0 && a < 0) q--;            // floor division
    int s = q + 127;
    return (uint8_t)(s < 0 ? 0 : (s > 255 ? 255 : s));
}
