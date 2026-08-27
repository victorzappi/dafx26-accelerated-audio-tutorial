/*
 * Copyright 2026 Victor Zappi
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#include "render.h"
#include "reverb_ir_f32.h"
#include "AudioFile.h"
#include "Fir.h"

#include <vector>
#include <cstdio>

const char *kInputFile = "dry_percussions.wav";
const float kInputGain = 0.4f;   // IR coefficients sum to ~82 in absolute
                                  // value; input would clip at full-scale

Fir fir;

// one vector per file channel (fileData[0] = left, fileData[1] = right)
std::vector<std::vector<float>> fileData;
unsigned int fileFrames = 0;
unsigned int readPos = 0;

// FIR input/output: one heap block per channel (row 0 = left, row 1 =
// right), sized once in setup() so render() never allocates. This is
// exactly the float** shape Fir::process() requires (de-interleaved,
// filtered in place) -- no separate storage/view pair needed.
float *firBuffer[2];

int setup(struct audio_ctx *ctx, void *user_data)
{
    if (ctx->channels != 2) {
        fprintf(stderr, "file_player_reverb: engine is configured for %u channel(s), this project requires stereo (2)\n", ctx->channels);
        return -1;
    }
    if (ctx->sample_rate != REVERB_IR_SAMPLERATE) {
        fprintf(stderr, "file_player_reverb: project sample rate (%u Hz) does not match the impulse response's rate (%u Hz)\n",
                ctx->sample_rate, (unsigned int)REVERB_IR_SAMPLERATE);
        return -1;
    }

    fileData = AudioFileUtilities::load(kInputFile);
    if (fileData.empty() || fileData[0].empty()) {
        fprintf(stderr, "file_player_reverb: failed to load audio file '%s' (expected in the current working directory)\n", kInputFile);
        return -1;
    }
    if (fileData.size() != 2) {
        fprintf(stderr, "file_player_reverb: '%s' has %zu channel(s), expected stereo (2)\n", kInputFile, fileData.size());
        return -1;
    }

    fileFrames = (unsigned int)fileData[0].size();
    readPos = 0;

    firBuffer[0] = new float[ctx->period_size];
    firBuffer[1] = new float[ctx->period_size];

    if (fir.setup(REVERB_IR_NUM_TAPS, ctx->channels, ctx->period_size) != 0) {
        fprintf(stderr, "file_player_reverb: Fir::setup failed\n");
        return -1;
    }
    if (fir.setCoefficients(kReverbIr, REVERB_IR_NUM_TAPS) != 0) {
        fprintf(stderr, "file_player_reverb: Fir::setCoefficients failed\n");
        return -1;
    }

    printf("file_player_reverb: loaded '%s' (stereo, %u frames)\n", kInputFile, fileFrames);

    return 0;
}

void render(struct audio_ctx *ctx, void *user_data)
{
    // fill the FIR input, one channel row at a time, with the dry (gained) file signal
    for (unsigned int c = 0; c < 2; c++) {
        unsigned int p = readPos;
        for (unsigned int n = 0; n < ctx->period_size; n++) {
            firBuffer[c][n] = fileData[c][p] * kInputGain;
            if (++p == fileFrames)
                p = 0;
        }
    }
    readPos = (readPos + ctx->period_size) % fileFrames;

    fir.process(firBuffer, ctx->period_size);   // float**, de-interleaved, filtered in place

    // re-interleave the filtered signal into the engine's output buffer
    for (unsigned int n = 0; n < ctx->period_size; n++)
        for (unsigned int c = 0; c < 2; c++)
            ctx->audio_out[n * ctx->channels + c] = firBuffer[c][n];
}

void cleanup(struct audio_ctx *ctx, void *user_data)
{
    delete[] firBuffer[0];
    delete[] firBuffer[1];
}
