/*
 * Copyright 2026 Victor Zappi
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#include "render.h"
#include "AudioFile.h"

#include <vector>
#include <cstdio>

const char *kInputFile = "dry_percussions.wav";

// one vector per file channel (fileData[0] = left, fileData[1] = right)
std::vector<std::vector<float>> fileData;
unsigned int fileFrames = 0;
unsigned int readPos = 0;

int setup(struct audio_ctx *ctx, void *user_data)
{
    if (ctx->channels != 2) {
        fprintf(stderr, "file_player: engine is configured for %u channel(s), this project requires stereo (2)\n", ctx->channels);
        return -1;
    }

    fileData = AudioFileUtilities::load(kInputFile);
    if (fileData.empty() || fileData[0].empty()) {
        fprintf(stderr, "file_player: failed to load audio file '%s' (expected in the current working directory)\n", kInputFile);
        return -1;
    }
    if (fileData.size() != 2) {
        fprintf(stderr, "file_player: '%s' has %zu channel(s), expected stereo (2)\n", kInputFile, fileData.size());
        return -1;
    }

    fileFrames = (unsigned int)fileData[0].size();
    readPos = 0;

    printf("file_player: loaded '%s' (stereo, %u frames)\n", kInputFile, fileFrames);

    return 0;
}

void render(struct audio_ctx *ctx, void *user_data)
{
    for (unsigned int n = 0; n < ctx->period_size; n++) {
        ctx->audio_out[n * ctx->channels + 0] = fileData[0][readPos];
        ctx->audio_out[n * ctx->channels + 1] = fileData[1][readPos];
        if (++readPos == fileFrames)
            readPos = 0;
    }
}

void cleanup(struct audio_ctx *ctx, void *user_data)
{
}
