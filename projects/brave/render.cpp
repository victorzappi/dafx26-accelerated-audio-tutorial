/*
 * Copyright 2026 Victor Zappi
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

// brave: streams drums.wav through a QNN model, period by period, along
// with the recurrent cache tensors the model needs alongside its single
// audio input/output.
//
// --qnn-backend and --qnn-system are passed at runtime, since those paths
// are environment-specific. The model itself is a single fixed asset for
// this project (brave_acoustic_drumset_1024.dlc, see kModelFile below) and
// is expected next to drums.wav in the current working directory.
//
// This code example uses this sound from freesound:
// TECH DRUMS.wav by shpira -- https://freesound.org/s/323623/ -- License: Creative Commons 0
// the original file has been renamed to drums.wav and exported as a mono, 16-bit track
//
// The input audio file is not bundled with the source -- copy it into the
// current working directory you launch the engine from (not necessarily
// the executable's own directory) before running.

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>

// clean-room QNN wrapper (public-API only)
#include "QnnModel.h"

// AR includes
#include "optparse.h"
#include "render.h"
#include "AudioFile.h"

// Prints render() load against the period's time budget (period_size / sample_rate)
// to stderr, about once a second. Comment out to disable.
//#define PROFILE_RENDER

// App parameters set by CLI args
std::string backendPath;
std::string systemLibraryPath;

std::unique_ptr<ar::qnn::QnnModel> model;

// App-specific variables
// This model has a single audio input/output plus a set of recurrent cache
// tensors. Audio sits at a fixed slot among the graph's inputs/outputs;
// everything else is cache and gets fed back from output to input every
// render() call. These indices are a property of this specific model (it's
// the same one used in the reference QNN render.cpp this project is based
// on), not something the QNN API exposes -- there's no way to discover them
// generically.
const uint32_t g_graphIdx = 0;
const int g_audioInputIndex = 1;
const int g_audioOutputIndex = 47;

// This model's batch size is fixed at export time to this period size.
const unsigned int kExpectedPeriodSize = 1024;

// The model is a single fixed asset for this project, expected in the
// current working directory (same as drums.wav).
const char *kModelFile = "brave_acoustic_drumset_1024.dlc";

// One flattened float buffer per input/output tensor of the graph.
float **g_inputDataBuffers = nullptr;
float **g_outputDataBuffers = nullptr;
size_t g_numInputs = 0;
size_t g_numOutputs = 0;
std::vector<size_t> g_outputNumElements; // cached from setup(), avoids querying the model in render()

// drums.wav playback (mono only)
const char *kInputFile = "drums.wav";
std::vector<std::vector<float>> fileData; // fileData[0] = mono channel
unsigned int fileFrames = 0;
unsigned int readPos = 0;

void showHelp()
{
    std::cout
        << "\nDESCRIPTION:\n"
        << "------------\n"
        << "Streams drums.wav through the bundled brave_acoustic_drumset_1024.dlc\n"
        << "QNN model, period by period. Both files are expected in the current\n"
        << "working directory.\n"
        << "\n\n"
        << "REQUIRED ARGUMENTS:\n"
        << "-------------------\n"
        << "  --qnn-backend      <FILE>   Path to a QNN backend to execute the model.\n"
        << "\n"
        << "  --qnn-system       <FILE>   Path to the QNN System library (libQnnSystem.so),\n"
        << "                              needed to load the model from its .dlc container.\n"
        << "\n\n"
        << "OPTIONAL ARGUMENTS:\n"
        << "-------------------\n"
        << "  --project-help              Show this help message.\n"
        << std::endl;
}

void processCommandLine(char **argv)
{
    // long-only option ids: start past the ASCII range so optparse treats them as
    // long-form only, i.e. there are no single-char short options
    // short-form may easily clash with the many arguments dealt with by main
    enum
    {
        OPT_QNN_BACKEND = 256,
        OPT_QNN_SYSTEM,
        OPT_PROJECT_HELP,
    };

    int c;
    struct optparse opts;
    struct optparse_long long_options[] = {
        {"qnn-backend", OPT_QNN_BACKEND, OPTPARSE_REQUIRED},
        {"qnn-system", OPT_QNN_SYSTEM, OPTPARSE_REQUIRED},
        {"project-help", OPT_PROJECT_HELP, OPTPARSE_NONE},
        {0, 0, OPTPARSE_NONE}};

    optparse_init(&opts, argv);
    while ((c = optparse_long(&opts, long_options, NULL)) != -1)
    {
        switch (c)
        {
        case OPT_QNN_BACKEND:
            backendPath = opts.optarg;
            break;
        case OPT_QNN_SYSTEM:
            systemLibraryPath = opts.optarg;
            break;
        case OPT_PROJECT_HELP:
            showHelp();
            std::exit(EXIT_SUCCESS);
            break;
        case '?':
            std::cerr << "ERROR: Invalid argument passed: " << argv[opts.optind - 1]
                      << "\nPlease check the Arguments section in the description below.\n";
            showHelp();
            std::exit(EXIT_FAILURE);
            break;
        }
    }
}

int setup(struct audio_ctx *ctx, void *user_data)
{
    if (ctx->period_size != kExpectedPeriodSize)
    {
        std::cerr << "brave: this model requires period_size == " << kExpectedPeriodSize
                  << " (got " << ctx->period_size << ")\n";
        return EXIT_FAILURE;
    }

    processCommandLine((char **)user_data);

    if (backendPath.empty() || systemLibraryPath.empty())
    {
        std::cerr << "brave: --qnn-backend and --qnn-system are both required\n";
        return EXIT_FAILURE;
    }

    model.reset(new ar::qnn::QnnModel(backendPath, kModelFile, systemLibraryPath));
    if (!model->load())
    {
        std::cerr << "brave: failed to load model\n";
        return EXIT_FAILURE;
    }

    const std::vector<ar::qnn::TensorInfo> &inputs = model->inputs(g_graphIdx);
    const std::vector<ar::qnn::TensorInfo> &outputs = model->outputs(g_graphIdx);

    // The cache copy-back below assumes the input and output tensor lists
    // are the same length (see the loop in render()).
    if (inputs.size() != outputs.size())
    {
        std::cerr << "brave: model has different amounts of input and output tensors ("
                  << inputs.size() << " vs. " << outputs.size() << ")\n";
        return EXIT_FAILURE;
    }
    if (g_audioInputIndex < 0 || (size_t)g_audioInputIndex >= inputs.size() ||
        g_audioOutputIndex < 0 || (size_t)g_audioOutputIndex >= outputs.size())
    {
        std::cerr << "brave: audio tensor index out of range for this model ("
                  << inputs.size() << " inputs, " << outputs.size() << " outputs)\n";
        return EXIT_FAILURE;
    }

    // Make sure the model's audio I/O dimensions are what we expect: one
    // sample per period_size frame, mono, batched as [1, 1, period_size].
    const std::vector<uint32_t> &inDims = inputs[g_audioInputIndex].dims;
    const std::vector<uint32_t> &outDims = outputs[g_audioOutputIndex].dims;

    if (inDims.size() != 3 || inDims[0] != 1 || inDims[1] != 1 || inDims[2] != ctx->period_size)
    {
        std::cerr << "Given model has incorrect audio input dimensions (expected [1, 1, " << ctx->period_size << "]): [ ";
        for (uint32_t dim : inDims)
            std::cerr << dim << " ";
        std::cerr << "]\n";
        return EXIT_FAILURE;
    }
    if (outDims.size() != 3 || outDims[0] != 1 || outDims[1] != 1 || outDims[2] != ctx->period_size)
    {
        std::cerr << "Given model has incorrect audio output dimensions (expected [1, 1, " << ctx->period_size << "]): [ ";
        for (uint32_t dim : outDims)
            std::cerr << dim << " ";
        std::cerr << "]\n";
        return EXIT_FAILURE;
    }

    // Allocate one flat float buffer per input/output tensor, zero-initialized
    // (cache tensors start at 0; the audio input is overwritten every render()).
    g_numInputs = inputs.size();
    g_numOutputs = outputs.size();
    g_inputDataBuffers = (float **)calloc(g_numInputs, sizeof(float *));
    g_outputDataBuffers = (float **)calloc(g_numOutputs, sizeof(float *));
    if (!g_inputDataBuffers || !g_outputDataBuffers)
        return EXIT_FAILURE;
    for (size_t i = 0; i < g_numInputs; ++i)
        g_inputDataBuffers[i] = (float *)calloc(inputs[i].numElements, sizeof(float));
    g_outputNumElements.resize(g_numOutputs);
    for (size_t i = 0; i < g_numOutputs; ++i)
    {
        g_outputDataBuffers[i] = (float *)calloc(outputs[i].numElements, sizeof(float));
        g_outputNumElements[i] = outputs[i].numElements;
    }

    // Load drums.wav -- must be mono, same loading pattern as file_player.
    fileData = AudioFileUtilities::load(kInputFile);
    if (fileData.empty() || fileData[0].empty())
    {
        std::cerr << "brave: failed to load audio file '" << kInputFile << "' (expected in the current working directory)\n";
        return EXIT_FAILURE;
    }
    if (fileData.size() != 1)
    {
        std::cerr << "brave: '" << kInputFile << "' has " << fileData.size() << " channel(s), expected mono (1)\n";
        return EXIT_FAILURE;
    }

    fileFrames = (unsigned int)fileData[0].size();
    readPos = 0;

    std::cout << "brave: loaded '" << kInputFile << "' (mono, " << fileFrames << " frames)\n";
    std::cout << "brave: model has " << g_numInputs << " input(s) and " << g_numOutputs << " output(s)\n";

    return EXIT_SUCCESS;
}

void render(struct audio_ctx *ctx, void *user_data)
{
#ifdef PROFILE_RENDER
    auto renderStart = std::chrono::high_resolution_clock::now();
#endif

    // 1. write drums.wav samples into the audio input tensor, looping the file
    for (size_t frame = 0; frame < ctx->period_size; ++frame)
    {
        g_inputDataBuffers[g_audioInputIndex][frame] = fileData[0][readPos];
        if (++readPos == fileFrames)
            readPos = 0;
    }

    // 2. run the model
    if (model->execute(g_graphIdx, g_inputDataBuffers, g_outputDataBuffers))
    {
        // 3. write the audio output tensor to the audio buffer (same sample to every channel)
        for (size_t frame = 0; frame < ctx->period_size; ++frame)
        {
            const float sample = g_outputDataBuffers[g_audioOutputIndex][frame];
            for (unsigned int channel = 0; channel < ctx->channels; ++channel)
                ctx->audio_out[(frame * ctx->channels) + channel] = sample;
        }

        // 4. copy cache outputs back to cache inputs for the next render() call.
        // This model's cache tensors are contiguous on the output side (with
        // audio appended at g_audioOutputIndex), but the input side has a gap at
        // g_audioInputIndex where the audio tensor sits -- so every cache index
        // at or past that gap shifts up by one on the input side.
        for (size_t outIdx = 0; outIdx < g_numOutputs; ++outIdx)
        {
            if ((int)outIdx == g_audioOutputIndex)
                continue;

            size_t inIdx = (int)outIdx < g_audioInputIndex ? outIdx : outIdx + 1;

            std::copy_n(g_outputDataBuffers[outIdx],
                        g_outputNumElements[outIdx],
                        g_inputDataBuffers[inIdx]);
        }
    }

#ifdef PROFILE_RENDER
    auto renderEnd = std::chrono::high_resolution_clock::now();
    float renderUs = std::chrono::duration<float, std::micro>(renderEnd - renderStart).count();
    static int printCounter = 0;
    if (++printCounter >= (int)(ctx->sample_rate / ctx->period_size))
    {
        float budgetUs = (float)ctx->period_size / (float)ctx->sample_rate * 1e6f;
        float load = renderUs / budgetUs * 100.0f;
        fprintf(stderr, "brave: render %.0f us | budget %.0f us | load %.1f%%\n", renderUs, budgetUs, load);
        printCounter = 0;
    }
#endif
}

void cleanup(struct audio_ctx *ctx, void *user_data)
{
    if (g_inputDataBuffers != nullptr)
    {
        for (size_t i = 0; i < g_numInputs; ++i)
            free(g_inputDataBuffers[i]);
        free(g_inputDataBuffers);
        g_inputDataBuffers = nullptr;
    }
    if (g_outputDataBuffers != nullptr)
    {
        for (size_t i = 0; i < g_numOutputs; ++i)
            free(g_outputDataBuffers[i]);
        free(g_outputDataBuffers);
        g_outputDataBuffers = nullptr;
    }

    // releases the backend, context, model and libraries (see QnnModel destructor)
    model.reset();
}
