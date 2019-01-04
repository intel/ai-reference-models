// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "data_reader.h"
#include "paddle/fluid/inference/paddle_inference_api.h"
#include "paddle/fluid/platform/profiler.h"
#include "stats.h"
#include <gflags/gflags.h>
#include <map>
#include <random>
#include <string>

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(data_list, "", "Path to a file with a list of image files.");
DEFINE_string(data_dir, "", "Path to a directory with image files.");
DEFINE_string(label_list, "", "Path to a file with a list of object names.");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(iterations, 1, "How many times to repeat run.");
DEFINE_int32(skip_batch_num, 0, "How many minibatches to skip in statistics.");
DEFINE_bool(use_fake_data, false, "Use fake data (1,2,...).");
DEFINE_bool(use_mkldnn, false, "Use MKL-DNN.");
DEFINE_bool(skip_passes, false, "Skip running passes.");
DEFINE_bool(enable_graphviz, false,
            "Enable graphviz to get .dot files with data flow graphs.");
DEFINE_bool(debug_display_images, false, "Show images in windows for debug.");
DEFINE_bool(with_labels, true, "The infer model do handle data labels.");
DEFINE_bool(one_file_params, false,
            "Parameters of the model are in one file 'params' and model in a "
            "file 'model'.");
DEFINE_bool(profile, false, "Turn on profiler for fluid");
DEFINE_int32(paddle_num_threads, 1,
             "Number of threads for each paddle instance.");
// dimensions of imagenet images are assumed as default:
DEFINE_int32(resize_size, 300,
             "Images are resized to make smaller side have length resize_size "
             "while keeping aspect ratio.");
DEFINE_int32(channels, 3, "Width of the image.");

namespace {
class Timer {
public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};
} // namespace

namespace paddle {

#define PRINT_OPTION(a)                                                        \
  do {                                                                         \
    std::cout << #a ": " << (FLAGS_##a) << std::endl;                          \
  } while (false)

void PrintInfo() {
  std::cout << std::endl
            << "--- Used Parameters: -----------------" << std::endl;
  PRINT_OPTION(batch_size);
  PRINT_OPTION(channels);
  PRINT_OPTION(data_dir);
  PRINT_OPTION(data_list);
  PRINT_OPTION(debug_display_images);
  PRINT_OPTION(enable_graphviz);
  PRINT_OPTION(infer_model);
  PRINT_OPTION(iterations);
  PRINT_OPTION(label_list);
  PRINT_OPTION(one_file_params);
  PRINT_OPTION(paddle_num_threads);
  PRINT_OPTION(profile);
  PRINT_OPTION(resize_size);
  PRINT_OPTION(skip_batch_num);
  PRINT_OPTION(skip_passes);
  PRINT_OPTION(use_fake_data);
  PRINT_OPTION(use_mkldnn);
  PRINT_OPTION(with_labels);
  std::cout << "--------------------------------------" << std::endl;
}

// Count elements of a tensor from its shape vector
size_t Count(const std::vector<int> &shapevec) {
  auto sum = shapevec.size() > 0 ? 1 : 0;
  for (unsigned int i = 0; i < shapevec.size(); ++i) {
    sum *= shapevec[i];
  }
  return sum;
}

paddle::PaddleTensor DefineInputData() {
  std::vector<int> shape;
  shape.push_back(FLAGS_batch_size);
  shape.push_back(FLAGS_channels);
  shape.push_back(FLAGS_resize_size);
  shape.push_back(FLAGS_resize_size);
  paddle::PaddleTensor input_data;
  input_data.name = "image";
  input_data.shape = shape;
  input_data.data.Resize(Count(shape) * sizeof(float));
  input_data.dtype = PaddleDType::FLOAT32;
  return input_data;
}

template <typename T> void fill_data(T *data, unsigned int count) {
  for (unsigned int i = 0; i < count; ++i) {
    *(data + i) = i;
  }
}

template <> void fill_data<float>(float *data, unsigned int count) {
  static unsigned int seed = std::random_device()();
  static std::minstd_rand engine(seed);
  float mean = 0;
  float std = 1;
  std::normal_distribution<float> dist(mean, std);
  for (unsigned int i = 0; i < count; ++i) {
    data[i] = dist(engine);
  }
}

void CreateFakeData(PaddleTensor &input_data, PaddleTensor &gt_label_tensor,
                    PaddleTensor &gt_bbox_tensor,
                    PaddleTensor &gt_difficult_tensor,
                    std::unique_ptr<DataReader> &reader) {
  fill_data<float>(static_cast<float *>(input_data.data.data()),
                   Count(input_data.shape));
  int labels_size = FLAGS_batch_size;
  if (FLAGS_with_labels) {
    // create fake labels
    fill_data<int64_t>(static_cast<int64_t *>(gt_label_tensor.data.data()),
                       labels_size);
  }
}

void InitializeReader(std::unique_ptr<DataReader> &reader,
                      bool convert_to_rgb) {
  reader.reset(new DataReader(FLAGS_data_list, FLAGS_data_dir, FLAGS_label_list,
                              FLAGS_resize_size, FLAGS_channels,
                              convert_to_rgb));
  if (!reader->SetSeparator('\t'))
    reader->SetSeparator(' ');
}

bool ReadNextBatch(PaddleTensor &input_data, PaddleTensor &gt_label_tensor,
                   PaddleTensor &gt_bbox_tensor,
                   PaddleTensor &gt_difficult_tensor,
                   std::unique_ptr<DataReader> &reader) {
  std::vector<std::vector<int64_t>> gt_label;
  std::vector<std::vector<std::vector<float>>> gt_bbox;
  std::vector<std::vector<int64_t>> gt_difficult;
  // return reader->nextbatch

  bool flag = reader->NextBatch(static_cast<float *>(input_data.data.data()),
                                gt_label, gt_bbox, gt_difficult,
                                FLAGS_batch_size, FLAGS_debug_display_images);
  if (flag == false)
    return false;
  // allocate paddle tensor memory
  gt_label_tensor.name = "gt_label";
  int gt_label_size_accum = 0;
  std::vector<size_t> lod;
  lod.push_back(0);
  for (int i = 0; i < gt_label.size(); i++) {
    gt_label_size_accum += gt_label[i].size();
    lod.push_back(gt_label_size_accum);
  }
  gt_label_tensor.shape = {gt_label_size_accum, 1};
  gt_label_tensor.data.Resize(gt_label_size_accum * sizeof(int64_t));
  gt_label_tensor.dtype = PaddleDType::INT64;
  gt_label_tensor.lod.clear();
  gt_label_tensor.lod.push_back(lod);

  gt_bbox_tensor.name = "gt_bbox";
  gt_bbox_tensor.shape = {gt_label_size_accum, 4};
  gt_bbox_tensor.data.Resize(gt_label_size_accum * 4 * sizeof(float));
  gt_bbox_tensor.dtype = PaddleDType::FLOAT32;
  gt_bbox_tensor.lod.clear();
  gt_bbox_tensor.lod.push_back(lod);

  gt_difficult_tensor.name = "gt_difficult";
  gt_difficult_tensor.shape = {gt_label_size_accum, 1};
  gt_difficult_tensor.data.Resize(gt_label_size_accum * sizeof(int64_t));
  gt_difficult_tensor.dtype = PaddleDType::INT64;
  gt_difficult_tensor.lod.clear();
  gt_difficult_tensor.lod.push_back(lod);

  // copy label vector to int pointer
  int64_t *datalabel = static_cast<int64_t *>(gt_label_tensor.data.data());
  for (int i = 0; i < gt_label.size(); i++) {
    for (int j = 0; j < gt_label[i].size(); j++) {
      (*datalabel) = gt_label[i][j];
      datalabel++;
    }
  }

  // copy to tensor
  float *databbox = static_cast<float *>(gt_bbox_tensor.data.data());
  for (int i = 0; i < gt_bbox.size(); i++) {
    for (int j = 0; j < gt_bbox[i].size(); j++) {
      for (int k = 0; k < 4; k++) {
        *databbox = gt_bbox[i][j][k];
        databbox++;
      }
    }
  }

  // copy difficult vector to int *
  int64_t *datadifficult =
      static_cast<int64_t *>(gt_difficult_tensor.data.data());
  for (int i = 0; i < gt_difficult.size(); i++) {
    for (int j = 0; j < gt_difficult[i].size(); j++) {
      (*datadifficult) = gt_difficult[i][j];
      datadifficult++;
    }
  }

  return true;
}

void PrepareConfig(contrib::AnalysisConfig &config) {
  if (FLAGS_one_file_params) {
    config.param_file = FLAGS_infer_model + "/params";
    config.prog_file = FLAGS_infer_model + "/model";
  } else {
    config.model_dir = FLAGS_infer_model;
  }
  config.use_gpu = false;
  config.device = 0;
  config.enable_ir_optim = !FLAGS_skip_passes;
  config.specify_input_name = false;
  config.SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
  if (FLAGS_use_mkldnn)
    config.EnableMKLDNN();

  // remove all passes so that we can add them in correct order
  for (int i = config.pass_builder()->AllPasses().size() - 1; i >= 0; i--)
    config.pass_builder()->DeletePass(i);

  // add passes
  config.pass_builder()->AppendPass("infer_clean_graph_pass");
  if (FLAGS_use_mkldnn) {
    // add passes to execute with MKL-DNN
    config.pass_builder()->AppendPass("mkldnn_placement_pass");
    config.pass_builder()->AppendPass("is_test_pass");
    config.pass_builder()->AppendPass("depthwise_conv_mkldnn_pass");
    config.pass_builder()->AppendPass("conv_bn_fuse_pass");
    config.pass_builder()->AppendPass("conv_eltwiseadd_bn_fuse_pass");
    config.pass_builder()->AppendPass("conv_bias_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("conv_elementwise_add_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("conv_relu_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("fc_fuse_pass");
  } else {
    // add passes to execute keeping the order - without MKL-DNN
    config.pass_builder()->AppendPass("conv_bn_fuse_pass");
    config.pass_builder()->AppendPass("fc_fuse_pass");
  }

  if (FLAGS_enable_graphviz)
    config.pass_builder()->TurnOnDebug();
}

void Main() {
  PrintInfo();

  if (FLAGS_batch_size <= 0)
    throw std::invalid_argument(
        "The batch_size option is less than or equal to 0.");
  if (FLAGS_iterations <= 0)
    throw std::invalid_argument(
        "The iterations option is less than or equal to 0.");
  if (FLAGS_skip_batch_num < 0)
    throw std::invalid_argument("The skip_batch_num option is less than 0.");
  struct stat sb;
  if (stat(FLAGS_infer_model.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
    throw std::invalid_argument(
        "The inference model directory does not exist.");
  }
  if (FLAGS_with_labels && FLAGS_use_fake_data)
    throw std::invalid_argument("Cannot use fake data for accuracy measuring.");

  paddle::PaddleTensor input_data = DefineInputData();
  paddle::PaddleTensor gt_label_tensor;
  paddle::PaddleTensor gt_bbox_tensor;
  paddle::PaddleTensor gt_difficult_tensor;

  // reader instance for real data
  std::unique_ptr<DataReader> reader;
  bool convert_to_rgb = false;

  // read the first batch
  if (FLAGS_use_fake_data) {
    CreateFakeData(input_data, gt_label_tensor, gt_bbox_tensor,
                   gt_difficult_tensor, reader);
  } else {
    InitializeReader(reader, convert_to_rgb);
    if (!ReadNextBatch(input_data, gt_label_tensor, gt_bbox_tensor,
                       gt_difficult_tensor, reader)) {
      throw std::invalid_argument(
          "Batch size cannot be bigger than dataset size.");
    }
  }

  // configure predictor
  contrib::AnalysisConfig config;
  PrepareConfig(config);

  auto predictor = CreatePaddlePredictor<contrib::AnalysisConfig,
                                         PaddleEngineKind::kAnalysis>(config);

  if (FLAGS_profile) {
    auto pf_state = paddle::platform::ProfilerState::kCPU;
    paddle::platform::EnableProfiler(pf_state);
  }

  // define output
  std::vector<PaddleTensor> output_slots;

  // run prediction
  Timer timer;
  Timer timer_total;
  Stats stats(FLAGS_batch_size, FLAGS_skip_batch_num);
  for (int i = 0; i < FLAGS_iterations + FLAGS_skip_batch_num; i++) {
    // start gathering performance data after `skip_batch_num` iterations
    if (i == FLAGS_skip_batch_num) {
      timer_total.tic();
      if (FLAGS_profile) {
        paddle::platform::ResetProfiler();
      }
    }

    // read next batch of data
    if (i > 0 && !FLAGS_use_fake_data) {
      if (!ReadNextBatch(input_data, gt_label_tensor, gt_bbox_tensor,
                         gt_difficult_tensor, reader)) {
        std::cout << "No more full batches. Stopping." << std::endl;
        break;
      }
    }

    // display images from batch if requested
    if (FLAGS_debug_display_images)
      DataReader::drawImages(static_cast<float *>(input_data.data.data()),
                             convert_to_rgb, FLAGS_batch_size, FLAGS_channels,
                             FLAGS_resize_size, FLAGS_resize_size);

    // define input
    std::vector<PaddleTensor> input;
    if (FLAGS_with_labels)
      input = {input_data, gt_bbox_tensor, gt_label_tensor,
               gt_difficult_tensor};
    else
      input = {input_data};

    // run inference
    timer.tic();
    if (!predictor->Run(input, &output_slots))
      throw std::runtime_error("Prediction failed.");
    double batch_time = timer.toc() / 1000;

    // gather statistics from the iteration
    stats.Gather(output_slots, batch_time, i);
  }

  if (FLAGS_profile) {
    paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                      "/tmp/profiler");
  }

  // postprocess statistics from all iterations
  double total_samples = FLAGS_iterations * FLAGS_batch_size;
  double total_time = timer_total.toc() / 1000;
  stats.Postprocess(total_time, total_samples);
}

} // namespace paddle

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    paddle::Main();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return 0;
}
