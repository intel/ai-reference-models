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

#include "stats.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iostream>
#include <numeric>
#include <sstream>

namespace paddle {

namespace {

template <typename T> void SkipFirstNData(std::vector<T> &v, int n) {
  std::vector<T>(v.begin() + n, v.end()).swap(v);
}

template <typename T> T FindAverage(const std::vector<T> &v) {
  if (v.size() == 0)
    throw std::invalid_argument("FindAverage: vector is empty.");
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template <typename T> T FindPercentile(std::vector<T> v, int p) {
  if (v.size() == 0)
    throw std::invalid_argument("FindPercentile: vector is empty.");
  std::sort(v.begin(), v.end());
  if (p == 100)
    return v.back();
  int i = v.size() * p / 100;
  return v[i];
}

template <typename T> T FindStandardDev(std::vector<T> v) {
  if (v.size() == 0)
    throw std::invalid_argument("FindStandardDev: vector is empty.");
  T mean = FindAverage(v);
  T var = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    var += (v[i] - mean) * (v[i] - mean);
  }
  var /= v.size();
  T std = sqrt(var);
  return std;
}

} // namespace

// gather statistics from the output
void Stats::Gather(const std::vector<PaddleTensor> &output_slots,
                   double batch_time, int iter) {
  latencies.push_back(batch_time);
  double fps = batch_size / batch_time;
  fpses.push_back(fps);

  std::stringstream ss;
  ss << "Iteration: " << iter + 1;
  if (iter < skip_batch_num)
    ss << " (warm-up)";
  // use standard float formatting
  ss << std::fixed << std::setw(11) << std::setprecision(6);
  ss << ", latency: " << batch_time << " s, fps: " << fps;

  // first output: avg_cost
  if (output_slots.size() == 0)
    throw std::invalid_argument("Gather: output_slots vector is empty.");

  // second output: total distance
  if (output_slots.size() >= 2UL) {
    if (output_slots[1].lod.size() > 0)
      throw std::invalid_argument(
          "Gather: total distance output has nonempty LoD.");
    if (output_slots[1].dtype != paddle::PaddleDType::FLOAT32)
      throw std::invalid_argument(
          "Gather: total distance output is of a wrong type.");
    double total_distance = *static_cast<float *>(output_slots[1].data.data());
    total_distance /= batch_size;
    total_distances.push_back(total_distance);
    // ss << ", total distance: " << total_distance;
    ss << ", avg distance: " << FindAverage(total_distances);
  }

  // third output: instance error count
  if (output_slots.size() >= 3UL) {
    if (output_slots[2].lod.size() > 0)
      throw std::invalid_argument(
          "Gather: instance error count output has nonempty LoD.");
    if (output_slots[2].dtype != paddle::PaddleDType::INT64)
      throw std::invalid_argument(
          "Gather: instance error count output is of a wrong type.");
    double instance_err_count =
        *static_cast<int64_t *>(output_slots[2].data.data());
    instance_err_count /= batch_size;
    instance_errors.push_back(instance_err_count);
    // ss << ", instance error count: " << instance_err_count;
    ss << ", avg instance error: " << FindAverage(instance_errors);
  }

  // fourth output: sequence number
  if (output_slots.size() >= 4UL) {
    if (output_slots[3].lod.size() > 0)
      throw std::invalid_argument(
          "Gather: sequence number output has nonempty LoD.");
    if (output_slots[3].dtype != paddle::PaddleDType::INT64)
      throw std::invalid_argument(
          "Gather: sequence number output is of a wrong type.");
    int64_t *seq_num = static_cast<int64_t *>(output_slots[3].data.data());
    // ss << ", sequence number: " << *seq_num;
  }

  // fifth output: instance error
  if (output_slots.size() >= 5UL) {
    if (output_slots[4].lod.size() > 0)
      throw std::invalid_argument(
          "Gather: instance error output has nonempty LoD.");
    if (output_slots[4].dtype != paddle::PaddleDType::INT64)
      throw std::invalid_argument(
          "Gather: instance error output is of a wrong type.");
    int64_t *instance_err = static_cast<int64_t *>(output_slots[3].data.data());
    // ss << ", instance error: " << *instance_err;
  }

  std::cout << ss.str() << std::endl;
}

// postprocess statistics from the whole test
void Stats::Postprocess(double total_time_sec, int total_samples) {
  SkipFirstNData(latencies, skip_batch_num);
  double lat_avg = FindAverage(latencies);
  double lat_pc99 = FindPercentile(latencies, 99);
  double lat_std = FindStandardDev(latencies);

  SkipFirstNData(fpses, skip_batch_num);
  double fps_avg = FindAverage(fpses);
  double fps_pc01 = FindPercentile(fpses, 1);
  double fps_std = FindStandardDev(fpses);

  float examples_per_sec = total_samples / total_time_sec;
  std::stringstream ss;
  // use standard float formatting
  ss << std::fixed << std::setw(11) << std::setprecision(6);
  ss << "\n";
  ss << "Avg fps: " << fps_avg << ", std fps: " << fps_std
     << ", fps for 99pc latency: " << fps_pc01 << std::endl;
  ss << "Avg latency: " << lat_avg << ", std latency: " << lat_std
     << ", 99pc latency: " << lat_pc99 << std::endl;
  ss << "Total examples: " << total_samples
     << ", total time: " << total_time_sec
     << ", total examples/sec: " << examples_per_sec << std::endl;

  if (total_distances.size() > 0) {
    SkipFirstNData(total_distances, skip_batch_num);
    double avg_distance = FindAverage(total_distances);
    ss << "Avg distance: " << avg_distance << std::endl;
  }

  if (instance_errors.size() > 0) {
    SkipFirstNData(instance_errors, skip_batch_num);
    double avg_instance_error = FindAverage(instance_errors);
    ss << "Avg instance error: " << avg_instance_error << std::endl;
  }
  std::cout << ss.str() << std::endl;
}

} // namespace paddle
