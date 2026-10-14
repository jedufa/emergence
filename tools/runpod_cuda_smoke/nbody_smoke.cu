#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Particle {
  float x;
  float y;
  float z;
  float vx;
  float vy;
  float vz;
};

__global__ void nbody_step(const Particle* input, Particle* output, int count,
                           float dt, float softening) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const Particle particle = input[index];
  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  for (int other_index = 0; other_index < count; ++other_index) {
    const Particle other = input[other_index];
    const float dx = other.x - particle.x;
    const float dy = other.y - particle.y;
    const float dz = other.z - particle.z;
    const float inverse_distance =
        rsqrtf(dx * dx + dy * dy + dz * dz + softening);
    const float inverse_distance_cubed =
        inverse_distance * inverse_distance * inverse_distance;
    ax += dx * inverse_distance_cubed;
    ay += dy * inverse_distance_cubed;
    az += dz * inverse_distance_cubed;
  }

  Particle next = particle;
  const float scale = dt / static_cast<float>(count);
  next.vx += ax * scale;
  next.vy += ay * scale;
  next.vz += az * scale;
  next.x += next.vx * dt;
  next.y += next.vy * dt;
  next.z += next.vz * dt;
  output[index] = next;
}

void cuda_check(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << expression << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

#define CUDA_CHECK(expression) cuda_check((expression), #expression)

std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(character);
  }
  return escaped;
}

void write_failure(const std::string& path, const std::string& message) {
  std::ofstream output(path);
  output << "{\n"
         << "  \"schema_version\": \"eiel_cuda_nbody_smoke_v1\",\n"
         << "  \"status\": \"failed\",\n"
         << "  \"error\": \"" << json_escape(message) << "\"\n"
         << "}\n";
}

struct Options {
  int particles = 2048;
  int seconds = 60;
  std::string output = "result.json";
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next_value = [&](const char* name) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++index];
    };
    if (argument == "--particles") {
      options.particles = std::stoi(next_value("--particles"));
    } else if (argument == "--seconds") {
      options.seconds = std::stoi(next_value("--seconds"));
    } else if (argument == "--output") {
      options.output = next_value("--output");
    } else if (argument == "--help") {
      std::cout << "usage: nbody_smoke [--particles N] [--seconds N] "
                   "[--output PATH]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.particles < 256 || options.particles > 16384) {
    throw std::runtime_error("--particles must be between 256 and 16384");
  }
  if (options.seconds < 1 || options.seconds > 600) {
    throw std::runtime_error("--seconds must be between 1 and 600");
  }
  return options;
}

float random_unit(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return static_cast<float>(state & 0x00ffffffu) /
         static_cast<float>(0x01000000u);
}

int run(const Options& options) {
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::vector<Particle> host(static_cast<std::size_t>(options.particles));
  std::uint32_t random_state = 0x5eed1234u;
  for (Particle& particle : host) {
    particle.x = random_unit(random_state) * 2.0f - 1.0f;
    particle.y = random_unit(random_state) * 2.0f - 1.0f;
    particle.z = random_unit(random_state) * 2.0f - 1.0f;
    particle.vx = (random_unit(random_state) - 0.5f) * 0.02f;
    particle.vy = (random_unit(random_state) - 0.5f) * 0.02f;
    particle.vz = (random_unit(random_state) - 0.5f) * 0.02f;
  }

  Particle* device_a = nullptr;
  Particle* device_b = nullptr;
  const std::size_t bytes = host.size() * sizeof(Particle);
  CUDA_CHECK(cudaMalloc(&device_a, bytes));
  CUDA_CHECK(cudaMalloc(&device_b, bytes));
  CUDA_CHECK(cudaMemcpy(device_a, host.data(), bytes, cudaMemcpyHostToDevice));

  Particle* current = device_a;
  Particle* next = device_b;
  std::uint64_t steps = 0;
  const auto started = std::chrono::steady_clock::now();
  try {
    while (true) {
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      if (elapsed >= static_cast<double>(options.seconds)) {
        break;
      }
      const int block_size = 256;
      const int block_count =
          (options.particles + block_size - 1) / block_size;
      nbody_step<<<block_count, block_size>>>(current, next, options.particles,
                                              0.002f, 0.0001f);
      CUDA_CHECK(cudaPeekAtLastError());
      std::swap(current, next);
      ++steps;
      if ((steps % 8u) == 0u) {
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), current, bytes, cudaMemcpyDeviceToHost));
  } catch (...) {
    cudaFree(device_a);
    cudaFree(device_b);
    throw;
  }
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));

  const double elapsed = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - started)
                              .count();
  double checksum = 0.0;
  double max_abs_position = 0.0;
  for (const Particle& particle : host) {
    checksum += static_cast<double>(particle.x) * 0.5;
    checksum += static_cast<double>(particle.y) * 0.3;
    checksum += static_cast<double>(particle.z) * 0.2;
    max_abs_position = std::max(
        max_abs_position,
        static_cast<double>(std::max({std::fabs(particle.x),
                                      std::fabs(particle.y),
                                      std::fabs(particle.z)})));
  }

  int runtime_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  std::ofstream output(options.output);
  if (!output) {
    throw std::runtime_error("cannot open result output: " + options.output);
  }
  output << std::setprecision(17);
  output << "{\n"
         << "  \"schema_version\": \"eiel_cuda_nbody_smoke_v1\",\n"
         << "  \"status\": \"succeeded\",\n"
         << "  \"gpu_name\": \"" << json_escape(properties.name) << "\",\n"
         << "  \"compute_capability\": \"" << properties.major << "."
         << properties.minor << "\",\n"
         << "  \"cuda_runtime_version\": " << runtime_version << ",\n"
         << "  \"particles\": " << options.particles << ",\n"
         << "  \"seconds_requested\": " << options.seconds << ",\n"
         << "  \"seconds_elapsed\": " << elapsed << ",\n"
         << "  \"steps\": " << steps << ",\n"
         << "  \"steps_per_second\": "
         << (static_cast<double>(steps) / elapsed) << ",\n"
         << "  \"checksum\": " << checksum << ",\n"
         << "  \"max_abs_position\": " << max_abs_position << ",\n"
         << "  \"cuda_errors_checked\": true\n"
         << "}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string output_path = "result.json";
  try {
    const Options options = parse_options(argc, argv);
    output_path = options.output;
    return run(options);
  } catch (const std::exception& error) {
    std::cerr << "CUDA smoke test failed: " << error.what() << '\n';
    write_failure(output_path, error.what());
    return 1;
  }
}
