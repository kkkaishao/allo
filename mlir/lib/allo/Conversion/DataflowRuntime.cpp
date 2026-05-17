#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace {
struct Lane {
  std::mutex mutex;
  std::condition_variable notEmpty;
  std::condition_variable notFull;
  std::deque<uint64_t> values;
};

struct Stream {
  int64_t depth;
  std::vector<std::unique_ptr<Lane>> lanes;
};

static Stream *asStream(uint64_t handle) {
  auto *stream = reinterpret_cast<Stream *>(handle);
  assert(stream && "invalid stream handle");
  return stream;
}
} // namespace

extern "C" uint64_t _mlir_ciface_allo_sim_stream_create(int64_t lanes,
                                                        int64_t depth,
                                                        int64_t itemBytes) {
  assert(lanes > 0 && "stream must have at least one lane");
  assert(depth > 0 && "stream depth must be positive");
  assert(itemBytes > 0 && itemBytes <= 8 &&
         "dataflow simulator stores scalar payloads up to 64 bits");

  auto stream = std::make_unique<Stream>();
  stream->depth = depth;
  stream->lanes.reserve(lanes);
  for (int64_t i = 0; i < lanes; ++i)
    stream->lanes.push_back(std::make_unique<Lane>());
  return reinterpret_cast<uint64_t>(stream.release());
}

extern "C" void _mlir_ciface_allo_sim_stream_write(uint64_t handle,
                                                   int64_t lane,
                                                   uint64_t value) {
  Stream *stream = asStream(handle);
  assert(0 <= lane && lane < static_cast<int64_t>(stream->lanes.size()) &&
         "stream lane out of bounds");
  Lane &queue = *stream->lanes[lane];

  std::unique_lock<std::mutex> lock(queue.mutex);
  queue.notFull.wait(lock, [&] {
    return static_cast<int64_t>(queue.values.size()) < stream->depth;
  });
  queue.values.push_back(value);
  lock.unlock();
  queue.notEmpty.notify_one();
}

extern "C" uint64_t _mlir_ciface_allo_sim_stream_read(uint64_t handle,
                                                      int64_t lane) {
  Stream *stream = asStream(handle);
  assert(0 <= lane && lane < static_cast<int64_t>(stream->lanes.size()) &&
         "stream lane out of bounds");
  Lane &queue = *stream->lanes[lane];

  std::unique_lock<std::mutex> lock(queue.mutex);
  queue.notEmpty.wait(lock, [&] { return !queue.values.empty(); });
  uint64_t value = queue.values.front();
  queue.values.pop_front();
  lock.unlock();
  queue.notFull.notify_one();
  return value;
}

extern "C" void _mlir_ciface_allo_sim_stream_destroy(uint64_t handle) {
  delete asStream(handle);
}
