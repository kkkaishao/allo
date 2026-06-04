#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace {
struct Lane {
  std::mutex mutex;
  std::condition_variable notEmpty;
  std::condition_variable notFull;
  std::deque<std::vector<uint8_t>> values;
};

struct Stream {
  int64_t depth;
  int64_t itemBytes;
  std::vector<std::unique_ptr<Lane>> lanes;
};

static Stream *asStream(uint64_t handle) {
  auto *stream = reinterpret_cast<Stream *>(handle);
  assert(stream && "invalid stream handle");
  return stream;
}

static Lane &getLane(Stream *stream, int64_t lane) {
  assert(0 <= lane && lane < static_cast<int64_t>(stream->lanes.size()) &&
         "stream lane out of bounds");
  return *stream->lanes[lane];
}

static void writeBytes(Stream *stream, int64_t lane, const void *data) {
  assert(data && "invalid stream write payload");
  auto *bytes = static_cast<const uint8_t *>(data);
  Lane &queue = getLane(stream, lane);

  std::unique_lock<std::mutex> lock(queue.mutex);
  queue.notFull.wait(lock, [&] {
    return static_cast<int64_t>(queue.values.size()) < stream->depth;
  });
  queue.values.emplace_back(bytes, bytes + stream->itemBytes);
  lock.unlock();
  queue.notEmpty.notify_one();
}

static void readBytes(Stream *stream, int64_t lane, void *data) {
  assert(data && "invalid stream read payload");
  Lane &queue = getLane(stream, lane);

  std::unique_lock<std::mutex> lock(queue.mutex);
  queue.notEmpty.wait(lock, [&] { return !queue.values.empty(); });
  std::vector<uint8_t> value = std::move(queue.values.front());
  queue.values.pop_front();
  lock.unlock();
  queue.notFull.notify_one();

  assert(static_cast<int64_t>(value.size()) == stream->itemBytes &&
         "stream payload size mismatch");
  std::memcpy(data, value.data(), value.size());
}
} // namespace

extern "C" uint64_t allo_sim_stream_create(int64_t lanes, int64_t depth,
                                           int64_t itemBytes) {
  assert(lanes > 0 && "stream must have at least one lane");
  assert(depth > 0 && "stream depth must be positive");
  assert(itemBytes > 0 && "stream payload size must be positive");

  auto stream = std::make_unique<Stream>();
  stream->depth = depth;
  stream->itemBytes = itemBytes;
  stream->lanes.reserve(lanes);
  for (int64_t i = 0; i < lanes; ++i)
    stream->lanes.push_back(std::make_unique<Lane>());
  return reinterpret_cast<uint64_t>(stream.release());
}

extern "C" void allo_sim_stream_write(uint64_t handle, int64_t lane,
                                      uint64_t value) {
  Stream *stream = asStream(handle);
  assert(stream->itemBytes <= static_cast<int64_t>(sizeof(value)) &&
         "scalar stream payload is too wide");
  writeBytes(stream, lane, &value);
}

extern "C" uint64_t allo_sim_stream_read(uint64_t handle, int64_t lane) {
  Stream *stream = asStream(handle);
  assert(stream->itemBytes <= static_cast<int64_t>(sizeof(uint64_t)) &&
         "scalar stream payload is too wide");
  uint64_t value = 0;
  readBytes(stream, lane, &value);
  return value;
}

extern "C" void allo_sim_stream_write_mem(uint64_t handle, int64_t lane,
                                          uint64_t ptr) {
  writeBytes(asStream(handle), lane, reinterpret_cast<const void *>(ptr));
}

extern "C" void allo_sim_stream_read_mem(uint64_t handle, int64_t lane,
                                         uint64_t ptr) {
  readBytes(asStream(handle), lane, reinterpret_cast<void *>(ptr));
}

extern "C" void allo_sim_stream_destroy(uint64_t handle) {
  delete asStream(handle);
}
