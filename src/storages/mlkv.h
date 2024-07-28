#ifndef DGL_STORAGES_MLKV_H_
#define DGL_STORAGES_MLKV_H_

#include <string>

#include "core/faster.h"
#include "device/null_disk.h"

namespace dgl {
namespace storages {

/// This benchmark stores 8-byte keys in key-value store.
class FASTER_Key {
 public:
  FASTER_Key(uint64_t key)
    : key_{ key } {
  }

  /// Methods and operators required by the (implicit) interface:
  inline static constexpr uint32_t size() {
    return static_cast<uint32_t>(sizeof(FASTER_Key));
  }
  inline FASTER::core::KeyHash GetHash() const {
    return FASTER::core::KeyHash{ FASTER::core::Utility::GetHashCode(key_) };
  }

  /// Comparison operators.
  inline bool operator==(const FASTER_Key& other) const {
    return key_ == other.key_;
  }
  inline bool operator!=(const FASTER_Key& other) const {
    return key_ != other.key_;
  }

 private:
  uint64_t key_;
};

class GenLock {
 public:
  GenLock()
    : control_{ 0 } {
  }
  GenLock(uint64_t control)
    : control_{ control } {
  }
  inline GenLock& operator=(const GenLock& other) {
    control_ = other.control_;
    return *this;
  }

  union {
      struct {
          int32_t staleness : 32;
          uint64_t gen_number : 30;
          uint64_t locked : 1;
          uint64_t replaced : 1;
      };
      uint64_t control_;
    };
};
static_assert(sizeof(GenLock) == 8, "sizeof(GenLock) != 8");

class AtomicGenLock {
 public:
  AtomicGenLock()
    : control_{ 0 } {
  }
  AtomicGenLock(uint64_t control)
    : control_{ control } {
  }

  inline GenLock load() const {
    return GenLock{ control_.load() };
  }
  inline void store(GenLock desired) {
    control_.store(desired.control_);
  }

  inline bool try_lock(bool& replaced, int32_t staleness_incr, int32_t staleness_bound) {
    replaced = false;
    GenLock expected{ control_.load() };
    expected.locked = 0;
    expected.replaced = 0;
    GenLock desired{ expected.control_ };
    desired.locked = 1;
    desired.staleness += staleness_incr;

    if(control_.compare_exchange_strong(expected.control_, desired.control_)) {
      return true;
    }
    if(expected.replaced) {
      replaced = true;
    }
    return false;
  }
  inline void unlock(bool replaced) {
    if(!replaced) {
      // Just turn off "locked" bit and increase gen number.
      uint64_t sub_delta = ((uint64_t)1 << 62) - ((uint64_t)1 << 32);
      control_.fetch_sub(sub_delta);
    } else {
      // Turn off "locked" bit, turn on "replaced" bit, and increase gen number
      uint64_t add_delta = ((uint64_t)1 << 63) - ((uint64_t)1 << 62) + ((uint64_t)1 << 32);
      control_.fetch_add(add_delta);
    }
  }

 private:
  std::atomic<uint64_t> control_;
};
static_assert(sizeof(AtomicGenLock) == 8, "sizeof(AtomicGenLock) != 8");

class FASTER_Value {
 public:
  FASTER_Value()
    : gen_lock_{ 0 }
    , size_{ 0 }
    , length_{ 0 } {
  }

  inline uint32_t size() const {
    return size_;
  }

  friend class UpsertContext;
  friend class ReadContext;
  friend class LookaheadContext;
  friend class MLKVReadContext;
  friend class MLKVUpsertContext;

 private:
  AtomicGenLock gen_lock_;
  uint32_t size_;
  uint32_t length_;

  inline const uint8_t* buffer() const {
    return reinterpret_cast<const uint8_t*>(this + 1);
  }
  inline uint8_t* buffer() {
    return reinterpret_cast<uint8_t*>(this + 1);
  }
};

class LookaheadContext : public FASTER::core::IAsyncContext {
  public:
   typedef FASTER_Key key_t;
   typedef FASTER_Value value_t;

   LookaheadContext(uint64_t key, uint32_t length)
     : key_{ key }
     , length_{ length } {
   }

   /// Copy (and deep-copy) constructor.
   LookaheadContext(const LookaheadContext& other)
     : key_{ other.key_ }
     , length_{ other.length_ } {
   }

   /// The implicit and explicit interfaces require a key() accessor.
   inline const key_t& key() const {
     return key_;
   }
   inline uint32_t value_size() const {
     return sizeof(value_t) + length_;
   }
   inline uint32_t value_size(const value_t& old_value) const {
     return sizeof(value_t) + old_value.length_;
   }
   inline void RmwInitial(value_t& value) {
     assert(false);
   }
   inline void RmwCopy(const value_t& old_value, value_t& value) {
     GenLock before, after;
     before = old_value.gen_lock_.load();
     after.staleness = before.staleness;

     value.gen_lock_.store(after);
     value.size_ = sizeof(value_t) + old_value.length_;
     value.length_ = old_value.length_;

     std::memcpy(value.buffer(), old_value.buffer(), old_value.length_);
   }
   inline bool RmwAtomic(value_t& value) {
     return true;
   }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
   FASTER::core::Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    uint32_t length_;
    key_t key_;
};

class ReadContext : public FASTER::core::IAsyncContext {
 public:
  typedef FASTER_Key key_t;
  typedef FASTER_Value value_t;

  ReadContext(uint64_t key, char* local_data_char)
    : key_{ key }
    , local_data_char_{ local_data_char } {
  }

  /// Copy (and deep-copy) constructor.
  ReadContext(const ReadContext& other)
    : key_{ other.key_ }
    , local_data_char_{ other.local_data_char_ } {
  }

  /// The implicit and explicit interfaces require a key() accessor.
  inline const key_t& key() const {
    return key_;
  }

  inline void Get(const value_t& value) {
    // TODO: make sure the correctness of disk-based operations
    std::memcpy(local_data_char_, value.buffer(), value.length_);
  }
  inline void GetAtomic(const value_t& value) {
    GenLock before, after;
    do {
      before = value.gen_lock_.load();
      std::memcpy(local_data_char_, value.buffer(), value.length_);
      after = value.gen_lock_.load();
    } while(before.gen_number != after.gen_number);
  }

 protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  FASTER::core::Status DeepCopy_Internal(FASTER::core::IAsyncContext*& context_copy) {
    return FASTER::core::IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

 private:
  key_t key_;
 public:
  char* local_data_char_;
};

class UpsertContext : public FASTER::core::IAsyncContext {
 public:
  typedef FASTER_Key key_t;
  typedef FASTER_Value value_t;

  UpsertContext(uint64_t key, char* local_data_char, uint32_t length)
    : key_{ key }
    , local_data_char_{ local_data_char }
    , length_{ length } {
  }

  /// Copy (and deep-copy) constructor.
  UpsertContext(const UpsertContext& other)
    : key_{ other.key_ }
    , local_data_char_{ other.local_data_char_ }
    , length_{ other.length_ } {
  }

  /// The implicit and explicit interfaces require a key() accessor.
  inline const key_t& key() const {
    return key_;
  }
  inline uint32_t value_size() const {
    return sizeof(value_t) + length_;
  }
  /// Non-atomic and atomic Put() methods.
  inline void Put(value_t& value) {
    value.gen_lock_.store(0);
    value.size_ = sizeof(FASTER_Value) + length_;
    value.length_ = length_;
    std::memcpy(value.buffer(), local_data_char_, length_);
  }
  inline bool PutAtomic(value_t& value) {
    bool replaced;
    while(!value.gen_lock_.try_lock(replaced, /*staleness_incr*/ 0, /*staleness_bound*/ INT32_MAX) && !replaced) {
      std::this_thread::yield();
    }
    if(replaced) {
      // Some other thread replaced this record.
      return false;
    }
    if(value.size_ < sizeof(value_t) + length_) {
      // Current value is too small for in-place update.
      value.gen_lock_.unlock(true);
      return false;
    }
    // In-place update overwrites length and buffer, but not size.
    value.length_ = length_;
    std::memcpy(value.buffer(), local_data_char_, length_);
    value.gen_lock_.unlock(false);
    return true;
  }

 protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  FASTER::core::Status DeepCopy_Internal(FASTER::core::IAsyncContext*& context_copy) {
    return FASTER::core::IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

 private:
  key_t key_;
  char* local_data_char_;
  uint32_t length_;
};

class MLKVReadContext : public FASTER::core::IAsyncContext {
 public:
  typedef FASTER_Key key_t;
  typedef FASTER_Value value_t;

  MLKVReadContext(uint64_t key, uint8_t* output, uint32_t length,
                  int32_t staleness_incr, int32_t staleness_bound)
    : key_{ key }
    , output_{ output }
    , length_{ length }
    , staleness_incr_{ staleness_incr }
    , staleness_bound_{ staleness_bound } {
  }

  /// Copy (and deep-copy) constructor.
  MLKVReadContext(const MLKVReadContext& other)
    : key_{ other.key_ }
    , output_{ other.output_ }
    , length_{ other.length_ }
    , staleness_incr_{ other.staleness_incr_ }
    , staleness_bound_{ other.staleness_bound_ } {
  }

  /// The implicit and explicit interfaces require a key() accessor.
  const key_t& key() const {
    return key_;
  }
  inline int32_t value_size() const {
    return sizeof(value_t) + length_;
  }
  inline uint32_t value_size(const value_t& old_value) const {
    return sizeof(value_t) + length_;
  }

  /// Initial, non-atomic, and atomic RMW methods.
  inline void RmwInitial(value_t& value) {
    // assert(false);
  }
  inline void RmwCopy(const value_t& old_value, value_t& value) {
    GenLock before, after;
    before = old_value.gen_lock_.load();
    after.staleness = before.staleness + staleness_incr_;

    value.gen_lock_.store(after);
    value.size_ = sizeof(value_t) + length_;
    value.length_ = length_;

    std::memcpy(value.buffer(), old_value.buffer(), old_value.length_);
    std::memcpy(output_, old_value.buffer(), old_value.length_);
  }
  inline bool RmwAtomic(value_t& value) {
    bool replaced;
    while(!value.gen_lock_.try_lock(replaced, staleness_incr_, staleness_bound_)
          && !replaced) {
      std::this_thread::yield();
    }
    if(replaced) {
      // Some other thread replaced this record.
      return false;
    }
    if(value.size_ < sizeof(value_t) + length_) {
      // Current value is too small for in-place update.
      value.gen_lock_.unlock(true);
      return false;
    }
    // In-place update overwrites length and buffer, but not size.
    value.length_ = length_;
    std::memcpy(output_, value.buffer(), value.length_);
    value.gen_lock_.unlock(false);
    return true;
  }

 protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  FASTER::core::Status DeepCopy_Internal(IAsyncContext*& context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

 private:
  key_t key_;
  uint8_t* output_;
  uint32_t length_;
  int32_t staleness_incr_;
  int32_t staleness_bound_;
};

class MLKVUpsertContext : public FASTER::core::IAsyncContext {
 public:
  typedef FASTER_Key key_t;
  typedef FASTER_Value value_t;

  MLKVUpsertContext(uint64_t key, uint8_t* input, uint32_t length,
                    int32_t staleness_incr, int32_t staleness_bound)
    : key_{ key }
    , input_{ input }
    , length_{ length }
    , staleness_incr_{ staleness_incr }
    , staleness_bound_{ staleness_bound } {
  }

  /// Copy (and deep-copy) constructor.
  MLKVUpsertContext(const MLKVUpsertContext& other)
    : key_{ other.key_ }
    , input_{ other.input_ }
    , length_{ other.length_ }
    , staleness_incr_{ other.staleness_incr_ }
    , staleness_bound_{ other.staleness_bound_ } {
  }

  /// The implicit and explicit interfaces require a key() accessor.
  const key_t& key() const {
    return key_;
  }
  inline int32_t value_size() const {
    return sizeof(value_t) + length_;
  }
  inline uint32_t value_size(const value_t& old_value) const {
    return sizeof(value_t) + length_;
  }

  /// Initial, non-atomic, and atomic RMW methods.
  inline void RmwInitial(value_t& value) {
    // assert(false);
  }
  inline void RmwCopy(const value_t& old_value, value_t& value) {
    GenLock before, after;
    before = old_value.gen_lock_.load();
    after.staleness = before.staleness + staleness_incr_;

    value.gen_lock_.store(after);
    value.size_ = sizeof(value_t) + length_;
    value.length_ = length_;

    std::memcpy(value.buffer(), input_, length_);
  }
  inline bool RmwAtomic(value_t& value) {
    bool replaced;
    while(!value.gen_lock_.try_lock(replaced, staleness_incr_, staleness_bound_)
          && !replaced) {
      std::this_thread::yield();
    }
    if(replaced) {
      // Some other thread replaced this record.
      return false;
    }
    if(value.size_ < sizeof(value_t) + length_) {
      // Current value is too small for in-place update.
      value.gen_lock_.unlock(true);
      return false;
    }
    // In-place update overwrites length and buffer, but not size.
    value.length_ = length_;
    std::memcpy(value.buffer(), input_, length_);
    value.gen_lock_.unlock(false);
    return true;
  }

 protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  FASTER::core::Status DeepCopy_Internal(IAsyncContext*& context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

 private:
  key_t key_;
  uint8_t* input_;
  uint32_t length_;
  int32_t staleness_incr_;
  int32_t staleness_bound_;
};

typedef FASTER::environment::QueueIoHandler handler_t;
typedef FASTER::device::FileSystemDisk<handler_t, 1073741824ull> disk_t;
using store_t = FASTER::core::FasterKv<FASTER_Key, FASTER_Value, disk_t>;

struct MLKVContext {
  std::vector<std::thread> async_threads_;
  store_t* store = nullptr;

  /*! \brief Get the MLKV context singleton */
  static MLKVContext* getInstance() {
    static MLKVContext ctx;
    return &ctx;
  }

};

}  // namespace storages
}  // namespace dgl

#endif
