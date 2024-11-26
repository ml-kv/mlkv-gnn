#if defined(__linux__)

#include <dgl/runtime/parallel_for.h>

#include "./mlkv.h"
#include "../c_api_common.h"

#include <chrono>

using namespace dgl::runtime;

namespace dgl {
namespace storages {

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStorageOpenMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  const int64_t table_size_bytes = args[0];
  const int64_t log_size_bytes = args[1];
  const std::string path = args[2];
  const double log_mutable_fraction = args[3];

  MLKVContext* ctx_ptr = MLKVContext::getInstance();
  ctx_ptr->store = new store_t( table_size_bytes / 64, log_size_bytes,
    path, log_mutable_fraction );
  LOG(INFO) << "MLKV with Path: " << path << " is created.";	
});

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStorageRecoverMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  const int64_t table_size_bytes = args[0];
  const int64_t log_size_bytes = args[1];
  const std::string path = args[2];
  const double log_mutable_fraction = args[3];
  static FASTER::core::Guid token = FASTER::core::Guid::Parse(args[4]);

  // Recovery.
  MLKVContext* ctx_ptr = MLKVContext::getInstance();
  if (ctx_ptr->store)
    delete ctx_ptr->store;
  store_t* store = new store_t( table_size_bytes / 64, log_size_bytes,
    path, log_mutable_fraction );
  ctx_ptr->store = store;

  uint32_t version;
  std::vector<FASTER::core::Guid> recovered_session_ids;
  FASTER::core::Status status = store->Recover(token, token, version, recovered_session_ids);
  if(status != FASTER::core::Status::Ok) {
    LOG(ERROR) << "Recovery failed with error " << static_cast<uint8_t>(status);
  }
  std::vector<uint64_t> serial_nums;
  for(const auto& recovered_session_id : recovered_session_ids) {
    serial_nums.push_back(store->ContinueSession(recovered_session_id));
    store->StopSession();
  }
  LOG(INFO) << "Recover from checkpoint token " << token << " successfully!";
});

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStorageCheckpointMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  MLKVContext* ctx_ptr = MLKVContext::getInstance();
  store_t* store = ctx_ptr->store;

  static FASTER::core::Guid token;
  static std::atomic<bool> index_checkpoint_completed;
  index_checkpoint_completed = false;
  static std::atomic<bool> hybrid_log_checkpoint_completed;
  hybrid_log_checkpoint_completed = false;

  auto index_persistence_callback = [](FASTER::core::Status result) {
    index_checkpoint_completed = true;
  };
  auto hybrid_log_persistence_callback = [](FASTER::core::Status result, uint64_t persistent_serial_num) {
    hybrid_log_checkpoint_completed = true;
  };

  store->StartSession();
  bool result = store->Checkpoint(index_persistence_callback, hybrid_log_persistence_callback, token);

  if (!result) {
    LOG(ERROR) << "Checkpoint failed.";
  }
  while(!index_checkpoint_completed) {
    store->CompletePending(false);
  }
  while(!hybrid_log_checkpoint_completed) {
    store->CompletePending(false);
  }
  store->CompletePending(true);
  store->StopSession();
  LOG(INFO) << "Checkpoint successfully! Please keep your checkpoint id: " << token;
});

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStoragePullDataFromMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  // Input
  NDArray id_tensor = args[0];
  const int64_t feat_dim = args[1];
  DGLDataType feat_dtype = args[2];
  
  // Data
  int64_t id_size = id_tensor.GetSize() / sizeof(int64_t);
  int64_t* id_tensor_data = static_cast<int64_t*>(id_tensor->data);

  // Get row size (in bytes)
  std::vector<int64_t> res_data_shape;
  res_data_shape.push_back(id_size);
  res_data_shape.push_back(feat_dim);
  int row_size = feat_dim * feat_dtype.bits / 8;

  NDArray res_tensor = NDArray::Empty(res_data_shape,
                                      feat_dtype,
                                      DGLContext{kDGLCPU, 0});
  char* return_data = static_cast<char*>(res_tensor->data);
  // Copy local data
  MLKVContext* ctx_ptr = MLKVContext::getInstance();
  store_t* store = ctx_ptr->store;
  parallel_for(0, id_size, [&](size_t b, size_t e) {
    store->StartSession();
    for (auto i = b; i < e; ++i) {
      CHECK_GE(id_size * row_size, i * row_size + row_size);
      CHECK_GE(id_tensor_data[i], 0);
      //auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
      //  FASTER::core::CallbackContext<ReadContext> context{ ctxt };
      //};
      //ReadContext context{ static_cast<uint64_t>(id_tensor_data[i]), return_data + i * row_size };
      //FASTER::core::Status result = store->Read(context, callback, 1);

      auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
        FASTER::core::CallbackContext<MLKVReadContext> context{ ctxt };
      };
      MLKVReadContext context{ static_cast<uint64_t>(id_tensor_data[i]),
        (uint8_t *)(return_data + i * row_size), row_size, 1, 128 };
      FASTER::core::Status result = store->Rmw(context, callback, 1);

      assert(FASTER::core::Status::Ok == result);
    }
    store->StopSession();
  });
  *rv = res_tensor;
});

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStoragePullAndLookaheadDataFromMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  // Input
  const int64_t nthread = args[0];
  NDArray id_tensor_pull = args[1];
  NDArray id_tensor_lookahead = args[2];
  const int64_t feat_dim = args[3];
  DGLDataType feat_dtype = args[4];

  // Get row size (in bytes)
  int row_size = feat_dim * feat_dtype.bits / 8;
  MLKVContext* ctx_ptr = MLKVContext::getInstance();
  store_t* store = ctx_ptr->store;

  // Lookahead
  int64_t id_lookahead_size = id_tensor_lookahead.GetSize() / sizeof(int64_t);
  int64_t* id_tensor_lookahead_data = static_cast<int64_t*>(id_tensor_lookahead->data);

  std::vector<std::thread> threads;
  for (int64_t tid = 0; tid < nthread; tid++) {
    threads.push_back(std::thread([&, tid] {
      int64_t b = tid * id_lookahead_size / nthread;
      int64_t e = (tid + 1) * id_lookahead_size / nthread;
      store->StartSession();
      for (auto i = b; i < e; i += 1) {
        CHECK_GE(id_tensor_lookahead_data[i], 0);
        auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
          FASTER::core::CallbackContext<LookaheadContext> context{ ctxt };
        };
        LookaheadContext context{ static_cast<uint64_t>(id_tensor_lookahead_data[i]), row_size };
        FASTER::core::Status result = store->Rmw(context, callback, 1);
        assert(FASTER::core::Status::Ok == result);
      }
      store->StopSession();
    }));
  }
  // Pull
  int64_t id_pull_size = id_tensor_pull.GetSize() / sizeof(int64_t);
  int64_t* id_tensor_pull_data = static_cast<int64_t*>(id_tensor_pull->data);

  std::vector<int64_t> res_data_shape;
  res_data_shape.push_back(id_pull_size);
  res_data_shape.push_back(feat_dim);

  NDArray res_tensor = NDArray::Empty(res_data_shape,
                                      feat_dtype,
                                      DGLContext{kDGLCPU, 0});
  char* return_data = static_cast<char*>(res_tensor->data);
  // Copy local data
  parallel_for(0, id_pull_size, [&](size_t b, size_t e) {
    store->StartSession();
    for (auto i = b; i < e; ++i) {
      CHECK_GE(id_pull_size * row_size, i * row_size + row_size);
      CHECK_GE(id_tensor_pull_data[i], 0);
      auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
        FASTER::core::CallbackContext<ReadContext> context{ ctxt };
      };
      ReadContext context{ static_cast<uint64_t>(id_tensor_pull_data[i]), return_data + i * row_size };
      FASTER::core::Status result = store->Read(context, callback, 1);
      assert(FASTER::core::Status::Ok == result);
    }
    store->StopSession();
  });
  for (int64_t i = 0; i < nthread; ++i) {
    threads[i].join();
  }
  *rv = res_tensor;
});

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStorageAsyncParallelLookaheadDataFromMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  // Input
  const int64_t nthread = args[0];
  NDArray id_tensor = args[1];
  const int64_t feat_dim = args[2];
  DGLDataType feat_dtype = args[3];

  // Data
  int64_t id_size = id_tensor.GetSize() / sizeof(int64_t);
  int64_t* id_tensor_data = static_cast<int64_t*>(id_tensor->data);

  // Get row size (in bytes)
  int row_size = feat_dim * feat_dtype.bits / 8;
  MLKVContext* ctx_ptr = MLKVContext::getInstance();

  for (int64_t tid = 0; tid < nthread; tid++) {
    // Get local id
    int64_t b = tid * id_size / nthread;
    int64_t e = (tid + 1) * id_size / nthread;
    std::vector<int64_t>* _ids_ptr = new std::vector<int64_t>;
    for (int64_t i = b; i < e; ++i) {
      int64_t _id = id_tensor_data[i];
      CHECK_GE(_id, 0);
      _ids_ptr->push_back(_id);
    }
    ctx_ptr->async_threads_.push_back(std::thread([row_size, _ids_ptr] {
      MLKVContext* ctx_ptr = MLKVContext::getInstance();
      store_t* store = ctx_ptr->store;
      store->StartSession();

      for (int64_t i = 0; i < _ids_ptr->size(); i++) {
        CHECK_GE(i, 0);
        auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
          FASTER::core::CallbackContext<LookaheadContext> context{ ctxt };
        };
        LookaheadContext context{ static_cast<uint64_t>(_ids_ptr->at(i)), row_size };
        FASTER::core::Status result = store->Rmw(context, callback, 1);
        assert(FASTER::core::Status::Ok == result);
        if (i % 64 == 0)
          store->CompletePending(true);
      }

      store->StopSession();
      delete _ids_ptr;
    }));
  }
});

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStorageAsyncPushDataToMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  // Input
  NDArray id_tensor = args[0];
  NDArray data_tensor = args[1];

  // Get row size (in bytes)
  std::vector<int64_t> data_tensor_shape;
  int row_size = 1;
  for (int i = 0; i < data_tensor->ndim; ++i) {
    data_tensor_shape.push_back(data_tensor->shape[i]);
    if (i != 0) {
      row_size *= data_tensor->shape[i];
    }
  }
  row_size *= (data_tensor->dtype.bits / 8);
  CHECK_GT(data_tensor_shape.size(), 0);

  // Data
  int64_t id_size = id_tensor.GetSize() / sizeof(int64_t);
  int64_t* id_tensor_data = static_cast<int64_t*>(id_tensor->data);
  size_t data_tensor_size = data_tensor.GetSize();
  CHECK_EQ(row_size * data_tensor_shape[0], data_tensor_size);
  char* data_tensor_char = new char[data_tensor_size];
  std::memcpy(data_tensor_char, static_cast<char*>(data_tensor->data), data_tensor_size);

  std::vector<int64_t>* _ids_ptr = new std::vector<int64_t>;
  for (int64_t i = 0; i < id_size; ++i) {
    int64_t _id = id_tensor_data[i];
    CHECK_GE(_id, 0);
    _ids_ptr->push_back(_id);
  }

  MLKVContext* ctx_ptr = MLKVContext::getInstance();
  ctx_ptr->async_threads_.push_back(std::thread([row_size, _ids_ptr, data_tensor_char] {
    MLKVContext* ctx_ptr = MLKVContext::getInstance();
    store_t* store = ctx_ptr->store;
    store->StartSession();

    for (int64_t i = 0; i < _ids_ptr->size(); i++) {
      CHECK_GE(i, 0);
      //auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
      //  FASTER::core::CallbackContext<LookaheadContext> context{ ctxt };
      //};
      //UpsertContext context{ static_cast<uint64_t>(_ids_ptr->at(i)),
      //  data_tensor_char + i * row_size, row_size };
      //FASTER::core::Status result = store->Upsert(context, callback, 1);

      auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
        FASTER::core::CallbackContext<MLKVUpsertContext> context{ ctxt };
      };
      MLKVUpsertContext context{ static_cast<uint64_t>(_ids_ptr->at(i)),
        (uint8_t *)(data_tensor_char + i * row_size), row_size, -1, 128 };
      FASTER::core::Status result = store->Rmw(context, callback, 1);

      assert(FASTER::core::Status::Ok == result);
      if (i % 64 == 0)
        store->CompletePending(true);
    }

    store->StopSession();
    delete _ids_ptr;
    delete data_tensor_char;
  }));
});

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStorageWaitAsyncThreadsInMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  MLKVContext* ctx_ptr = MLKVContext::getInstance();
  for (auto& thread : ctx_ptr->async_threads_) {
    thread.join();
  }
  ctx_ptr->async_threads_.clear();
});

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStoragePushDataToMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  // Input
  NDArray id_tensor = args[0];
  NDArray data_tensor = args[1];

  // Get row size (in bytes)
  std::vector<int64_t> data_tensor_shape;
  int row_size = 1;
  for (int i = 0; i < data_tensor->ndim; ++i) {
    data_tensor_shape.push_back(data_tensor->shape[i]);
    if (i != 0) {
      row_size *= data_tensor->shape[i];
    }
  }
  row_size *= (data_tensor->dtype.bits / 8);
  CHECK_GT(data_tensor_shape.size(), 0);

  // Data
  int64_t id_size = id_tensor.GetSize() / sizeof(int64_t);
  int64_t* id_tensor_data = static_cast<int64_t*>(id_tensor->data);
  size_t data_tensor_size = data_tensor.GetSize();
  char* data_tensor_char = static_cast<char*>(data_tensor->data);
  CHECK_EQ(row_size * data_tensor_shape[0], data_tensor_size);

  // Copy local data
  MLKVContext* ctx_ptr = MLKVContext::getInstance();
  store_t* store = ctx_ptr->store;
  parallel_for(0, id_size, [&](size_t b, size_t e) {
    store->StartSession();
    for (auto i = b; i < e; i += 1) {
      CHECK_GE(id_tensor_data[i], 0);
      auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
        FASTER::core::CallbackContext<UpsertContext> context{ ctxt };
      };
      UpsertContext context{ static_cast<uint64_t>(id_tensor_data[i]),
        data_tensor_char + i * row_size, row_size };
      FASTER::core::Status result = store->Upsert(context, callback, 1);
      assert(FASTER::core::Status::Ok == result);
    }
    store->StopSession();
  });
});

DGL_REGISTER_GLOBAL("storages.mlkv._CAPI_DGLStorageParallelPushDataToMLKV")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  // Input
  const int64_t nthread = args[0];
  NDArray id_tensor = args[1];
  NDArray data_tensor = args[2];
  
  // Get row size (in bytes)
  std::vector<int64_t> data_tensor_shape;
  int row_size = 1;
  for (int i = 0; i < data_tensor->ndim; ++i) {
    data_tensor_shape.push_back(data_tensor->shape[i]);
    if (i != 0) {
      row_size *= data_tensor->shape[i];
    }
  }
  row_size *= (data_tensor->dtype.bits / 8);
  CHECK_GT(data_tensor_shape.size(), 0);

  // Data
  int64_t id_size = id_tensor.GetSize() / sizeof(int64_t);
  int64_t* id_tensor_data = static_cast<int64_t*>(id_tensor->data);
  size_t data_tensor_size = data_tensor.GetSize();
  char* data_tensor_char = static_cast<char*>(data_tensor->data);
  CHECK_EQ(row_size * data_tensor_shape[0], data_tensor_size);

  // Copy local data
  MLKVContext* ctx_ptr = MLKVContext::getInstance();
  store_t* store = ctx_ptr->store;
  std::vector<std::thread> threads;
  for (int64_t tid = 0; tid < nthread; tid++) {
    threads.push_back(std::thread([&, tid] {
      int64_t b = tid * id_size / nthread;
      int64_t e = (tid + 1) * id_size / nthread;
      e = e < id_size ? e : id_size;

      store->StartSession();
      for (int64_t i = b; i < e; ++i) {
        CHECK_GE(id_size*row_size, i * row_size + row_size);
        CHECK_GE(id_tensor_data[i], 0);
        auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
          FASTER::core::CallbackContext<UpsertContext> context{ ctxt };
        };
        UpsertContext context{ static_cast<uint64_t>(id_tensor_data[i]),
          data_tensor_char + i * row_size, row_size };
        FASTER::core::Status result = store->Upsert(context, callback, 1);
        assert(FASTER::core::Status::Ok == result);
      }
      store->CompletePending(true);
      store->StopSession();
    }));
  }
  for (int64_t i = 0; i < nthread; ++i) {
    threads[i].join();
  }
});

}  // namespace storages
}  // namespace dgl

#endif
