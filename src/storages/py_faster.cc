#if defined(__linux__)

#include <dgl/runtime/parallel_for.h>

#include "./py_faster.h"
#include "../c_api_common.h"

#include <chrono>

using namespace dgl::runtime;

namespace dgl {
namespace storages {

DGL_REGISTER_GLOBAL("storages.py_faster._CAPI_DGLStorageOpenFASTER")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  const int64_t table_size_bytes = args[0];
  const int64_t log_size_bytes = args[1];
  const std::string path = args[2];
  const double log_mutable_fraction = args[3];

  FasterContext* ctx_ptr = FasterContext::getInstance();
  ctx_ptr->store = new store_t( table_size_bytes / 64, log_size_bytes,
    path, log_mutable_fraction );
  LOG(INFO) << "FASTER with Path: " << path << " is created.";	
});

DGL_REGISTER_GLOBAL("storages.py_faster._CAPI_DGLStorageRecoverFASTER")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  const int64_t table_size_bytes = args[0];
  const int64_t log_size_bytes = args[1];
  const std::string path = args[2];
  const double log_mutable_fraction = args[3];
  static FASTER::core::Guid token = FASTER::core::Guid::Parse(args[4]);

  // Recovery.
  FasterContext* ctx_ptr = FasterContext::getInstance();
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

DGL_REGISTER_GLOBAL("storages.py_faster._CAPI_DGLStorageCheckpointFASTER")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  FasterContext* ctx_ptr = FasterContext::getInstance();
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

DGL_REGISTER_GLOBAL("storages.py_faster._CAPI_DGLStoragePullDataFromFASTER")
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

  // Get local id
  std::vector<int64_t> _ids;
  std::vector<int64_t> _ids_orginal;
  for (int64_t i = 0; i < id_size; ++i) {
    int64_t _id = id_tensor_data[i];
    CHECK_GE(_id, 0);
    _ids.push_back(_id);
    _ids_orginal.push_back(i);
  }

  NDArray res_tensor = NDArray::Empty(res_data_shape,
                                      feat_dtype,
                                      DGLContext{kDGLCPU, 0});
  char* return_data = static_cast<char*>(res_tensor->data);
  // Copy local data
  FasterContext* ctx_ptr = FasterContext::getInstance();
  store_t* store = ctx_ptr->store;
  parallel_for(0, _ids.size(), [&](size_t b, size_t e) {
    store->StartSession();
    for (auto i = b; i < e; ++i) {
      CHECK_GE(id_size * row_size, _ids_orginal[i] * row_size + row_size);
      CHECK_GE(_ids[i], 0);
      auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
        FASTER::core::CallbackContext<ReadContext> context{ ctxt };
      };
      ReadContext context{ static_cast<uint64_t>(_ids[i]), return_data + _ids_orginal[i] * row_size };
      FASTER::core::Status result = store->Read(context, callback, 1);
      assert(FASTER::core::Status::Ok == result);
    }
    store->StopSession();
  });
  *rv = res_tensor;
});

DGL_REGISTER_GLOBAL("storages.py_faster._CAPI_DGLStoragePushDataToFASTER")
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

  // Get local id
  std::vector<int64_t> _ids;
  std::vector<int64_t> _ids_orginal;
  for (int64_t i = 0; i < id_size; ++i) {
    int64_t _id = id_tensor_data[i];
    CHECK_GE(_id, 0);
    _ids.push_back(_id);
    _ids_orginal.push_back(i);
  }
  // Copy local data
  FasterContext* ctx_ptr = FasterContext::getInstance();
  store_t* store = ctx_ptr->store;
  parallel_for(0, _ids.size(), [&](size_t b, size_t e) {
    store->StartSession();
    for (auto i = b; i < e; i += 1) {
      CHECK_GE(_ids[i], 0);
      auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
        FASTER::core::CallbackContext<UpsertContext> context{ ctxt };
      };
      UpsertContext context{ static_cast<uint64_t>(_ids[i]),
        data_tensor_char + _ids_orginal[i] * row_size, row_size };
      FASTER::core::Status result = store->Upsert(context, callback, 1);
      assert(FASTER::core::Status::Ok == result);
    }
    store->CompletePending(true);
    store->StopSession();
  });
});

DGL_REGISTER_GLOBAL("storages.py_faster._CAPI_DGLStorageParallelPushDataToFASTER")
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

  // Get local id
  std::vector<int64_t> _ids;
  std::vector<int64_t> _ids_orginal;
  for (int64_t i = 0; i < id_size; ++i) {
    int64_t _id = id_tensor_data[i];
    CHECK_GE(_id, 0);
    _ids.push_back(_id);
    _ids_orginal.push_back(i);
  }
  // Copy local data
  FasterContext* ctx_ptr = FasterContext::getInstance();
  store_t* store = ctx_ptr->store;
  std::vector<std::thread> threads;
  for (int64_t tid = 0; tid < nthread; tid++) {
    threads.push_back(std::thread([&, tid] {
      int64_t b = tid * id_size / nthread;
      int64_t e = (tid + 1) * id_size / nthread;
      e = e < id_size ? e : id_size;

      store->StartSession();
      for (int64_t i = b; i < e; ++i) {
        CHECK_GE(id_size*row_size, _ids_orginal[i] * row_size + row_size);
        CHECK_GE(_ids[i], 0);
        auto callback = [](FASTER::core::IAsyncContext* ctxt, FASTER::core::Status result) {
          FASTER::core::CallbackContext<UpsertContext> context{ ctxt };
        };
        UpsertContext context{ static_cast<uint64_t>(_ids[i]),
          data_tensor_char + _ids_orginal[i] * row_size, row_size };
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
