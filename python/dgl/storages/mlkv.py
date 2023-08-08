from .._ffi.function import _init_api
from .._ffi.runtime_ctypes import DGLType
from .. import backend as F

__all__ = ['']

class MLKV:
    def __init__(self, table_size_bytes, log_size_bytes, path, log_mutable_fraction,
            feat_dim, feat_dtype):
        self.table_size_bytes = table_size_bytes
        self.log_size_bytes = log_size_bytes
        self.path = path
        self.log_mutable_fraction = log_mutable_fraction
        self.feat_dim = feat_dim
        self.feat_dtype = feat_dtype

    def open_mlkv(self):
        _CAPI_DGLStorageOpenMLKV(self.table_size_bytes, self.log_size_bytes,
            self.path, self.log_mutable_fraction)

    def recover_mlkv(self, checkpoint_token):
        _CAPI_DGLStorageRecoverMLKV(self.table_size_bytes, self.log_size_bytes,
            self.path, self.log_mutable_fraction, checkpoint_token)

    def checkpoint_mlkv(self):
        _CAPI_DGLStorageCheckpointMLKV()

    def pull_data_from_mlkv(self, id_tensor):
        if id_tensor.dtype != F.int64:
            raise TypeError("Please use int64 for feature id tensors")
        res_tensor = _CAPI_DGLStoragePullDataFromMLKV(F.zerocopy_to_dgl_ndarray(id_tensor.cpu()),
                                                      self.feat_dim, DGLType(get_type_str(self.feat_dtype)))
        return F.zerocopy_from_dgl_ndarray(res_tensor)

    def pull_and_lookahead_data_from_mlkv(self, num_thread, id_tensor_pull, id_tensor_lookahead):
        if id_tensor_pull.dtype != F.int64 or id_tensor_lookahead.dtype != F.int64:
            raise TypeError("Please use int64 for feature id tensors")
        res_tensor = _CAPI_DGLStoragePullAndLookaheadDataFromMLKV(num_thread,
                                                                      F.zerocopy_to_dgl_ndarray(id_tensor_pull),
                                                                      F.zerocopy_to_dgl_ndarray(id_tensor_lookahead),
                                                                      self.feat_dim, DGLType(get_type_str(self.feat_dtype)))
        return F.zerocopy_from_dgl_ndarray(res_tensor)

    def async_parallel_lookahead_data_from_mlkv(self, num_thread, id_tensor):
        _CAPI_DGLStorageAsyncParallelLookaheadDataFromMLKV(num_thread,
                                                           F.zerocopy_to_dgl_ndarray(id_tensor),
                                                           self.feat_dim, DGLType(get_type_str(self.feat_dtype)))

    def async_push_data_to_mlkv(self, id_tensor, data_tensor):
        if id_tensor.dtype != F.int64:
            raise TypeError("Please use int64 for feature id tensors")
        _CAPI_DGLStorageAsyncPushDataToMLKV(F.zerocopy_to_dgl_ndarray(id_tensor),
                                            F.zerocopy_to_dgl_ndarray(data_tensor))

    def wait_async_threads_in_mlkv(self):
        _CAPI_DGLStorageWaitAsyncThreadsInMLKV()

    def push_data_to_mlkv(self, id_tensor, data_tensor):
        if id_tensor.dtype != F.int64:
            raise TypeError("Please use int64 for feature id tensors")
        _CAPI_DGLStoragePushDataToMLKV(F.zerocopy_to_dgl_ndarray(id_tensor),
                                       F.zerocopy_to_dgl_ndarray(data_tensor))

    def parallel_push_data_to_mlkv(self, num_thread, id_tensor, data_tensor):
        if id_tensor.dtype != F.int64:
            raise TypeError("Please use int64 for feature id tensors")
        _CAPI_DGLStorageParallelPushDataToMLKV(num_thread,
                                               F.zerocopy_to_dgl_ndarray(id_tensor),
                                               F.zerocopy_to_dgl_ndarray(data_tensor))

def get_type_str(dtype):
    """Get data type string
    """
    if 'float16' in str(dtype):
        return 'float16'
    elif 'float32' in str(dtype):
        return 'float32'
    elif 'float64' in str(dtype):
        return 'float64'
    elif 'uint8' in str(dtype):
        return 'uint8'
    elif 'int8' in str(dtype):
        return 'int8'
    elif 'int16' in str(dtype):
        return 'int16'
    elif 'int32' in str(dtype):
        return 'int32'
    elif 'int64' in str(dtype):
        return 'int64'
    else:
        raise RuntimeError('Unknown data type: %s' % str(dtype))

_init_api("dgl.storages.mlkv")
