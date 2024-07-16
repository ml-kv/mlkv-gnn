from .._ffi.function import _init_api
from .._ffi.runtime_ctypes import DGLDataType
from .. import backend as F

__all__ = ['']

class FASTER:
    def __init__(self, table_size_bytes, log_size_bytes, path, log_mutable_fraction,
            feat_dim, feat_dtype):
        self.table_size_bytes = table_size_bytes
        self.log_size_bytes = log_size_bytes
        self.path = path
        self.log_mutable_fraction = log_mutable_fraction
        self.feat_dim = feat_dim
        self.feat_dtype = feat_dtype

    def open_faster(self):
        _CAPI_DGLStorageOpenFASTER(self.table_size_bytes, self.log_size_bytes,
            self.path, self.log_mutable_fraction)

    def recover_faster(self, checkpoint_token):
        _CAPI_DGLStorageRecoverFASTER(self.table_size_bytes, self.log_size_bytes,
            self.path, self.log_mutable_fraction, checkpoint_token)

    def checkpoint_faster(self):
        _CAPI_DGLStorageCheckpointFASTER()

    def pull_data_from_faster(self, id_tensor):
        if id_tensor.dtype != F.int64:
            raise TypeError("Please use int64 for feature id tensors")
        res_tensor = _CAPI_DGLStoragePullDataFromFASTER(F.zerocopy_to_dgl_ndarray(id_tensor),
                                                        self.feat_dim,
                                                        DGLDataType(get_type_str(self.feat_dtype)))
        return F.zerocopy_from_dgl_ndarray(res_tensor)

    def push_data_to_faster(self, id_tensor, data_tensor):
        if id_tensor.dtype != F.int64:
            raise TypeError("Please use int64 for feature id tensors")
        _CAPI_DGLStoragePushDataToFASTER(F.zerocopy_to_dgl_ndarray(id_tensor),
                                         F.zerocopy_to_dgl_ndarray(data_tensor))

    def parallel_push_data_to_faster(self, num_thread, id_tensor, data_tensor):
        if id_tensor.dtype != F.int64:
            raise TypeError("Please use int64 for feature id tensors")
        _CAPI_DGLStorageParallelPushDataToFASTER(num_thread,
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

_init_api("dgl.storages.py_faster")
