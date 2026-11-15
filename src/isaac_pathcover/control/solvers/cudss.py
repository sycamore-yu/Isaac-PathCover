"""Structured linear solves through NVIDIA's official Python bindings.

CSR/input/output buffers belong to PyTorch; cuDSS owns its internal factors
and workspace. Already-dense tiny bands use cuBLAS batched pivoted LU.
The official nvmath binding is the only vendor interface:
there is no project-built native extension or ctypes wrapper.
"""

import numpy as np
import torch


def linear_system(batch, stages, width, device):
    # Two-stage block-tridiagonal bands are already fully dense.
    if stages == 2 and stages * width <= 32:
        return BatchedLU(batch, stages, width, device)
    return SparseLinearSystem(batch, stages, width, device)


class BatchedLU:
    """Stable column-major factors and RHS pointers for tiny dense bands."""

    def __init__(self, batch, stages, width, device):
        from nvmath.bindings import cublas

        self.cublas = cublas
        self.stream = torch.cuda.current_stream(device)
        self.batch, self.size = batch, stages * width
        n = self.size
        indices = []
        for column in range(n):
            ct, c = divmod(column, width)
            for row in range(n):
                t, r = divmod(row, width)
                indices.append((t * width + r) * 3 * width + (ct - t + 1) * width + c)
        self.index = torch.tensor(indices, dtype=torch.int64, device=device)
        self.lu = torch.empty((batch, n * n), dtype=torch.float64, device=device)
        self.pivots = torch.empty((batch, n), dtype=torch.int32, device=device)
        self.info = torch.empty(batch, dtype=torch.int32, device=device)
        self.rhs = torch.empty((batch, stages, width), dtype=torch.float64, device=device)
        self.solution = torch.empty_like(self.rhs)
        self.lu_ptrs = torch.tensor([self.lu.data_ptr() + b * n * n * 8 for b in range(batch)],
                                   dtype=torch.int64, device=device)
        self.x_ptrs = torch.tensor([self.solution.data_ptr() + b * n * 8 for b in range(batch)],
                                  dtype=torch.int64, device=device)
        self.solve_info = np.zeros(1, dtype=np.int32)
        self.handle = None
        try:
            self.handle = cublas.create()
            cublas.set_stream(self.handle, self.stream.cuda_stream)
        except Exception:
            self.close()
            raise

    def factorize(self, band):
        torch.index_select(band.flatten(1), 1, self.index, out=self.lu)
        self.cublas.dgetrf_batched(self.handle, self.size, self.lu_ptrs.data_ptr(), self.size,
                                  self.pivots.data_ptr(), self.info.data_ptr(), self.batch)

    def solve(self):
        self.solution.copy_(self.rhs)
        self.cublas.dgetrs_batched(self.handle, 0, self.size, 1, self.lu_ptrs.data_ptr(),
                                  self.size, self.pivots.data_ptr(), self.x_ptrs.data_ptr(),
                                  self.size, self.solve_info.ctypes.data, self.batch)
        return self.solution

    def close(self):
        self.stream.synchronize()
        if self.handle is not None:
            self.cublas.destroy(self.handle)
            self.handle = None


class SparseLinearSystem:
    """Source-style block-diagonal CSR with stable CUDA Graph solve buffers."""

    def __init__(self, batch, stages, width, device):
        from nvmath.bindings import cudss

        self.cudss = cudss
        self.stream = torch.cuda.current_stream(device)
        self.batch, self.stages, self.width = batch, stages, width
        size = stages * width
        columns, indices, row_ptr = [], [], [0]
        for t in range(stages):
            for i in range(width):
                for neighbour in range(max(0, t - 1), min(stages, t + 2)):
                    columns.extend(range(neighbour * width, (neighbour + 1) * width))
                    offset = (t * width + i) * 3 * width + (neighbour - t + 1) * width
                    indices.extend(range(offset, offset + width))
                row_ptr.append(len(columns))
        nnz = len(columns)
        self.index = torch.tensor(indices, dtype=torch.int64, device=device)
        rows = np.r_[np.concatenate([np.asarray(row_ptr[:-1]) + b * nnz
                                     for b in range(batch)]), batch * nnz].astype(np.int32)
        cols = np.concatenate([np.asarray(columns) + b * size
                               for b in range(batch)]).astype(np.int32)
        total_size, declared_nnz = batch * size, batch * nnz
        self.row_ptr = torch.as_tensor(rows, device=device)
        self.columns = torch.as_tensor(cols, device=device)
        self.values = torch.empty((batch, nnz), dtype=torch.float64, device=device)
        self.rhs = torch.empty((batch, stages, width), dtype=torch.float64, device=device)
        self.solution = torch.empty_like(self.rhs)
        self.handle = self.config = self.data = self.matrix = self.x = self.b = None
        self.planned = False
        try:
            self.handle = cudss.create()
            cudss.set_stream(self.handle, self.stream.cuda_stream)
            self.config = cudss.config_create()
            self.data = cudss.data_create(self.handle)
            self._set_config(cudss.ConfigParam.MATCHING_ALG, int(cudss.MatchingAlg.NONE))
            self.matrix = cudss.matrix_create_csr(
                total_size, total_size, declared_nnz, self.row_ptr.data_ptr(), 0,
                self.columns.data_ptr(), self.values.data_ptr(), cudss.DataType.R_32I,
                cudss.DataType.R_32I, cudss.DataType.R_64F,
                cudss.MatrixType.GENERAL,
                cudss.MatrixViewType.FULL, cudss.IndexBase.ZERO,
            )
            self.x = cudss.matrix_create_dn(total_size, 1, total_size,
                                            self.solution.data_ptr(), cudss.DataType.R_64F,
                                            cudss.Layout.COL_MAJOR)
            self.b = cudss.matrix_create_dn(total_size, 1, total_size,
                                            self.rhs.data_ptr(), cudss.DataType.R_64F,
                                            cudss.Layout.COL_MAJOR)
        except Exception:
            self.close()
            raise

    def _set_config(self, key, value):
        scalar = np.array([value], dtype=np.int32)
        self.cudss.config_set(self.config, key, scalar.ctypes.data, scalar.nbytes)

    def factorize(self, band):
        self.values.copy_(band.flatten(1).index_select(1, self.index))
        if not self.planned:
            self._execute(self.cudss.Phase.ANALYSIS)
            self.planned = True
        self._execute(self.cudss.Phase.FACTORIZATION)

    def _execute(self, phase):
        self.cudss.execute(self.handle, phase, self.config, self.data, self.matrix, self.x, self.b)

    def solve(self):
        """Read the stable RHS and write the stable solution; no allocation or sync."""
        self._execute(self.cudss.Phase.SOLVE)
        return self.solution

    def close(self):
        self.stream.synchronize()
        for name in ("matrix", "x", "b"):
            value = getattr(self, name)
            if value is not None:
                self.cudss.matrix_destroy(value)
                setattr(self, name, None)
        if self.data is not None:
            self.cudss.data_destroy(self.handle, self.data)
            self.data = None
        if self.config is not None:
            self.cudss.config_destroy(self.config)
            self.config = None
        if self.handle is not None:
            self.cudss.destroy(self.handle)
            self.handle = None
