import numpy as np

class SymmetricMatrixInverter:
    def __init__(self, n_size=16, block_size=4, lambda_reg=1e-5):
        """
        初始化算法环境
        :param n_size: 矩阵维度 (必须是2的幂)
        :param block_size: 基础分块大小 (必须是2的幂)
        :param lambda_reg: 对角线加载常数，用于稳定性
        """
        self.n = n_size
        self.block_size = block_size
        
        # 1. 构造随机的稳定实对称矩阵 A = H^T * H + lambda * I
        H = np.random.randn(self.n, self.n)
        self.A = H.T @ H + lambda_reg * np.eye(self.n)
        
    # ==========================================
    # 基础算子层：用最基础的 + - * / sqrt 实现
    # ==========================================
    def _base_cholesky(self, A):
        """标量级别计算 Cholesky: A = LL^T"""
        n = A.shape[0]
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i, k] * L[j, k] for k in range(j))
                if i == j:
                    L[i, i] = np.sqrt(A[i, i] - s)
                else:
                    L[i, j] = (A[i, j] - s) / L[j, j]
        return L

    def _base_ldl(self, A):
        """标量级别计算 LDL^T: 无开方运算"""
        n = A.shape[0]
        L = np.eye(n)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                s = sum(L[i, k] * L[j, k] * D[k, k] for k in range(j))
                L[i, j] = (A[i, j] - s) / D[j, j]
            s = sum((L[i, k] ** 2) * D[k, k] for k in range(i))
            D[i, i] = A[i, i] - s
        return L, D

    def _base_tril_inv(self, L, is_unit_diag=False):
        """标量级别计算下三角矩阵求逆 (前向代换)"""
        n = L.shape[0]
        X = np.zeros((n, n))
        for i in range(n):
            X[i, i] = 1.0 if is_unit_diag else 1.0 / L[i, i]
            for j in range(i):
                s = sum(L[i, k] * X[k, j] for k in range(j, i))
                X[i, j] = -s if is_unit_diag else -s / L[i, i]
        return X

    def _invert_diagonal(self, D):
        """对角矩阵求逆"""
        n = D.shape[0]
        D_inv = np.zeros((n, n))
        for i in range(n):
            D_inv[i, i] = 1.0 / D[i, i]
        return D_inv

    # ==========================================
    # 算法层：递归分块调度器
    # ==========================================
    def _block_cholesky_core(self, A):
        n = A.shape[0]
        # 到达基础分块大小，调用标量算子
        if n <= self.block_size:
            L = self._base_cholesky(A)
            L_inv = self._base_tril_inv(L)
            return L, L_inv

        mid = n // 2
        A11, A12 = A[:mid, :mid], A[:mid, mid:]
        A21, A22 = A[mid:, :mid], A[mid:, mid:]

        # 1. 求解左上角
        L11, L11_inv = self._block_cholesky_core(A11)

        # 2. 求解左下角 L21 = A21 * L11_inv^T (这里全是矩阵乘，硬件性能极高)
        L21 = A21 @ L11_inv.T

        # 3. Schur 补更新 S = A22 - L21 * L21^T
        S = A22 - L21 @ L21.T

        # 4. 求解右下角
        L22, L22_inv = self._block_cholesky_core(S)

        # 5. 组合 L 和 L_inv
        L = np.zeros((n, n))
        L[:mid, :mid] = L11
        L[mid:, :mid] = L21
        L[mid:, mid:] = L22

        L_inv = np.zeros((n, n))
        L_inv[:mid, :mid] = L11_inv
        L_inv[mid:, mid:] = L22_inv
        L_inv[mid:, :mid] = -L22_inv @ L21 @ L11_inv

        return L, L_inv

    def _block_ldl_core(self, A):
        n = A.shape[0]
        # 到达基础分块大小，调用标量算子
        if n <= self.block_size:
            L, D = self._base_ldl(A)
            L_inv = self._base_tril_inv(L, is_unit_diag=True)
            return L, D, L_inv

        mid = n // 2
        A11, A12 = A[:mid, :mid], A[:mid, mid:]
        A21, A22 = A[mid:, :mid], A[mid:, mid:]

        L11, D11, L11_inv = self._block_ldl_core(A11)

        D11_inv = self._invert_diagonal(D11)
        # L21 = A21 * L11_inv^T * D11_inv
        L21 = A21 @ L11_inv.T @ D11_inv

        # Schur补更新
        S = A22 - L21 @ D11 @ L21.T

        L22, D22, L22_inv = self._block_ldl_core(S)

        L = np.eye(n)
        L[:mid, :mid] = L11
        L[mid:, :mid] = L21
        L[mid:, mid:] = L22

        D = np.zeros((n, n))
        D[:mid, :mid] = D11
        D[mid:, mid:] = D22

        L_inv = np.eye(n)
        L_inv[:mid, :mid] = L11_inv
        L_inv[mid:, mid:] = L22_inv
        L_inv[mid:, :mid] = -L22_inv @ L21 @ L11_inv

        return L, D, L_inv

    # ==========================================
    # 对外接口层
    # ==========================================
    def inverse_via_cholesky(self):
        _, L_inv = self._block_cholesky_core(self.A)
        return L_inv.T @ L_inv

    def inverse_via_ldl(self):
        _, D, L_inv = self._block_ldl_core(self.A)
        D_inv = self._invert_diagonal(D)
        return L_inv.T @ D_inv @ L_inv


# === 测试与比较 ===
if __name__ == "__main__":
    N = 16
    BLOCK_SIZE = 16
    
    print(f"正在测试矩阵求逆，矩阵大小: {N}x{N}, 分块大小: {BLOCK_SIZE}x{BLOCK_SIZE}\n")
    
    inverter = SymmetricMatrixInverter(n_size=N, block_size=BLOCK_SIZE)
    
    # 1. 我们的 Cholesky 分块求逆
    inv_cholesky = inverter.inverse_via_cholesky()
    
    # 2. 我们的 LDL 分块求逆
    inv_ldl = inverter.inverse_via_ldl()
    
    # 3. Numpy 官方库对照组
    inv_numpy = np.linalg.inv(inverter.A)
    
    # 计算误差 (Frobenius 范数)
    err_cholesky = np.linalg.norm(inv_cholesky - inv_numpy)
    err_ldl = np.linalg.norm(inv_ldl - inv_numpy)
    
    print(f"✅ 分块 Cholesky 求逆误差 (vs Numpy): {err_cholesky:.4e}")
    print(f"✅ 分块 LDL^T 求逆误差 (vs Numpy): {err_ldl:.4e}")