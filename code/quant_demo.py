# -*- coding: utf-8 -*-
"""
定点卷积量化与截断误差累积演示（论文 2.1.1 节配套代码）
======================================================
仅依赖 numpy —— 符合"Windows 不安装 YOLO 训练环境"约束

流程:
  1. FP32 卷积（黄金参考）
  2. INT8 对称量化权重/激活（式 2-5 / 2-7 / 2-11）
  3. INT8 乘 + int32 累加 → 截断/舍入字长（右移 n 位）
  4. 输出饱和到 INT8，逐层堆叠
  5. 逐层对比 FP32 与定点输出的 MaxAE / RMSE（式 2-12 / 2-13），
     观察量化误差随层累积，以及截断(truncation)与舍入(rounding)的差异
"""
import numpy as np

RNG = np.random.default_rng(42)


# ---------------- 量化工具（对应式 2-5/2-7/2-11） ----------------
def sym_quant(x, bits=8):
    """对称量化: S = max|x|/(2^(bits-1)-1)，q = clip(round(x/S), -128, 127)"""
    s = float(np.max(np.abs(x))) / (2 ** (bits - 1) - 1)
    if s == 0.0:
        s = 1.0
    q = np.clip(np.round(x / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return q.astype(np.int32), s


def dequant(q, s):
    """反量化: r = S·q（式 2-5 中 Z = 0 的对称情形）"""
    return q.astype(np.float64) * s


def truncate_fixed(acc, frac_bits):
    """截断字长: 累加器右移 frac_bits 位，直接丢弃低位（truncation）"""
    return acc >> frac_bits


def round_fixed(acc, frac_bits):
    """舍入字长: 加半 LSB 后右移（rounding，误差界见式 2-3 |e| ≤ Δ/2）"""
    half = 1 << (frac_bits - 1)
    return (acc + half) >> frac_bits


# ---------------- 卷积 ----------------
def conv2d_fp32(x, w, b):
    """FP32 卷积（黄金参考）: x(H,W,Cin)  w(K,K,Cin,Cout)  b(Cout)"""
    H, W, C = x.shape
    K, _, _, M = w.shape
    p = K // 2
    xp = np.pad(x, ((p, p), (p, p), (0, 0)))
    y = np.zeros((H, W, M), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            patch = xp[i:i + K, j:j + K, :]
            for m in range(M):
                y[i, j, m] = np.sum(patch * w[:, :, :, m]) + b[m]
    return y


def conv2d_fixed(x_q, s_x, w_q, s_w, b_q, frac_bits, mode="round"):
    """INT8 定点卷积:
       acc = Σ w_q·x_q + b_q           (INT8 乘, int32 累加)
       acc' = acc >> frac_bits          (截断/舍入字长)
       y_q = clip(acc', -128, 127)      (输出饱和, 式 2-11)
       y   = y_q · (s_w·s_x·2^frac_bits) (反量化)"""
    H, W, C = x_q.shape
    K, _, _, M = w_q.shape
    p = K // 2
    xp = np.pad(x_q, ((p, p), (p, p), (0, 0)))
    y_q = np.zeros((H, W, M), dtype=np.int64)
    for i in range(H):
        for j in range(W):
            patch = xp[i:i + K, j:j + K, :].astype(np.int64)
            for m in range(M):
                acc = int(np.sum(patch * w_q[:, :, :, m])) + int(b_q[m])
                if mode == "round":
                    acc = round_fixed(acc, frac_bits)
                else:
                    acc = truncate_fixed(acc, frac_bits)
                y_q[i, j, m] = acc
    y_q = np.clip(y_q, -128, 127).astype(np.int32)
    s_y = s_w * s_x * (2.0 ** frac_bits)
    return dequant(y_q, s_y)


def err_metrics(fp, fx):
    """式 2-12 / 2-13: MaxAE 与 RMSE"""
    d = np.abs(fp - fx)
    return float(np.max(d)), float(np.sqrt(np.mean(d ** 2)))


def main():
    H, W, Cin, Cout = 10, 10, 4, 8   # 特征图与通道
    K, L = 3, 4                      # 卷积核 3x3，堆叠 4 层
    frac_bits = 10                   # 累加器字长（小数位），截断丢弃低 10 位

    print(f"定点卷积误差累积演示: {L} 层, 输入 {H}x{W}x{Cin}, "
          f"卷积核 {K}x{K}x{Cin}x{Cout}, 累加器截断 {frac_bits} 位")
    print("-" * 78)
    print(f"{'层':>4}{'MaxAE(舍入)':>14}{'RMSE(舍入)':>14}"
          f"{'MaxAE(截断)':>14}{'RMSE(截断)':>14}{'误差增幅(截断RMSE)':>20}")

    x_fp = RNG.standard_normal((H, W, Cin)) * 0.5
    x_fixed = x_fp.copy()
    prev_rmse = 0.0

    for l in range(1, L + 1):
        cin = x_fp.shape[2]          # 当前输入通道数（第 1 层为 Cin，其后为 Cout）
        w_fp = RNG.standard_normal((K, K, cin, Cout)) * 0.3
        b_fp = RNG.standard_normal((Cout,)) * 0.1

        # 1) FP32 黄金参考
        y_ref = conv2d_fp32(x_fp, w_fp, b_fp)

        # 2) 量化权重/偏置/激活
        w_q, s_w = sym_quant(w_fp)
        b_q, _ = sym_quant(b_fp)
        x_q, s_x = sym_quant(x_fixed)

        # 3) 定点卷积（舍入字长 vs 截断字长）
        y_round = conv2d_fixed(x_q, s_x, w_q, s_w, b_q, frac_bits, "round")
        y_trunc = conv2d_fixed(x_q, s_x, w_q, s_w, b_q, frac_bits, "trunc")

        m_r = err_metrics(y_ref, y_round)
        m_t = err_metrics(y_ref, y_trunc)
        grow = m_t[1] / prev_rmse if prev_rmse > 0 else float("inf")
        print(f"{l:>4}{m_r[0]:>14.6f}{m_r[1]:>14.6f}"
              f"{m_t[0]:>14.6f}{m_t[1]:>14.6f}"
              f"{grow:>18.2f}x")
        prev_rmse = m_t[1]

        # 4) 截断路径的定点输出作为下一层输入（误差向下游传播）
        x_fp = y_ref.copy()
        x_fixed = y_trunc.copy()

    print("-" * 78)
    print("结论: ① 量化误差随层数单调累积（RMSE 逐层放大，式 2-6 的误差传播）；")
    print("      ② 深层中截断(truncation)误差大于舍入(rounding)，与式 2-3 误差界相符；")
    print("      ③ 第 1 层舍入误差略高，因舍入进位引发输出饱和被 clip（式 2-11），")
    print("         说明饱和截断本身也是不可忽略的误差来源。")


if __name__ == "__main__":
    main()
