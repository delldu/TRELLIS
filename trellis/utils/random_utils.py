import numpy as np

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    # 特点：
    #     低差异序列，比随机采样更均匀
    #     不同维度的序列相互独立
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    # 原理：
    #     第 1 维固定为n/num_samples（均匀分布）
    #     其余dim-1维使用 Halton 序列
    # 优势：
    #     在一维上完全均匀
    #     整体保持低差异特性
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0), remap=False):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    if remap:
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3

    # 将(u, v)转换为球坐标系：
    #     theta（倾角）：从 [-π/2, π/2]（对应从南到北）
    #     phi（方位角）：从 [0, 2π]（绕球一周）
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]