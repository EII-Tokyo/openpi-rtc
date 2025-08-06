
# import jax.numpy as jnp
# import jax

# def make_W_diag(H: int, d: int, s: int) -> jnp.ndarray:
#     """
#     生成论文中定义的权重向量 W ∈ ℝ^H
#     参数
#     ----
#     H : int  # 序列总长度
#     d : int  # “确定区”阈值
#     s : int  # “截断”窗口长度
#     返回
#     ----
#     W : jnp.ndarray, shape (H,)
#     """
#     # H = self.action_horizon
#     i = jnp.arange(H)           # 0,1,2,...,H-1

#     # 三段式条件
#     cond_1 = i < d
#     cond_2 = (i >= d) & (i < H - s)
#     cond_3 = i >= H - s         # 其实可以直接 else

#     # 段 (1): 全 1
#     w1 = jnp.ones_like(i, dtype=float)

#     # 段 (2): 指数递减
#     # c_i = (H - s - i) / (H - s - d + 1)
#     c_i = (H - s - i) / (H - s - d + 1)
#     w2  = jnp.exp(c_i) - 1
#     w2  = c_i * w2 / (jnp.e - 1)      # (e^{c_i} - 1) / (e - 1)

#     # 段 (3): 全 0
#     w3 = jnp.zeros_like(i, dtype=float)

#     # 合并三段
#     W = jnp.where(cond_1, w1,
#         jnp.where(cond_2, w2, w3)
#     )

#     D = jnp.diag(W)
#     D_batch = jnp.stack([D] * 1, axis=0)
#     return D_batch

# jax.vjp(make_W_diag, 50, 5, 5, argnums=0)
# # ====== demo ======
# H, d, s = 50, 5, 5
# D = make_W_diag(H, d, s)
# print(D[0][44], D.shape)


# def vjp_test(A, B):
#     return jnp.matmul(A, B), jnp.matmul(A, B)

# A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# B = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# _, vjp_fn = jax.vjp(vjp_test, A, B)
# a, b = vjp_fn((jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])))
# print(_)

import matplotlib.pyplot as plt

def draw(time):
    r_t = time * time / (time * time + (1 - time) * (1 - time))
    result = time / ((1 - time) * r_t * r_t + 1e-6)
    return result


X = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

# 计算对应的y值
y = [draw(x) for x in X]
print(y)
# 创建图像
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'b-o', linewidth=2, markersize=8, label='draw(t)')

# 设置标题和标签
plt.title('函数图像: draw(t)', fontsize=14)
plt.xlabel('t', fontsize=12)
plt.ylabel('draw(t)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# 保存图像
plt.tight_layout()
plt.savefig('draw_function.png', dpi=300, bbox_inches='tight')
plt.close() 