import random
import numpy as np

# ===============================
# Tham số hệ thống SIMO
# ===============================
M = 4                  # số anten thu
N0 = 1                 # công suất nhiễu
h = np.random.randn(M) # kênh truyền Rayleigh

P_min = 0.01
P_max = 10

# ===============================
# Hàm fitness: SNR
# ===============================
def fitness(P):
    return (P * np.sum(h**2)) / N0

# ===============================
# Grey Wolf Optimizer
# ===============================
class GWO:
    def __init__(self, wolves=10, iterations=30):
        self.wolves = wolves
        self.iterations = iterations

    def optimize(self):
        # Khởi tạo công suất
        P = np.random.uniform(P_min, P_max, self.wolves)

        for t in range(self.iterations):
            # Đánh giá fitness
            scores = np.array([fitness(p) for p in P])

            # Chọn alpha, beta, delta
            idx = np.argsort(scores)[::-1]
            alpha, beta, delta = P[idx[0]], P[idx[1]], P[idx[2]]

            a = 2 - 2 * t / self.iterations

            for i in range(self.wolves):
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * alpha - P[i])
                X1 = alpha - A1 * D_alpha

                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * beta - P[i])
                X2 = beta - A2 * D_beta

                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * delta - P[i])
                X3 = delta - A3 * D_delta

                P[i] = np.clip((X1 + X2 + X3) / 3, P_min, P_max)

            print(f"Iteration {t+1}: Best SNR = {fitness(alpha):.4f}")

        return alpha, fitness(alpha)

# ===============================
# Chạy thuật toán
# ===============================
if __name__ == "__main__":
    gwo = GWO()
    best_P, best_SNR = gwo.optimize()
    print("\nKẾT QUẢ CUỐI:")
    print("Công suất phát tối ưu:", best_P)
    print("SNR tối ưu:", best_SNR)