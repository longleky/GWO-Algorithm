import numpy as np
# Tham số hệ thống SIMO
Nr = 4
h = np.random.randn(Nr)
N0 = 1
Pmax = 10
# Logistic map
def logistic_map(x, mu=4.0):
    return mu * x * (1 - x)
# Hàm mục tiêu
def fitness(P):
    snr = P * np.sum(h**2) / N0
    return snr / P
# Tham số CGWO
num_wolves = 10
max_iter = 30
a = 2
# Khởi tạo sói
positions = np.random.uniform(0.1, Pmax, num_wolves)
# Chaotic sequence
chaos = np.random.rand()
for t in range(max_iter):
    fitness_vals = np.array([fitness(p) for p in positions])
    idx = np.argsort(fitness_vals)[::-1]

    alpha = positions[idx[0]]
    beta  = positions[idx[1]]
    delta = positions[idx[2]]

    for i in range(num_wolves):
        chaos = logistic_map(chaos)

        A = 2 * a * chaos - a
        C = 2 * chaos

        D_alpha = abs(C * alpha - positions[i])
        X1 = alpha - A * D_alpha

        D_beta = abs(C * beta - positions[i])
        X2 = beta - A * D_beta

        D_delta = abs(C * delta - positions[i])
        X3 = delta - A * D_delta

        positions[i] = np.clip((X1 + X2 + X3) / 3, 0.1, Pmax)

    a -= 2 / max_iter
    print(f"Iteration {t+1}: Best fitness = {fitness(alpha)}")

print("\nKẾT QUẢ CUỐI:")
print("Công suất tối ưu:", alpha)
print("Giá trị hàm mục tiêu:", fitness(alpha))
