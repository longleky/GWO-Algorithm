import random
import math

# random.seed(0) # Có thể bỏ chú thích để đảm bảo kết quả giống nhau mỗi lần chạy

## 1. Hàm Fitness (Hàm Sphere)
def sphere_fitness(individual):
    """Tính toán độ thích nghi (fitness) cho hàm Sphere."""
    fitness = 0.0
    for i in individual:
        fitness += i ** 2
    return fitness

# ====================================================================
## 2. Thuật toán GWO Gốc (CGWO)
# Sử dụng hàm random.random() truyền thống.

class ConventionalGWO():
    def __init__(self, iterations, pack_size, vector_size, fitness_func):
        self.iterations = iterations
        self.pack_size = pack_size
        self.vector_size = vector_size
        self.fitness = fitness_func
        
    def _create_wolf(self, min_range, max_range):
        """Tạo vị trí sói ngẫu nhiên."""
        return [(max_range - min_range) * random.random() + min_range 
                for _ in range(self.vector_size)]
  
    def _create_pack(self, min_range, max_range):
        """Tạo bầy sói."""
        return [self._create_wolf(min_range, max_range) for _ in range(self.pack_size)]
    
    def _get_random_value(self):
        """Lấy giá trị ngẫu nhiên truyền thống (trong [0, 1])."""
        return random.random()
    
    def hunt(self):
        wolf_pack = self._create_pack(-10, 10) 
        pack_fit = sorted([(self.fitness(i), i) for i in wolf_pack])
        
        for k in range(self.iterations):
            alpha, beta, delta = pack_fit[0][1], pack_fit[1][1], pack_fit[2][1]
            
            print(f'[CGWO] Iteration: {k:2d}, Best Fitness: {self.fitness(alpha):.6f}')
            
            a = 2 * (1 - k / self.iterations) # Giảm tuyến tính từ 2 về 0
            
            for i in range(self.pack_size):
                new_position = [0.0] * self.vector_size
                current_wolf = wolf_pack[i]
                
                # Tính toán A và C (sử dụng ngẫu nhiên truyền thống)
                r1_1, r1_2, r1_3 = self._get_random_value(), self._get_random_value(), self._get_random_value()
                r2_1, r2_2, r2_3 = self._get_random_value(), self._get_random_value(), self._get_random_value()
                
                A1, A2, A3 = a * (2 * r1_1 - 1), a * (2 * r1_2 - 1), a * (2 * r1_3 - 1)
                C1, C2, C3 = 2 * r2_1, 2 * r2_2, 2 * r2_3
                
                for j in range(self.vector_size):
                    # Bước 1: Tính toán X1, X2, X3
                    X1 = alpha[j] - A1 * abs(C1 * alpha[j] - current_wolf[j])
                    X2 = beta[j] - A2 * abs(C2 * beta[j] - current_wolf[j])
                    X3 = delta[j] - A3 * abs(C3 * delta[j] - current_wolf[j])
                    
                    # Bước 2: Cập nhật vị trí mới (Trung bình cộng)
                    new_position[j] = (X1 + X2 + X3) / 3.0
                
                # Cập nhật tham lam (Greedy Update)
                if self.fitness(new_position) < self.fitness(current_wolf):
                    wolf_pack[i] = new_position
         
            pack_fit = sorted([(self.fitness(i), i) for i in wolf_pack])
            
        return pack_fit[0]

# ====================================================================
## 3. Thuật toán Chaotic GWO (CGWO)
# Sử dụng Logistic Map để tạo ra các giá trị ngẫu nhiên.

class ChaoticGWO(ConventionalGWO):
    def __init__(self, iterations, pack_size, vector_size, fitness_func):
        super().__init__(iterations, pack_size, vector_size, fitness_func)
        # Khởi tạo giá trị hỗn loạn ban đầu z0 
        # Cần đảm bảo z0 không phải 0.25, 0.5, 0.75 để tránh chu kỳ lặp ngắn.
        self.logistic_state = 0.51 # Ví dụ giá trị khởi tạo
        
    def _get_chaotic_value(self):
        """Tạo giá trị hỗn loạn bằng Logistic Map (mu = 4)."""
        # Nếu giá trị Logistic Map tiến gần đến 0 hoặc 1, khởi tạo lại
        if self.logistic_state < 1e-6 or self.logistic_state > 1.0 - 1e-6:
             self.logistic_state = random.uniform(0.1, 0.9)
        
        # Công thức Logistic Map: z(k+1) = 4 * z(k) * (1 - z(k))
        self.logistic_state = 4 * self.logistic_state * (1 - self.logistic_state)
        return self.logistic_state
    
    def hunt(self):
        wolf_pack = self._create_pack(-10, 10)
        pack_fit = sorted([(self.fitness(i), i) for i in wolf_pack])
        
        for k in range(self.iterations):
            alpha, beta, delta = pack_fit[0][1], pack_fit[1][1], pack_fit[2][1]
            
            print(f'[CGWO_Chaotic] Iteration: {k:2d}, Best Fitness: {self.fitness(alpha):.6f}')
            
            a = 2 * (1 - k / self.iterations) # Giảm tuyến tính từ 2 về 0
            
            for i in range(self.pack_size):
                new_position = [0.0] * self.vector_size
                current_wolf = wolf_pack[i]
                
                # Tính toán A và C (SỬ DỤNG ÁNH XẠ HỖN LOẠN)
                r1_1, r1_2, r1_3 = self._get_chaotic_value(), self._get_chaotic_value(), self._get_chaotic_value()
                r2_1, r2_2, r2_3 = self._get_chaotic_value(), self._get_chaotic_value(), self._get_chaotic_value()

                A1, A2, A3 = a * (2 * r1_1 - 1), a * (2 * r1_2 - 1), a * (2 * r1_3 - 1)
                C1, C2, C3 = 2 * r2_1, 2 * r2_2, 2 * r2_3
                
                for j in range(self.vector_size):
                    # Bước 1: Tính toán X1, X2, X3
                    X1 = alpha[j] - A1 * abs(C1 * alpha[j] - current_wolf[j])
                    X2 = beta[j] - A2 * abs(C2 * beta[j] - current_wolf[j])
                    X3 = delta[j] - A3 * abs(C3 * delta[j] - current_wolf[j])
                    
                    # Bước 2: Cập nhật vị trí mới (Trung bình cộng)
                    new_position[j] = (X1 + X2 + X3) / 3.0
                
                # Cập nhật tham lam (Greedy Update)
                if self.fitness(new_position) < self.fitness(current_wolf):
                    wolf_pack[i] = new_position
         
            pack_fit = sorted([(self.fitness(i), i) for i in wolf_pack])
            
        return pack_fit[0]

# ====================================================================
## 4. Chạy Demo và So sánh

def main_demo():
    iterations = 20
    pack_size = 10
    vector_size = 3
    
    print("--- DEMO THUẬT TOÁN GWO GỐC (GWO) ---")
    model_cgwo = ConventionalGWO(iterations, pack_size, vector_size, sphere_fitness)
    best_cgwo = model_cgwo.hunt()
    print(f"\n✅ Kết quả GWO: Best Fitness = {best_cgwo[0]:.6f}, Position = {best_cgwo[1]}")
    
    print("\n" + "="*70)

    print("--- DEMO THUẬT TOÁN CHAOTIC GWO (CGWO) ---")
    model_cgwo_chaotic = ChaoticGWO(iterations, pack_size, vector_size, sphere_fitness)
    best_cgwo_chaotic = model_cgwo_chaotic.hunt()
    print(f"\n✅ Kết quả Chaotic GWO: Best Fitness = {best_cgwo_chaotic[0]:.6f}, Position = {best_cgwo_chaotic[1]}")

if __name__ == '__main__':
    main_demo()