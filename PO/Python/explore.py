import numpy as np
import random

class Explore:
    def __init__(self, sol_range, size, pCR=0.5):

        self.sol_range = sol_range
        self.size = size
        self.pCR = pCR

    def exploration_phase(self, sol, fitness_values):

        sorted_indices = np.argsort(fitness_values)
        sol = [sol[i] for i in sorted_indices]
        fitness_values = [fitness_values[i] for i in sorted_indices]
        
        new_sol = []
        new_fitness = []
        
        for i in range(self.size):
            current = sol[i]
            
            # Tạo list indices khác i
            available_indices = list(range(self.size))
            available_indices.remove(i)
            
            # Chọn 6 indices ngẫu nhiên (a,b,c,d,e,f)
            if len(available_indices) >= 6:
                selected_indices = random.sample(available_indices, 6)
                a, b, c, d, e, f = selected_indices
            else:
                # Nếu không đủ 6, lặp lại một số indices
                selected_indices = random.choices(available_indices, k=6)
                a, b, c, d, e, f = selected_indices
            
            # G coefficient
            G = 2 * random.random() - 1
            
            # Tạo individual mới
            new_individual = {}
            
            # Chọn j0 - tham số bắt buộc phải thay đổi
            param_keys = list(self.sol_range.keys())
            j0 = random.choice(param_keys)
            
            for param, range_info in self.sol_range.items():
                if param == j0 or random.random() <= self.pCR:
                    if random.random() < 0.5:
                        # Tạo giá trị ngẫu nhiên mới (Eq 25 - phần đầu)
                        if range_info['type'] == 'int':
                            new_individual[param] = random.randint(range_info['min'], range_info['max'])
                        elif range_info['type'] == 'float':
                            new_individual[param] = random.uniform(range_info['min'], range_info['max'])
                        elif range_info['type'] == 'log_uniform':
                            log_min = np.log10(range_info['min'])
                            log_max = np.log10(range_info['max'])
                            new_individual[param] = 10 ** random.uniform(log_min, log_max)
                        elif range_info['type'] == 'choice':
                            new_individual[param] = random.choice(range_info['options'])
                    else:
                        # Sử dụng công thức PUMA (Eq 25 - phần sau)
                        if range_info['type'] in ['int', 'float', 'log_uniform']:
                            term1 = sol[a][param] + G * (sol[a][param] - sol[b][param])
                            term2 = G * (((sol[a][param] - sol[b][param]) - 
                                        (sol[c][param] - sol[d][param])) + 
                                    ((sol[c][param] - sol[d][param]) - 
                                        (sol[e][param] - sol[f][param])))
                            new_val = term1 + term2
                            
                            if range_info['type'] == 'int':
                                new_individual[param] = int(round(new_val))
                            else:
                                new_individual[param] = new_val
                        elif range_info['type'] == 'choice':
                            new_individual[param] = random.choice(range_info['options'])
                else:
                    # Giữ nguyên giá trị cũ
                    new_individual[param] = current[param]
            
            # Clip parameters
            new_individual = self.clip_individual(new_individual)
            
            # Đánh giá fitness
            new_fitness_val = self.evaluate_individual(new_individual)
            
            # So sánh và cập nhật (tối đa hóa)
            if new_fitness_val > fitness_values[i]:
                new_sol.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_sol.append(current)
                new_fitness.append(fitness_values[i])
                # Cập nhật pCR khi không cải thiện (Eq 30)
                p = (1 - self.pCR) / self.size
                self.pCR = min(self.pCR + p, 0.9)
                
        return new_sol, new_fitness
