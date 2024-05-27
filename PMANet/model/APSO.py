import random

class Ant:
    def __init__(self, path_length):
        self.path = [random.randint(0, 1) for _ in range(path_length)]
        self.velocity = [0] * path_length

def initialize_ants(num_ants, path_length):
    """
    初始化蚂蚁群体
    """
    return [Ant(path_length) for _ in range(num_ants)]

def update_velocity_and_path(ant, global_best_path, alpha=0.5):
    """
    更新蚂蚁的路径和速度
    """
    for i in range(len(ant.path)):
        if random.random() < alpha:
            ant.path[i] = global_best_path[i]
    return ant

def fitness(path):
    # path: 一个包含选择的投资标的（0或1）的序列
    returns = sum([expected_returns[i] * path[i] for i in range(len(path))])
    risk = sum([risk_matrix[i][j] * path[i] * path[j] for i in range(len(path)) for j in range(len(path))])
    return returns - risk

def APSO(num_ants, path_length, num_iterations):
    """
    APSO算法的主函数
    """
    # 初始化蚂蚁群体
    ants = initialize_ants(num_ants, path_length)
    global_best_path = [random.randint(0, 1) for _ in range(path_length)]
    global_best_fitness = fitness(global_best_path)
    
    for iteration in range(num_iterations):
        for ant in ants:
            # 更新蚂蚁路径和速度
            ant = update_velocity_and_path(ant, global_best_path)
        
        # 更新全局最优路径
        for ant in ants:
            current_fitness = fitness(ant.path)
            if current_fitness < global_best_fitness:
                global_best_path = ant.path[:]
                global_best_fitness = current_fitness
        
        print(f"Iteration {iteration+1}/{num_iterations}, Global Best Fitness: {global_best_fitness}")

    return global_best_path

