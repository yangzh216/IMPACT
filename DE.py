import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn


# imagenet
mu = torch.tensor([0.485, 0.456, 0.406])  # 通道均值
std = torch.tensor([0.229, 0.224, 0.225])  # 通道标准差

# mu = torch.tensor([0.5071, 0.4867, 0.4408]).cuda()  # 通道均值
# std = torch.tensor([0.2675, 0.2565, 0.2761]).cuda()  # 通道标准差

# mu = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()  # 通道均值
# std = torch.tensor([0.2470, 0.2435, 0.2616]).cuda()  # 通道标准差



# IMG_SIZE = 224
# PATCH_SIZE = 4
# SEQ_SIZE = int((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE)) # 196

###########################################
#   parameters setting
F = 3
CR = 0.8
###########################################


query_count = 0


def init_population(population_size, patch_num, minipatch_num, img_size, tile_size):
    """
    init_population
    inputs:
        - population_size: population size
        - patch_num: patch number
        - minipatch_num: patch area
    rerturns:
        - population: [population_size, SEQ_SIZE]
    """
    individual_length = (img_size // tile_size) ** 2 # coding length

    # 初始化二进制种群
    binary_population = np.zeros((population_size, individual_length), dtype=int)
    # 初始化RGB像素值种群
    rgb_population = np.zeros((population_size, minipatch_num, 3), dtype=int) # shape: (1, 25, 64, 3)

    for p in range(population_size):  # 遍历每个个体
        # 每个个体随机选择 PATCH_NUM 个位置设置为 1
        patch_indices = np.random.choice(individual_length, minipatch_num, replace=False)
        individual = np.zeros(individual_length, dtype=int)
        individual[patch_indices] = 1

        individual_c = squeeze_individual(individual, patch_num, individual_length)
        # 存入种群
        binary_population[p] = individual_c
        
        # 为每个选择的位置生成随机的RGB值
        rgb_values = np.random.randint(0, 256, size=(minipatch_num, 3), dtype=int)

        # rgb_values = np.random.normal(loc=128, scale=50, size=(minipatch_num, 3))

        # rgb_values = np.clip(rgb_values, 0, 255).astype(np.uint8)
        rgb_values = np.clip(rgb_values, 0, 255).astype(np.float32)

        # step = 255 // (64 - 1) if 64 > 1 else 255
        # rgb_values = np.array([[i * step % 256, 255 - (i * step % 256), (i * step // 2) % 256] 
        #                        for i in range(64)], dtype=int)
        rgb_population[p] = rgb_values

    # 合并种群为一个列表
    population = [binary_population, rgb_population]


    return population


def decode_individual(binary_code, rgb_code, grid_size, tile_size, 
                      minipatch_num, mu, std, target_images, device='cuda'):
    """
    将单个个体（二进制编码 + RGB编码）解码为 mask 和 perturbation。
    
    参数:
        binary_code: 1D numpy 数组，形状 [grid_size^2]，表示格子是否放置补丁
        rgb_code: numpy 数组，形状 [minipatch_num, 3]，每个补丁的 RGB 值 (0-255)
        grid_size: 网格大小 (img_size // tile_size)
        tile_size: 每个小块的像素尺寸
        minipatch_num: 每个个体中 patch 的数量
        mu, std: 数据集归一化参数，形状 [3] 或 [3,1,1]
        target_images: torch.Tensor，用于确定输出大小 (B, C, H, W)
        device: 'cuda' 或 'cpu'
    
    返回:
        mask: torch.Tensor [B, 1, H, W]
        perturbation: torch.Tensor [B, 3, H, W]
    """
    batch_size, _, H, W = target_images.shape

    # --- 构造 mask ---
    mask_array = binary_code.reshape(grid_size, grid_size)
    expanded_mask = np.kron(mask_array, np.ones((tile_size, tile_size), dtype=int))  # 扩展
    mask = torch.tensor(expanded_mask, dtype=torch.float32, device=device)
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B,1,H,W]

    # --- 构造 perturbation ---
    perturbation = torch.zeros_like(target_images, device=device)
    one_indices = np.where(binary_code == 1)[0]

    for p in range(min(minipatch_num, len(one_indices))):
        x, y = divmod(one_indices[p], grid_size)
        patch_rgb = rgb_code[p]
        patch_rgb = torch.tensor(patch_rgb, dtype=torch.float32, device=device) / 255.0

        # 标准化
        mu = mu.to(device)
        std = std.to(device)
        if mu.ndim == 1:
            patch_rgb = (patch_rgb - mu) / std
        else:
            patch_rgb = (patch_rgb.view(3,1,1) - mu) / std

        # 赋值到对应位置
        perturbation[:, :, 
                     x*tile_size:(x+1)*tile_size, 
                     y*tile_size:(y+1)*tile_size] = patch_rgb.view(3, 1, 1)

    return mask, perturbation

def calculate_fitness(model, minipatch_num, clean_images, population, tru_labels, img_size, tile_size, device='cuda', targeted=False):
    """
    批量计算适应度，保留对 population_size 的单循环。
    参数:
        - model: 神经网络模型
        - clean_images: 输入的干净图像，形状 [batch_size, channels, height, width]
        - population: 列表，包含二进制种群和RGB种群：
          - population[0]: numpy.ndarray，二进制种群，形状 [batch_size, population_size, SEQ_SIZE]
          - population[1]: numpy.ndarray，RGB种群，形状 [batch_size, population_size, PATCH_NUM, 3]
        - tru_labels: 每张图片的真实标签，形状 [batch_size]
    返回:
        - fitness: 每个图像的适应度值列表
    """

    global query_count

    binary_population = population[0]
    rgb_population = population[1]

    population_size, seq_size = binary_population.shape

    clean_images = clean_images.to(device)
    tru_labels = tru_labels.to(device)
    perturbation = torch.zeros_like(clean_images).to(device)

    grid_size = img_size // tile_size  # e.g., 224/4=56


    fitness = np.zeros(population_size)  # 存储适应度值

    for i in range(population_size):  # 遍历种群中的每个个体
        mask, perturbation = decode_individual(binary_population[i], rgb_population[i], grid_size, tile_size, 
                          minipatch_num, mu, std, clean_images, device)


        # Create adversarial images for this individual
        adv_images = clean_images * (1 - mask) + perturbation * mask

        # Forward pass for this population individual
        
        out = model(adv_images) 
        query_count += 1
        
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        if targeted:
            loss = - criterion(out, tru_labels)
        else:
            loss = criterion(out, tru_labels)
        fitness[i] = loss.item()  # Store fitness for this individual

    return fitness





def mutation(population_size, population):
    """
    批量处理种群变异操作，确保变异向量形成连通域。
    population: numpy.ndarray，形状为 [batch_size, population_size, SEQ_SIZE]
    """
    binary_population = population[0]
    rgb_population = population[1]
    
    # 初始化变异后的种群
    M_binary_population = np.zeros_like(binary_population)
    M_rgb_population = np.zeros_like(rgb_population)


    for i in range(population_size):
        # 随机选择三个不同的个体 r1, r2, r3
        while True:
            r1, r2, r3 = np.random.choice(population_size, 3, replace=False)
            if r1 != i and r2 != i and r3 != i:
                break

        # 二进制种群变异
        mutant_vector = (binary_population[r1] + 1 * (binary_population[r2] - binary_population[r3])) % 2
        # mutant_vector = ensure_clusters(mutant_vector)  # 修复变异向量（如果需要）
        M_binary_population[i] = mutant_vector

        # # RGB种群变异
        # np.random.shuffle(rgb_population[b, r1])
        # 复制原始数组的部分并打乱
        rgb_copy1 = np.copy(rgb_population[r2])
        rgb_copy2 = np.copy(rgb_population[r3])

        # 打乱复制的数组
        np.random.shuffle(rgb_copy1)
        np.random.shuffle(rgb_copy2)
        mutant_rgb = rgb_population[r1] +  F * (rgb_copy1 - rgb_copy2)
        # 随机生成RGB
        # mutant_rgb = np.random.randint(0, 256, size=(PATCH_NUM, 3), dtype=int)
        mutant_rgb = np.clip(mutant_rgb, 0, 255)  # 确保RGB值在有效范围内
        M_rgb_population[i] = mutant_rgb

    return [M_binary_population, M_rgb_population]


def crossover(population_size, Mpopulation, population, cluster_num, minipatch_num, individual_length):

    binary_population = population[0]
    rgb_population = population[1]
    M_binary_population = Mpopulation[0]
    M_rgb_population = Mpopulation[1]

    # 初始化交叉后的种群
    C_binary_population = np.zeros_like(binary_population)
    C_rgb_population = np.zeros_like(rgb_population)

    

    for i in range(population_size):
        # 随机生成 [0, 1) 的数，决定每个位置是否从变异种群继承
        crossover_mask = np.random.rand(individual_length) < CR
        
        # 确保至少有一个位置来自变异种群，避免完全复制当前个体
        mutated_one_indices = np.where(M_binary_population[i] == 1)[0]  # 变异种群中的 1 的位置
        rand_one_index = np.random.choice(mutated_one_indices)
        crossover_mask[rand_one_index] = True

        # 生成试验向量
        trial_vector = np.where(crossover_mask, M_binary_population[i], binary_population[i])

        # 修复试验向量，使其包含 PATCH_NUM 个 1
        trial_vector = fix_patch_num(trial_vector, minipatch_num)

        # 对试验向量进行白点聚集
        trial_vector = squeeze_individual(trial_vector, cluster_num, individual_length)

        # 存入试验种群（二进制部分）
        C_binary_population[i] = trial_vector

        # 生成试验向量（RGB部分）
        crossover_mask_rgb = np.random.rand(minipatch_num, 3) < CR
        trial_rgb = np.where(crossover_mask_rgb, M_rgb_population[i], rgb_population[i])
        C_rgb_population[i] = trial_rgb

    return [C_binary_population, C_rgb_population]


def fix_patch_num(vector, PATCH_NUM):
    """
    修复向量中的 1 的数量，使其刚好等于 PATCH_NUM。
    """
    one_indices = np.where(vector == 1)[0]
    zero_indices = np.where(vector == 0)[0]

    if len(one_indices) > PATCH_NUM:
        # 若 1 的数量过多，随机保留 PATCH_NUM 个 1
        selected_indices = np.random.choice(one_indices, PATCH_NUM, replace=False)
        vector[:] = 0
        vector[selected_indices] = 1
    elif len(one_indices) < PATCH_NUM:
        # 若 1 的数量不足，从 0 的位置补充到 PATCH_NUM 个 1
        additional_indices = np.random.choice(zero_indices, PATCH_NUM - len(one_indices), replace=False)
        vector[additional_indices] = 1

    return vector




def move_towards_nearby_point(point, target_point, image):
    """
    将一个点向质心的邻域点移动。
    参数:
        - point: 当前点坐标 (x, y)。
        - target_point: 目标邻域点坐标 (tx, ty)。
        - image: 当前图像。
    返回:
        - new_point: 新的位置，如果没有合法位置，返回 None。
    """
    x, y = point
    tx, ty = target_point

    # 计算可能的移动方向
    vertical_direction = (x + 1, y) if tx > x else (x - 1, y) if tx < x else None  # 上下移动
    horizontal_direction = (x, y + 1) if ty > y else (x, y - 1) if ty < y else None  # 左右移动

    # # 随机决定先检查上下方向还是左右方向
    if np.random.rand() < 0.5:  # 50% 概率
        directions = [vertical_direction, horizontal_direction]
    else:
        directions = [horizontal_direction, vertical_direction]


    # directions = [horizontal_direction, vertical_direction]
    # 遍历可能的方向，找到第一个空白位置
    for direction in directions:
        if direction is not None:
            nx, ny = direction
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and image[nx, ny] == 0:
                return (nx, ny)

    return None

def squeeze_individual(individual, num_clusters, individual_length, max_iterations=100, neighborhood_radius=1):
    """
    对一个个体进行聚集操作，将 1 的位置聚集成紧密的簇，通过向随机选定的簇内白点及其邻域中的点移动。
    参数:
        - individual: 输入个体 (1D numpy array)，长度为 SEQ_SIZE。
        - num_clusters: 聚类的数量。
        - individual_length: 编码序列长度，假设为平方数以便转化为二维形状。
        - max_iterations: 最大迭代次数。
        - neighborhood_radius: 选定白点邻域的半径。
    返回:
        - squeezed_individual: 聚集后的个体 (1D numpy array)。
    """
    # 确保 individual_length 是一个完全平方数，便于映射到二维
    grid_size = int(np.sqrt(individual_length))
    assert grid_size ** 2 == individual_length, "individual_length 必须是一个完全平方数！"

    # 转换为二维图像
    image = individual.reshape(grid_size, grid_size)

    # 获取白点的坐标
    white_coords = np.argwhere(image == 1)

    if len(white_coords) == 0:
        return individual  # 如果没有白点，直接返回

    # 聚类：将白点分为 num_clusters 个簇
    kmeans = KMeans(n_clusters=min(num_clusters, len(white_coords)), n_init=20, random_state=42)
    labels = kmeans.fit_predict(white_coords)

    # 对每个簇进行聚集
    for cluster_idx in range(num_clusters):
        cluster_points = white_coords[labels == cluster_idx]

        if len(cluster_points) == 0:
            continue

        # 随机选择一个簇内白点作为目标点
        target_point = cluster_points[np.random.randint(len(cluster_points))]

        # centroid = kmeans.cluster_centers_[cluster_idx]
        # target_point = tuple(np.round(centroid).astype(int))


        # 按离目标点的距离排序
        distances = np.linalg.norm(cluster_points - target_point, axis=1)
        sorted_indices = np.argsort(distances)
        cluster_points = cluster_points[sorted_indices]

        for _ in range(max_iterations):
            has_moved = False
            for point_idx, point in enumerate(cluster_points):
                if np.array_equal(point, target_point):
                    continue
                x, y = point

                # # 获取目标点邻域中的点
                neighborhood = [(cx, cy) for cx in range(target_point[0] - neighborhood_radius, target_point[0] + neighborhood_radius + 1)
                                for cy in range(target_point[1] - neighborhood_radius, target_point[1] + neighborhood_radius + 1)
                                if 0 <= cx < grid_size and 0 <= cy < grid_size]

                # 随机选择一个邻域点作为新目标
                new_target_point = neighborhood[np.random.randint(len(neighborhood))]

                # new_target_point = target_point

                # 尝试向新目标点靠近
                new_point = move_towards_nearby_point(point, new_target_point, image)
                if new_point is not None:
                    image[x, y] = 0  # 清除当前位置
                    image[new_point[0], new_point[1]] = 1  # 移动到新位置
                    cluster_points[point_idx] = new_point  # 更新点位置
                    has_moved = True

            if not has_moved:
                break

    # 将二维图像展平为一维数组
    squeezed_individual = image.flatten()
    return squeezed_individual

def selection(minipatch_num, population_size, model, target_image, Cpopulation, population, pfitness, tru_label, img_size, tile_size, device='cuda'):

    binary_population = population[0]
    rgb_population = population[1]
    C_binary_population = Cpopulation[0]
    C_rgb_population = Cpopulation[1]

    next_binary_population = np.zeros_like(binary_population)
    next_rgb_population = np.zeros_like(rgb_population)
    next_fitness = np.zeros_like(pfitness)


    # 计算 population, Cpopulation 的适应度
    cfitness = calculate_fitness(model, minipatch_num, target_image, Cpopulation, tru_label, img_size, tile_size, device)


    for i in range(population_size):
        # 比较适应度，选择更优的个体
        if cfitness[i] > pfitness[i]:  # 假设适应度越大越好
            next_binary_population[i] = C_binary_population[i]
            next_rgb_population[i] = C_rgb_population[i]
            next_fitness[i] = cfitness[i]
        else:
            next_binary_population[i] = binary_population[i]
            next_rgb_population[i] = rgb_population[i]
            next_fitness[i] = pfitness[i]

    return [next_binary_population, next_rgb_population], next_fitness


def fitness_selection(fitness):
    """
    根据适应度值选择最优个体
    参数:
    - fitness: list，每个个体的适应度值列表，可能包含多个值

    返回:
    - fitness_index: int，最优个体的索引
    - fitness_max_value: float，最优个体的最大适应度值
    """

    # 找到每个批次的最优适应度值索引
    best_indices = np.argmax(fitness)  # 每批次中最大适应度值的索引

    # 根据索引获取每批次的最优适应度值
    best_fitness_values = fitness[best_indices]  # 取出每批次最优适应度值

    return best_indices, best_fitness_values

