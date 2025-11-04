import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn


# ImageNet normalization constants (mean and std per channel)
mu = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# mu = torch.tensor([0.5071, 0.4867, 0.4408]).cuda()
# std = torch.tensor([0.2675, 0.2565, 0.2761]).cuda()

# mu = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
# std = torch.tensor([0.2470, 0.2435, 0.2616]).cuda()

###########################################
# Evolution strategy / DE hyperparameters
F = 2
CR = 0.8
###########################################


query_count = 0


def init_population(population_size, patch_num, minipatch_num, img_size, tile_size):
    """
    Initialize the DE population.

    Inputs:
        - population_size: number of individuals in the population
        - patch_num: number of patch clusters to maintain when squeezing
        - minipatch_num: number of minipatches (selected active tiles) per individual
        - img_size: full image size in pixels
        - tile_size: size of each tile (pixel side length)

    Returns:
        - population: [binary_population, rgb_population]
          - binary_population: numpy array shape (population_size, SEQ_SIZE) with 0/1 mask per tile
          - rgb_population: numpy array shape (population_size, minipatch_num, 3) with initial RGB values
    """
    individual_length = (img_size // tile_size) ** 2  # sequence length (number of tiles)

    # Binary population (tile on/off) initialized to zeros
    binary_population = np.zeros((population_size, individual_length), dtype=int)
    # RGB population per selected minipatch (initialized randomly)
    rgb_population = np.zeros((population_size, minipatch_num, 3), dtype=int)

    for p in range(population_size):
        # Randomly choose minipatch_num tile indices to set to 1
        patch_indices = np.random.choice(individual_length, minipatch_num, replace=False)
        individual = np.zeros(individual_length, dtype=int)
        individual[patch_indices] = 1

        # Optionally squeeze/cluster the selected tiles
        individual_c = squeeze_individual(individual, patch_num, individual_length)
        binary_population[p] = individual_c

        # Initialize RGB values for selected tiles (integers 0..255)
        rgb_values = np.random.randint(0, 256, size=(minipatch_num, 3), dtype=int)
        rgb_values = np.clip(rgb_values, 0, 255).astype(np.float32)
        rgb_population[p] = rgb_values

    population = [binary_population, rgb_population]

    return population


def decode_individual(binary_code, rgb_code, grid_size, tile_size, 
                      minipatch_num, mu, std, target_images, device='cuda'):
    """
    Decode a single individual (binary mask + RGB codes) into a mask tensor and a perturbation tensor.

    Parameters:
        binary_code: 1D numpy array of length grid_size^2 indicating which tiles are active (1)
        rgb_code: numpy array of shape (minipatch_num, 3) with RGB values in [0,255]
        grid_size: number of tiles per side (img_size // tile_size)
        tile_size: size of each tile in pixels
        minipatch_num: number of patches represented by rgb_code
        mu, std: normalization tensors (shape [3] or [3,1,1]) used to normalize RGB values
        target_images: torch.Tensor used to determine batch size and image shape (B, C, H, W)
        device: 'cuda' or 'cpu'

    Returns:
        mask: torch.Tensor shaped [B, 1, H, W] with binary mask (float)
        perturbation: torch.Tensor shaped [B, 3, H, W] with normalized RGB values applied to active tiles
    """
    batch_size, _, H, W = target_images.shape

    # Build 2D mask from binary code then expand each tile to tile_size x tile_size
    mask_array = binary_code.reshape(grid_size, grid_size)
    expanded_mask = np.kron(mask_array, np.ones((tile_size, tile_size), dtype=int))
    mask = torch.tensor(expanded_mask, dtype=torch.float32, device=device)
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, 1, H, W]

    # Build perturbation: set selected tiles to the corresponding RGB values (normalized)
    perturbation = torch.zeros_like(target_images, device=device)
    one_indices = np.where(binary_code == 1)[0]

    for p in range(min(minipatch_num, len(one_indices))):
        x, y = divmod(one_indices[p], grid_size)
        patch_rgb = rgb_code[p]
        patch_rgb = torch.tensor(patch_rgb, dtype=torch.float32, device=device) / 255.0

        # Normalize by dataset mean/std
        mu = mu.to(device)
        std = std.to(device)
        if mu.ndim == 1:
            patch_rgb = (patch_rgb - mu) / std
        else:
            patch_rgb = (patch_rgb.view(3,1,1) - mu) / std

        # Assign the normalized RGB to the tile region
        perturbation[:, :, 
                     x*tile_size:(x+1)*tile_size, 
                     y*tile_size:(y+1)*tile_size] = patch_rgb.view(3, 1, 1)

    return mask, perturbation

def calculate_fitness(model, minipatch_num, clean_images, population, tru_labels, img_size, tile_size, device='cuda', targeted=False):
    """
    Compute fitness values for each individual in the population.

    This function evaluates each individual by decoding it into an adversarial image, forwarding it
    through the model, and computing the cross-entropy loss (or negative loss for targeted attacks).

    Arguments:
        model: neural network model
        clean_images: input images tensor [B, C, H, W]
        population: [binary_population, rgb_population]
        tru_labels: true labels tensor [B]
        img_size, tile_size: used to reconstruct tile grid
        targeted: if True, use negative loss to encourage target class

    Returns:
        fitness: numpy array with a scalar fitness value per individual
    """
    global query_count

    binary_population = population[0]
    rgb_population = population[1]

    population_size, seq_size = binary_population.shape

    clean_images = clean_images.to(device)
    tru_labels = tru_labels.to(device)

    grid_size = img_size // tile_size

    fitness = np.zeros(population_size)

    for i in range(population_size):
        mask, perturbation = decode_individual(binary_population[i], rgb_population[i], grid_size, tile_size, 
                          minipatch_num, mu, std, clean_images, device)

        # Create adversarial images for this individual
        adv_images = clean_images * (1 - mask) + perturbation * mask

        # Forward pass and compute loss
        out = model(adv_images)
        query_count += 1

        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        if targeted:
            loss = - criterion(out, tru_labels)
        else:
            loss = criterion(out, tru_labels)
        fitness[i] = loss.item()

    return fitness


def mutation(population_size, population):
    """
    Perform DE mutation for the binary and RGB parts of the population.

    The binary mutation is implemented modulo 2 to keep values in {0,1}.
    The RGB mutation uses integer arithmetic and clipping to remain in [0,255].
    """
    binary_population = population[0]
    rgb_population = population[1]
    
    M_binary_population = np.zeros_like(binary_population)
    M_rgb_population = np.zeros_like(rgb_population)

    for i in range(population_size):
        # Randomly select three distinct individuals r1, r2, r3 different from i
        while True:
            r1, r2, r3 = np.random.choice(population_size, 3, replace=False)
            if r1 != i and r2 != i and r3 != i:
                break

        # Binary mutation (mod 2 arithmetic)
        mutant_vector = (binary_population[r1] + 1 * (binary_population[r2] - binary_population[r3])) % 2
        M_binary_population[i] = mutant_vector

        # RGB mutation: shuffle copies and apply differential update, then clip
        rgb_copy1 = np.copy(rgb_population[r2])
        rgb_copy2 = np.copy(rgb_population[r3])
        np.random.shuffle(rgb_copy1)
        np.random.shuffle(rgb_copy2)
        mutant_rgb = rgb_population[r1] +  F * (rgb_copy1 - rgb_copy2)
        mutant_rgb = np.clip(mutant_rgb, 0, 255)
        M_rgb_population[i] = mutant_rgb

    return [M_binary_population, M_rgb_population]


def crossover(population_size, Mpopulation, population, cluster_num, minipatch_num, individual_length):
    """
    Perform crossover between current population and mutated population to produce trial individuals.

    - For binary masks, a per-position Bernoulli mask with rate CR is used.
    - For RGB values, a per-channel mask is applied.
    After crossover, trial binary vectors are fixed to have exactly minipatch_num ones and squeezed/clusted.
    """
    binary_population = population[0]
    rgb_population = population[1]
    M_binary_population = Mpopulation[0]
    M_rgb_population = Mpopulation[1]

    C_binary_population = np.zeros_like(binary_population)
    C_rgb_population = np.zeros_like(rgb_population)

    for i in range(population_size):
        crossover_mask = np.random.rand(individual_length) < CR

        # Ensure at least one position comes from the mutant (avoid exact copy)
        mutated_one_indices = np.where(M_binary_population[i] == 1)[0]
        rand_one_index = np.random.choice(mutated_one_indices)
        crossover_mask[rand_one_index] = True

        trial_vector = np.where(crossover_mask, M_binary_population[i], binary_population[i])

        # Fix the number of active tiles and cluster them
        trial_vector = fix_patch_num(trial_vector, minipatch_num)
        trial_vector = squeeze_individual(trial_vector, cluster_num, individual_length)
        C_binary_population[i] = trial_vector

        # RGB crossover
        crossover_mask_rgb = np.random.rand(minipatch_num, 3) < CR
        trial_rgb = np.where(crossover_mask_rgb, M_rgb_population[i], rgb_population[i])
        C_rgb_population[i] = trial_rgb

    return [C_binary_population, C_rgb_population]


def fix_patch_num(vector, PATCH_NUM):
    """
    Fix the number of ones in a binary vector so that it equals PATCH_NUM.
    If there are too many ones, randomly keep PATCH_NUM of them.
    If there are too few ones, randomly turn some zeros into ones.
    """
    one_indices = np.where(vector == 1)[0]
    zero_indices = np.where(vector == 0)[0]

    if len(one_indices) > PATCH_NUM:
        selected_indices = np.random.choice(one_indices, PATCH_NUM, replace=False)
        vector[:] = 0
        vector[selected_indices] = 1
    elif len(one_indices) < PATCH_NUM:
        additional_indices = np.random.choice(zero_indices, PATCH_NUM - len(one_indices), replace=False)
        vector[additional_indices] = 1

    return vector


def move_towards_nearby_point(point, target_point, image):
    """
    Move a point one step towards a nearby target point if an empty spot exists.

    Parameters:
        - point: current coordinates (x, y)
        - target_point: target coordinates (tx, ty)
        - image: 2D grid representing occupancy (1 = occupied, 0 = empty)

    Returns:
        - new_point tuple or None if no valid move exists
    """
    x, y = point
    tx, ty = target_point

    # Determine vertical/horizontal directions toward target
    vertical_direction = (x + 1, y) if tx > x else (x - 1, y) if tx < x else None
    horizontal_direction = (x, y + 1) if ty > y else (x, y - 1) if ty < y else None

    # Randomly choose which direction to try first
    if np.random.rand() < 0.5:
        directions = [vertical_direction, horizontal_direction]
    else:
        directions = [horizontal_direction, vertical_direction]

    for direction in directions:
        if direction is not None:
            nx, ny = direction
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and image[nx, ny] == 0:
                return (nx, ny)

    return None

def squeeze_individual(individual, num_clusters, individual_length, max_iterations=100, neighborhood_radius=1):
    """
    Cluster and compact active tiles (1s) in an individual to form tighter groups.

    The function maps the 1D mask into a 2D grid, applies KMeans clustering to the coordinates
    of active tiles, and attempts to move points toward randomly selected neighbors inside
    each cluster to form compact shapes.
    """
    grid_size = int(np.sqrt(individual_length))
    assert grid_size ** 2 == individual_length, "individual_length must be a perfect square"

    image = individual.reshape(grid_size, grid_size)
    white_coords = np.argwhere(image == 1)

    if len(white_coords) == 0:
        return individual

    kmeans = KMeans(n_clusters=min(num_clusters, len(white_coords)), n_init=20, random_state=42)
    labels = kmeans.fit_predict(white_coords)

    for cluster_idx in range(num_clusters):
        cluster_points = white_coords[labels == cluster_idx]
        if len(cluster_points) == 0:
            continue

        # Randomly pick a point inside the cluster as a (local) target
        target_point = cluster_points[np.random.randint(len(cluster_points))]

        distances = np.linalg.norm(cluster_points - target_point, axis=1)
        sorted_indices = np.argsort(distances)
        cluster_points = cluster_points[sorted_indices]

        for _ in range(max_iterations):
            has_moved = False
            for point_idx, point in enumerate(cluster_points):
                if np.array_equal(point, target_point):
                    continue
                x, y = point

                neighborhood = [(cx, cy) for cx in range(target_point[0] - neighborhood_radius, target_point[0] + neighborhood_radius + 1)
                                for cy in range(target_point[1] - neighborhood_radius, target_point[1] + neighborhood_radius + 1)
                                if 0 <= cx < grid_size and 0 <= cy < grid_size]

                new_target_point = neighborhood[np.random.randint(len(neighborhood))]
                new_point = move_towards_nearby_point(point, new_target_point, image)
                if new_point is not None:
                    image[x, y] = 0
                    image[new_point[0], new_point[1]] = 1
                    cluster_points[point_idx] = new_point
                    has_moved = True

            if not has_moved:
                break

    squeezed_individual = image.flatten()
    return squeezed_individual

def selection(minipatch_num, population_size, model, target_image, Cpopulation, population, pfitness, tru_label, img_size, tile_size, device='cuda'):
    """
    Selection step: compare current population with trial population and keep the better individuals.

    Returns the next population and the corresponding fitness values.
    """
    binary_population = population[0]
    rgb_population = population[1]
    C_binary_population = Cpopulation[0]
    C_rgb_population = Cpopulation[1]

    next_binary_population = np.zeros_like(binary_population)
    next_rgb_population = np.zeros_like(rgb_population)
    next_fitness = np.zeros_like(pfitness)

    # Evaluate fitness of trial (crossover) population
    cfitness = calculate_fitness(model, minipatch_num, target_image, Cpopulation, tru_label, img_size, tile_size, device)

    for i in range(population_size):
        # Choose the individual with higher fitness
        if cfitness[i] > pfitness[i]:
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
    Select the index and value of the best fitness from a fitness array.

    Returns:
        - best_indices: index of the best individual
        - best_fitness_values: the best fitness value
    """
    best_indices = np.argmax(fitness)
    best_fitness_values = fitness[best_indices]

    return best_indices, best_fitness_values
