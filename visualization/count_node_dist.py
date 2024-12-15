import torch
from collections import deque, Counter

# Load dataset
file_path = "../data/politifact/processed/pol_data_spacy.pt"
data, slices = torch.load(file_path)


distance_counts = Counter()
distance_counts_true = Counter()
distance_counts_fake = Counter()

# Iterate through all graphs
for i in range(len(slices['x']) - 1):
    start, end = slices['x'][i], slices['x'][i + 1]
    edge_start, edge_end = slices['edge_index'][i], slices['edge_index'][i + 1]
    x = data.x[start:end]
    edge_index = data.edge_index[:, edge_start:edge_end]

    label = data.y[i]   # 0 = true, 1 = fake
    
    # BFS to calculate distances from the news node 
    num_nodes = x.size(0)
    distances = [float('inf')] * num_nodes
    distances[0] = 0 
    queue = deque([0])
    
    while queue:
        current = queue.popleft()
        for neighbor in edge_index[1, edge_index[0] == current]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    
    distance_counts.update(Counter(distances))

    if label == 1:  # Fake
        distance_counts_fake.update(Counter(distances))
    else:  # True
        distance_counts_true.update(Counter(distances))


def summarize_distances(distance_counts):
    total_nodes = sum(distance_counts.values())
    proportions = {dist: count / total_nodes for dist, count in distance_counts.items()}

    bins = {"1-hop": 0, "2-hop": 0, "3-hop": 0, "4-hop": 0, "5-hop": 0, ">6-hop": 0}
    for dist, proportion in proportions.items():
        if dist == 1:
            bins["1-hop"] += proportion
        elif dist == 2:
            bins["2-hop"] += proportion
        elif dist == 3:
            bins["3-hop"] += proportion
        elif dist == 4:
            bins["4-hop"] += proportion
        elif dist == 5: 
            bins["5-hop"] += proportion
        elif dist >= 6:
            bins[">6-hop"] += proportion

    return bins

# Summarize distances for true and false labels
proportions = summarize_distances(distance_counts)
proportions_true = summarize_distances(distance_counts_true)
proportions_false = summarize_distances(distance_counts_fake)

print("Proportion of nodes by hop distance:")
for bin_name, proportion in proportions.items():
    print(f"{bin_name}: {proportion:.4f}")

print("\nProportion of nodes by hop distance (True News):")
for bin_name, proportion in proportions_true.items():
    print(f"{bin_name}: {proportion:.4f}")

print("\nProportion of nodes by hop distance (False News):")
for bin_name, proportion in proportions_false.items():
    print(f"{bin_name}: {proportion:.4f}")


# All graphs
# 1-hop: 0.5219
# 2-hop: 0.3404
# 3-hop: 0.0773
# 4-hop: 0.0326
# 5-hop: 0.0132
# >6-hop: 0.0071

# True news
# 1-hop: 0.5168
# 2-hop: 0.3405
# 3-hop: 0.0734
# 4-hop: 0.0357
# 5-hop: 0.0155
# >6-hop: 0.0093

# Fake news
# 1-hop: 0.5258
# 2-hop: 0.3402
# 3-hop: 0.0804
# 4-hop: 0.0302
# 5-hop: 0.0114
# >6-hop: 0.0055