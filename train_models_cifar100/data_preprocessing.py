# need to make reachability matrix (row, col) = 1 if row is an ancestor of column
# fine label is from 0 - 99 representing specific class. we will add 20 to each fine label, so
# coarse label is from 0 - 19 representing superclass
# also need to make diluted encoding

import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
DILUTION = 0.7

#read meta data from 
meta_data = unpickle('cifar-100-python/meta')
#create dictionary mapping fine label id to the actual name
fine_label_names = [label.decode('utf-8') for label in meta_data[b'fine_label_names']]
fine_id_to_name = {i + 20: name for i, name in enumerate(fine_label_names)} #offset by 20 to distinguish from coarse labels
#create dictionary mapping coarse label id to actual name
coarse_label_names = [label.decode('utf-8') for label in meta_data[b'coarse_label_names']]
coarse_name_to_id = {name: i for i, name in enumerate(coarse_label_names)}
fine_to_coarse = {
    "beaver": "aquatic_mammals", "dolphin": "aquatic_mammals", "otter": "aquatic_mammals", "seal": "aquatic_mammals", "whale": "aquatic_mammals",
    "aquarium_fish": "fish", "flatfish": "fish", "ray": "fish", "shark": "fish", "trout": "fish",
    "orchid": "flowers", "poppy": "flowers", "rose": "flowers", "sunflower": "flowers", "tulip": "flowers",
    "bottle": "food_containers", "bowl": "food_containers", "can": "food_containers", "cup": "food_containers", "plate": "food_containers",
    "apple": "fruit_and_vegetables", "mushroom": "fruit_and_vegetables", "orange": "fruit_and_vegetables", "pear": "fruit_and_vegetables", "sweet_pepper": "fruit_and_vegetables",
    "clock": "household_electrical_devices", "keyboard": "household_electrical_devices", "lamp": "household_electrical_devices", "telephone": "household_electrical_devices", "television": "household_electrical_devices",
    "bed": "household_furniture", "chair": "household_furniture", "couch": "household_furniture", "table": "household_furniture", "wardrobe": "household_furniture",
    "bee": "insects", "beetle": "insects", "butterfly": "insects", "caterpillar": "insects", "cockroach": "insects",
    "bear": "large_carnivores", "leopard": "large_carnivores", "lion": "large_carnivores", "tiger": "large_carnivores", "wolf": "large_carnivores",
    "bridge": "large_man-made_outdoor_things", "castle": "large_man-made_outdoor_things", "house": "large_man-made_outdoor_things", "road": "large_man-made_outdoor_things", "skyscraper": "large_man-made_outdoor_things",
    "cloud": "large_natural_outdoor_scenes", "forest": "large_natural_outdoor_scenes", "mountain": "large_natural_outdoor_scenes", "plain": "large_natural_outdoor_scenes", "sea": "large_natural_outdoor_scenes",    
    "camel": "large_omnivores_and_herbivores", "cattle": "large_omnivores_and_herbivores", "chimpanzee": "large_omnivores_and_herbivores", "elephant": "large_omnivores_and_herbivores", "kangaroo": "large_omnivores_and_herbivores",
    "fox": "medium_mammals", "porcupine": "medium_mammals", "possum": "medium_mammals", "raccoon": "medium_mammals", "skunk": "medium_mammals",
    "crab": "non-insect_invertebrates", "lobster": "non-insect_invertebrates", "snail": "non-insect_invertebrates", "spider": "non-insect_invertebrates", "worm": "non-insect_invertebrates",
    "baby": "people", "boy": "people", "girl": "people", "man": "people", "woman": "people",
    "crocodile": "reptiles", "dinosaur": "reptiles", "lizard": "reptiles", "snake": "reptiles", "turtle": "reptiles",
    "hamster": "small_mammals", "mouse": "small_mammals", "rabbit": "small_mammals", "shrew": "small_mammals", "squirrel": "small_mammals",
    "maple_tree": "trees", "oak_tree": "trees", "palm_tree": "trees", "pine_tree": "trees", "willow_tree": "trees",
    "bicycle": "vehicles_1", "bus": "vehicles_1", "motorcycle": "vehicles_1", "pickup_truck": "vehicles_1", "train": "vehicles_1",
    "lawn_mower": "vehicles_2", "rocket": "vehicles_2", "streetcar": "vehicles_2", "tank": "vehicles_2", "tractor": "vehicles_2",
}


hierarchy_matrix = np.zeros((120, 120), dtype=int)
diluted_encoding = np.zeros((120, 100), dtype=float) #cols = leaves, rows = all (120, 100)
normal_encoding = np.zeros((120, 100), dtype=float) #cols = leaves, rows = all (120, 100)
new_encoding = np.zeros((120, 100), dtype=float) #cols = leaves, rows = all (120, 100)
np.fill_diagonal(hierarchy_matrix, 1)

for fine_idx in range(20, 120):
    fine_name = fine_id_to_name[fine_idx]
    ancestor_id = coarse_name_to_id[fine_to_coarse[fine_name]]
    hierarchy_matrix[ancestor_id, fine_idx] = 1

for i in range(0, 100):
    fine_idx = i + 20
    fine_name = fine_id_to_name[fine_idx]
    ancestor_id = coarse_name_to_id[fine_to_coarse[fine_name]]
    diluted_encoding[fine_idx, i] = DILUTION
    diluted_encoding[ancestor_id, i] = 1 - DILUTION

coarse_to_fines = {}
for i, fine_name in enumerate(fine_label_names):
    coarse_name = fine_to_coarse[fine_name]
    coarse_id = coarse_name_to_id[coarse_name]
    if coarse_id not in coarse_to_fines:
        coarse_to_fines[coarse_id] = set()
    coarse_to_fines[coarse_id].add(i)

fine_to_siblings = {}
for fine_id in range(100):
    fine_name = fine_label_names[fine_id]
    coarse_name = fine_to_coarse[fine_name]
    coarse_id = coarse_name_to_id[coarse_name]
    fine_to_siblings[fine_id] = coarse_to_fines[coarse_id]
print(f"sum of columns for soft one hot encodings: {diluted_encoding.sum(axis=0)}")
np.save('hierarchy_matrix.npy', hierarchy_matrix)
np.save('diluted_encoding.npy', diluted_encoding)
import pickle
with open('fine_to_siblings.pkl', 'wb') as f:
    pickle.dump(fine_to_siblings, f)


