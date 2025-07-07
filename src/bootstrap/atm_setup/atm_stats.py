# Load the .vtp file
import glob
import pyvista as pv
from matplotlib import pyplot as plt
from tqdm import tqdm

base_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_raw/train_data"
all_radius_arr = []
for input_filename in tqdm(glob.glob(f"{base_folder}/*.vtp")):
    graph = pv.read(input_filename)

    # Assume the points are stored in graph.points and the lines in graph.lines.
    # Also assume that the edge attribute "radius" is stored in graph.cell_data["radius"].
    points = graph.points.copy()  # shape: (num_points, 3)
    lines = graph.lines.copy()  # flat connectivity array
    if "radius" not in graph.cell_data:
        raise ValueError("No 'radius' attribute found in the graph cell data.")
    radii = graph.cell_data["radius"]  # should be one value per edge
    all_radius_arr.extend(radii)


def discretize_radii(all_radius_arr):
    def discrete_me(x):
        if x > 5:
            return 1
        elif 3 < x <= 5:
            return 2
        elif 2 < x <= 3:
            return 3
        else:
            return 4
    return [discrete_me(x) for x in all_radius_arr]


plt.hist(all_radius_arr, bins=500, color='blue', edgecolor='black', log=True)
all_radius_arr_discrete = discretize_radii(all_radius_arr)
plt.hist(all_radius_arr_discrete, bins=3, color='red', edgecolor='black', log=True)

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram radii')

# Show the plot
plt.show()

# Now, we check if the graph breaks its validity criterion