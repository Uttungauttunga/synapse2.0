import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import matplotlib.patches as patches

# Path to the folder containing the images
images_folder = "images"

json_folder = "labels"

# Define a color map for severity levels
severity_colors = {
    "no-damage": (0, 1, 0, 1),      # Green for no damage
    "minor-damage": (1, 1, 0, 1),    # Yellow for minor damage
    "major-damage": (1, 0, 0, 1),    # Red for major damage
    "destroyed": (0, 0, 0, 1)        # Black for destroyed
}

# Function to compute pixel-wise difference between two images
def compute_difference(image1, image2):
    return np.abs(np.array(image1) - np.array(image2))

# Get a sorted list of image files
image_files = sorted([file for file in os.listdir(images_folder) if file.endswith(".png")])

# Iterate over every other image in the folder
for i in range(0, len(image_files), 2):
    post_image_file = image_files[i]
    pre_image_file = image_files[i + 1]
    
    # Load post-disaster image
    post_image_path = os.path.join(images_folder, post_image_file)
    post_image = Image.open(post_image_path)

    # Load pre-disaster image
    pre_image_path = os.path.join(images_folder, pre_image_file)
    pre_image = Image.open(pre_image_path)
    
    # Load corresponding JSON file for post-disaster image
    post_json_file = os.path.join(json_folder, post_image_file.replace(".png", ".json"))

    if os.path.exists(post_json_file):
        with open(post_json_file, "r") as f:
            data = json.load(f)

        # Extract building coordinates and severity from JSON if available
        building_features = data.get("features", {}).get("xy", [])

        print(f"Building features for {post_image_file}: {building_features}")
        
        # Plot post-disaster image
        plt.imshow(post_image)

        # Plot polygons around buildings if coordinates are available
        if building_features and len(building_features) > 0:
            for building in building_features:
                # Check if building features are empty
                if building:
                    # Extract polygon coordinates
                    polygon_coordinates = building["wkt"].split("POLYGON ((")[1].split("))")[0].split(", ")
                    polygon_coordinates = [tuple(map(float, point.split())) for point in polygon_coordinates]
                    polygon = patches.Polygon(polygon_coordinates, linewidth=0.2, edgecolor='b', facecolor='none')

                    # Get severity from subtype field or default to "unknown"
                    severity = building["properties"].get("subtype", "unknown")
                    
                    # Assign color based on severity
                    color = severity_colors.get(severity, (1, 1, 1, 1))  # Default to white if severity not found

                    # Plot polygon with assigned color
                    plt.gca().add_patch(polygon)
                    polygon.set_facecolor(color)  # Set facecolor directly based on severity

        plt.axis('off')  # Turn off axis
        plt.show()
    else:
        print(f"No JSON file found for image: {post_image_file}. Skipping...")
    
    # Load corresponding JSON file for pre-disaster image
    pre_json_file = os.path.join(json_folder, pre_image_file.replace(".png", ".json"))

    if os.path.exists(pre_json_file):
        with open(pre_json_file, "r") as f:
            data = json.load(f)

        # Extract building coordinates and severity from JSON if available
        building_features = data.get("features", {}).get("xy", [])

        print(f"Building features for {pre_image_file}: {building_features}")
        
        # Plot pre-disaster image
        plt.imshow(pre_image)

        # Plot polygons around buildings if coordinates are available
        if building_features and len(building_features) > 0:
            for building in building_features:
                # Check if building features are empty
                if building:
                    # Extract polygon coordinates
                    polygon_coordinates = building["wkt"].split("POLYGON ((")[1].split("))")[0].split(", ")
                    polygon_coordinates = [tuple(map(float, point.split())) for point in polygon_coordinates]
                    polygon = patches.Polygon(polygon_coordinates, linewidth=0.2, edgecolor='b', facecolor='none')

                    # Get severity from subtype field or default to "unknown"
                    severity = building["properties"].get("subtype", "unknown")
                    
                    # Assign color based on severity
                    color = severity_colors.get(severity, (1, 1, 1, 1))  # Default to white if severity not found

                    # Plot polygon with assigned color
                    plt.gca().add_patch(polygon)
                    polygon.set_facecolor(color)  # Set facecolor directly based on severity

        plt.axis('off')  # Turn off axis
        plt.show()
    else:
        print(f"No JSON file found for image: {pre_image_file}. Skipping...")

    # Check if images have the same dimensions
    if post_image.size == pre_image.size:
        # Compute pixel-wise difference between post and pre-disaster images
        difference = compute_difference(post_image, pre_image)

        # Plot heatmap of the pixel-wise difference
        plt.imshow(difference, cmap='hot', vmin=0, vmax=255)  # Adjust vmin and vmax as needed
        plt.title(f"Pixel-wise difference for {post_image_file} and {pre_image_file}")
        plt.colorbar()
        plt.axis('off')  # Turn off axis
        plt.show()
    else:
        print(f"Images {post_image_file} and {pre_image_file} have different dimensions. Skipping...")
