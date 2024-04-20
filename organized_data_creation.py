import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import matplotlib.patches as patches

# Path to the folder containing the images
images_folder = "/Users/rolwinpinto/disaster/xview2-toolkit/train/images"
json_folder = "/Users/rolwinpinto/disaster/xview2-toolkit/train/labels"
organized_folder = "/Users/rolwinpinto/disaster/organized_data"

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

# Create the organized folder if it doesn't exist
if not os.path.exists(organized_folder):
    os.makedirs(organized_folder)

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
        
        # Create a folder for the current sample in the organized folder
        sample_folder = os.path.join(organized_folder, post_image_file.split(".")[0])
        os.makedirs(sample_folder, exist_ok=True)

        # Initialize lists to store the severity scores and polygons with damage
        severities_with_damage = []
        polygons_with_damage = []

        # Save post-disaster image
        post_image.save(os.path.join(sample_folder, "post_disaster.png"))

        # Plot post-disaster image with polygons and severity colors
        plt.imshow(post_image)

        # Plot polygons around buildings if coordinates are available
        if building_features and len(building_features) > 0:
            for building in building_features:
                if building:
                    # Extract polygon coordinates
                    polygon_coordinates = building["wkt"].split("POLYGON ((")[1].split("))")[0].split(", ")
                    polygon_coordinates = [tuple(map(float, point.split())) for point in polygon_coordinates]
                    
                    # Get severity from subtype field or default to "unknown"
                    severity = building["properties"].get("subtype", "unknown")
                    
                    # If severity indicates damage, save severity score and polygon
                    if severity != "no-damage":
                        severities_with_damage.append(severity)
                        polygons_with_damage.append(polygon_coordinates)

                        # Assign color based on severity
                        color = severity_colors.get(severity, (1, 1, 1, 1))  # Default to white if severity not found

                        # Plot polygon with assigned color
                        polygon = patches.Polygon(polygon_coordinates, linewidth=0.2, edgecolor='b', facecolor='none')
                        plt.gca().add_patch(polygon)
                        polygon.set_facecolor(color)  # Set facecolor directly based on severity

        plt.axis('off')  # Turn off axis
        plt.savefig(os.path.join(sample_folder, "post_disaster_with_polygons.png"))  # Save the plot
        plt.close()

        # Save severity scores for polygons with damage
        with open(os.path.join(sample_folder, "severity_scores.txt"), "w") as f:
            for severity in severities_with_damage:
                f.write(f"{severity}\n")
        
        # Save polygon coordinates for polygons with damage
        with open(os.path.join(sample_folder, "polygon_coordinates.txt"), "w") as f:
            for polygon_coordinates in polygons_with_damage:
                f.write(f"{polygon_coordinates}\n")
        
        # Save pixel-wise difference image only if there are polygons with damage
        if polygons_with_damage:
            # Load corresponding JSON file for pre-disaster image
            pre_json_file = os.path.join(json_folder, pre_image_file.replace(".png", ".json"))
            if os.path.exists(pre_json_file):
                # Load pre-disaster image
                pre_image = Image.open(pre_image_path)

                # Save pre-disaster image
                pre_image.save(os.path.join(sample_folder, "pre_disaster.png"))

                # Plot pre-disaster image with polygons and severity colors
                plt.imshow(pre_image)

                # Plot polygons around buildings if coordinates are available
                for polygon_coordinates in polygons_with_damage:
                    # Plot polygon with assigned color
                    polygon = patches.Polygon(polygon_coordinates, linewidth=0.2, edgecolor='b', facecolor='none')
                    plt.gca().add_patch(polygon)
                    polygon.set_facecolor(color)  # Set facecolor directly based on severity

                plt.axis('off')  # Turn off axis
                plt.savefig(os.path.join(sample_folder, "pre_disaster_with_polygons.png"))  # Save the plot
                plt.close()

                # Compute pixel-wise difference between post and pre-disaster images
                difference = compute_difference(post_image, pre_image)

                # Save pixel-wise difference image
                difference_image = Image.fromarray(difference)
                difference_image.save(os.path.join(sample_folder, "pixel_difference.png"))

                # Plot heatmap of the pixel-wise difference
                plt.imshow(difference, cmap='hot', vmin=0, vmax=255)  # Adjust vmin and vmax as needed
                plt.title("Pixel-wise difference")
                plt.colorbar()
                plt.axis('off')  # Turn off axis
                plt.savefig(os.path.join(sample_folder, "pixel_difference_heatmap.png"))  # Save the plot
                plt.close()
            else:
                print(f"No JSON file found for image: {pre_image_file}. Skipping...")
    else:
        print(f"No JSON file found for image: {post_image_file}. Skipping...")
