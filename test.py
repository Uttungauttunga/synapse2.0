import os
import numpy as np
from PIL import Image
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the trained model
model = tf.keras.models.load_model('building_detection_model.h5')

# Define severity colors
severity_colors = {
    0: 'green',    # Example: No damage
    1: 'yellow',   # Example: Minor damage
    2: 'orange',   # Example: Major damage
    3: 'red'       # Example: Destroyed
}

# Preprocess new pre and post-disaster images and polygon coordinates
def preprocess_new_data(post_image_path, pre_image_path, polygon_coordinates_list):
    post_images = []
    pre_images = []
    pixel_differences = []
    new_polygon_coordinates = []  # Rename to avoid conflict

    for polygon_coordinates in polygon_coordinates_list:
        post_image = load_images(post_image_path)
        pre_image = load_images(pre_image_path)
        pixel_difference_img = compute_difference(post_image, pre_image, polygon_coordinates)

        # Append pre and post-disaster images and pixel differences
        post_images.append(post_image)
        pre_images.append(pre_image)
        pixel_differences.append(pixel_difference_img)

        # Append polygon coordinates to the new list
        new_polygon_coordinates.append(polygon_coordinates)

    return np.array(post_images), np.array(pre_images), np.array(pixel_differences), new_polygon_coordinates

# Load images
def load_images(image_path):
    img = Image.open(image_path).resize((224, 224))  # Resize if needed
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Compute pixel-wise difference for the specific polygon region
# Compute pixel-wise difference for the specific polygon region
def compute_difference(image1, image2, polygon_coordinates):
    # Create a mask for the polygon region
    mask = np.zeros_like(image1[:, :, 0])
    rr, cc = zip(*polygon_coordinates)  # Unzip the coordinates
    rr = np.clip(rr, 0, mask.shape[0] - 1).astype(int)
    cc = np.clip(cc, 0, mask.shape[1] - 1).astype(int)
    mask[rr, cc] = 1

    # Apply the mask to the images
    masked_image1 = image1 * mask[:, :, np.newaxis]
    masked_image2 = image2 * mask[:, :, np.newaxis]

    # Compute pixel-wise difference
    return np.abs(masked_image1 + masked_image2)


# Example post and pre-disaster image paths
post_image_path = "images/guatemala-volcano_00000001_post_disaster.png"
pre_image_path = "images/guatemala-volcano_00000001_post_disaster.png"

# Example polygon coordinates for multiple buildings (from JSON file)
json_file_path = "labels/hurricane-harvey_00000501_post_disaster.json"
with open(json_file_path, "r") as json_file:
    polygon_data = json.load(json_file)
    polygon_coordinates_list = []

    # Iterate through features and extract coordinates
    for feature in polygon_data["features"]["lng_lat"]:
        coordinates_str = feature["wkt"].split("((")[1].split(")")[0]  # Extract everything between "((" and ")"
        coordinates_list = coordinates_str.split(", ")
        polygon_coordinates_list.append([tuple(map(float, coord.split(" "))) for coord in coordinates_list])

# Preprocess new data
post_images, pre_images, pixel_differences, polygon_coordinates_list = preprocess_new_data(post_image_path,
                                                                                           pre_image_path,
                                                                                           polygon_coordinates_list)

# Perform inference for each building instance
predicted_labels_list = []
for post_image, pre_image, pixel_difference_img, polygon_coordinates in zip(post_images, pre_images,
                                                                            pixel_differences,
                                                                            polygon_coordinates_list):
    # Expand dimensions for single instance
    post_image = np.expand_dims(post_image, axis=0)
    pre_image = np.expand_dims(pre_image, axis=0)
    pixel_difference_img = np.expand_dims(pixel_difference_img, axis=0)

    # Perform inference
    predictions = model.predict([post_image, pre_image, pixel_difference_img])

    # Decode predictions to severity labels
    predicted_label = np.argmax(predictions, axis=1)[0]
    predicted_labels_list.append(predicted_label)

    # Print the predicted label for debugging
    print("Predicted Label:", predicted_label)

# Plot post-disaster image
plt.imshow(post_images[0])
ax = plt.gca()

# Draw polygon patches for each building instance
for polygon_coordinates, predicted_label in zip(polygon_coordinates_list, predicted_labels_list):
    polygon = patches.Polygon(polygon_coordinates, linewidth=2, edgecolor=severity_colors[predicted_label],
                              facecolor='none')
    ax.add_patch(polygon)

# Show the plot
plt.axis('off')
plt.show()
