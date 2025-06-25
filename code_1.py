import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Create synthetic grayscale image
def create_synthetic_image():
    image = np.zeros((100, 100), dtype=np.uint8)

    # Square with intensity 95
    square = np.array([
        [20, 20], [20, 40], [40, 40], [40, 20]
    ], dtype=np.int32)
    cv2.fillPoly(image, [square], color=95)

    # Hexagon with intensity 190
    hexagon = np.array([
        [65, 60], [75, 60], [85, 70],
        [75, 80], [65, 80], [55, 70]
    ], dtype=np.int32)
    cv2.fillPoly(image, [hexagon], color=190)

    return image

# Add Gaussian noise
def add_gaussian_noise(image, mean=0, std=20):
    noise = np.random.normal(mean, std, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Otsu's Thresholding
def apply_otsu(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, otsu_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_img

# Region Growing
def region_growing(img, seed, threshold=20):
    height, width = img.shape
    segmented = np.zeros_like(img, dtype=np.uint8)
    visited = np.zeros_like(img, dtype=bool)

    stack = [seed]
    seed_val = img[seed]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-neighborhood

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True

        if abs(int(img[x, y]) - int(seed_val)) <= threshold:
            segmented[x, y] = 255
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                    stack.append((nx, ny))
    return segmented

# Create output directory
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Step-by-step image processing
image = create_synthetic_image()
cv2.imwrite(os.path.join(output_dir, "1_original.png"), image)

noisy_image = add_gaussian_noise(image, std=20)
cv2.imwrite(os.path.join(output_dir, "2_noisy.png"), noisy_image)

otsu_result = apply_otsu(noisy_image)
cv2.imwrite(os.path.join(output_dir, "3_otsu_threshold.png"), otsu_result)

smoothed = cv2.GaussianBlur(noisy_image, (3, 3), 0)
cv2.imwrite(os.path.join(output_dir, "4_smoothed.png"), smoothed)

# Choose seed point for region growing
seed_point = (70, 70)  # Inside Hexagon
# seed_point = (30, 30)  # Inside Square
# seed_point = (90, 90)  # Background

region_result = region_growing(smoothed, seed_point, threshold=25)
cv2.imwrite(os.path.join(output_dir, "5_region_grown.png"), region_result)

# Save the figure to the folder 
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Otsu Thresholding")
plt.imshow(otsu_result, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Smoothed for Region Growing")
plt.imshow(smoothed, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Region Grown")
plt.imshow(region_result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "6_combined_plot.png"))
plt.close()  # 

print("All images saved successfully to 'output_images/' folder.")
