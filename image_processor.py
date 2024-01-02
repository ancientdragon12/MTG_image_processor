
"""
File: image_processor.py
Author: Ancientdragon12
Date: 1-2-2024
Description: Image Processing pipeline
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

def image_process(input_image, threshold = 75, verbose = False):
    # Convert the input image to grayscale
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # Set a threshold value (you can adjust this value as needed)
    threshold_value = threshold
    
    # Apply thresholding to isolate dark pixels
    _, thresholded_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Invert the thresholded image
    inverted_thresholded_img = cv2.bitwise_not(thresholded_img)
    
    # Find contours
    contours, _ = cv2.findContours(inverted_thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the largest contour
    mask = np.zeros_like(inverted_thresholded_img)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)  # Draw the largest contour filled in the mask
    
    # Draw the largest contour on a copy of the original image
    original_with_contour = input_image.copy()
    cv2.drawContours(original_with_contour, [largest_contour], -1, (0, 255, 0), 2)  # Draw the largest contour
    
    # Approximate the largest contour to get a polygon (assuming it's a rectangle)
    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx_corners) == 4:
        # Draw red lines connecting the corners on a copy of the original image
        corners_image = input_image.copy()
        corners = approx_corners.reshape(-1, 2)
        for i in range(4):
            cv2.line(corners_image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 0, 255), 2)
        
        # Get the destination points for perspective transformation (image edges)
        card_width = max(np.linalg.norm(corners[0] - corners[1]), np.linalg.norm(corners[2] - corners[3]))
        card_height = max(np.linalg.norm(corners[1] - corners[2]), np.linalg.norm(corners[3] - corners[0]))
        dst_points = np.array([[0, 0], [card_width, 0], [card_width, card_height], [0, card_height]], dtype=np.float32)
        
        # Calculate the perspective transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
        
        # Apply the perspective transformation to the original image
        transformed_image = cv2.warpPerspective(input_image, transformation_matrix, (int(card_width), int(card_height)))
        
        # Flip the transformed image along the y-axis
        transformed_image = cv2.flip(transformed_image, 1)

        # Resize the transformed image
        final_image = cv2.resize(transformed_image, (480, 680))  # Specify new width and height

        # Save the transformed image as final_img.jpg
        cv2.imwrite('final_img.jpg', final_image)
        
        if verbose:
            # Plotting the images
            plt.figure(figsize=(20, 6))  # Adjust figure size as needed
            
            # Plot original image
            plt.subplot(1, 6, 1)
            plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            # Plot grayscale image
            plt.subplot(1, 6, 2)
            plt.imshow(gray_img, cmap='gray')
            plt.title('Grayscale Image')
            plt.axis('off')
            
            # Plot inverted thresholded image
            plt.subplot(1, 6, 3)
            plt.imshow(cv2.bitwise_and(inverted_thresholded_img, inverted_thresholded_img, mask=mask), cmap='gray')
            plt.title('Inverted Thresholded Image')
            plt.axis('off')
            
            # Plot original image with largest contour
            plt.subplot(1, 6, 4)
            plt.imshow(cv2.cvtColor(original_with_contour, cv2.COLOR_BGR2RGB))
            plt.title('Largest Contour')
            plt.axis('off')
            
            # Plot original image with red lines connecting corners
            plt.subplot(1, 6, 5)
            plt.imshow(cv2.cvtColor(corners_image, cv2.COLOR_BGR2RGB))
            plt.title('Corners Detected')
            plt.axis('off')
            
            # Plot transformed image (perspective corrected)
            plt.subplot(1, 6, 6)
            plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
            plt.title('Transformed Image')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        return final_image
    else:
        print("Could not find a valid rectangle contour.")

        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py input_image_filename")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    image = cv2.imread(input_filename)
    
    if image is None:
        print("Error: Could not read the image.")
        sys.exit(1)
    
    verbose_mode = "--verbose" in sys.argv

    processed_image = image_process(input_image=image, verbose=verbose_mode)
