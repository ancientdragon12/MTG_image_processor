import os
import cv2
from image_processor import image_process

def batch_image_process(input_folder, output_folder, starting_threshold=75):
    threshold = starting_threshold

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through input images
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg','.png')):  # Process only image files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"processed_{filename}")

            # Reset threshold for each image
            current_threshold = threshold
            
            while current_threshold >= 5:
                # Read the image
                image = cv2.imread(input_path)
                
                # Process the image using the function from the external file
                processed_image = image_process(image, threshold=current_threshold, verbose=False)
                
                if processed_image is not None:
                    # Save the processed image
                    cv2.imwrite(output_path, processed_image)
                    print(f"Processed: {filename} -> Saved as: {output_path}")
                    break  # Exit the loop if successful
                
                # Reduce threshold by 5 for the next attempt
                current_threshold -= 5

            if current_threshold < 5:
                print(f"Could not process {filename} with any threshold from {starting_threshold} to 5.")

if __name__ == "__main__":
    input_images_folder = "input_images"
    result_images_folder = "result_images"

    batch_image_process(input_images_folder, result_images_folder)

