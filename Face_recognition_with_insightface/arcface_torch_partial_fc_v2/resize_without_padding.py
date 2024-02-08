import os
import cv2
pixel_change = 28
def resize_and_save(input_folder, output_folder, target_size):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Recursively list all files in the input folder
    for root, _, file_list in os.walk(input_folder):
        for file_name in file_list:
            # Check if the file is an image (modify the condition based on your file extensions)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                # Construct the full path to the image
                image_path = os.path.join(root, file_name)
                
                # Read the image
                image = cv2.imread(image_path)
                try:
                    height, width,_ = image.shape
                    if (height!=112) and (width!=112):

                        height_start = 0
                        height_end = height
                        width_start = pixel_change
                        width_end = width-pixel_change

                        unpadded_image = image[int(height_start):int(height_end), int(width_start):int(width_end)]

                        # Resize the image
                        resized_image = cv2.resize(unpadded_image, target_size)

                        # Construct the output path in the output folder
                        relative_path = os.path.relpath(image_path, input_folder)
                        output_path = os.path.join(output_folder, relative_path)

                        # Ensure the directory structure exists in the output folder
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        # Save the resized image
                        cv2.imwrite(output_path, resized_image)

                        print(f"Resized and saved: {relative_path}")
                    else:
                        print('already resized',image.shape)
                except Exception as E:
                    print(E)

if __name__ == "__main__":
    # Set your input and output folders
    input_folder = ""#srcfolder
    output_folder = ""#des folder

    # Set the target size for resizing (e.g., (width, height))
    target_size = (112, 112)

    # Call the function to resize and save images
    resize_and_save(input_folder, output_folder, target_size)
