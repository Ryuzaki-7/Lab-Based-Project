import cv2
import os

# Path to the input image
# input_path = 'image1.jpg'

# Directory to save the augmented images
output_dir = 'output/'
folder_path='new_data'
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path,filename)
    img = cv2.imread(image_path)
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(image_path)

    # Extract the filename from the image path
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Rotate the image by 90, 270, and 360 degrees and flip vertically
    for angle in [0, 90, 180, 270]:
        # Rotate the image
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))

        # Flip the rotated image vertically
        flipped = cv2.flip(rotated, 0)

        # Save the rotated and flipped image
        output_path = os.path.join(output_dir, f'{basename}_rotated_{angle}.jpg')
        output_path2 = os.path.join(output_dir, f'{basename}_rotated_{angle}_vflip.jpg')
        cv2.imwrite(output_path, rotated)
        cv2.imwrite(output_path2, flipped)
