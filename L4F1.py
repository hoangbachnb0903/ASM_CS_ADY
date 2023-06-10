import cv2
import numpy as np
from skimage import io
from function1 import snakes
from function2 import watershed_segmentation
from function3 import kmeans_segmentation
from function4 import mean_shift_segmentation

# Import image
I='A.jpg'
image = io.imread(I, as_gray=True)
img = cv2.imread(I)
# Menu
while True:
    # Display options
    print("\nChoose an option:")
    print("1. Snakes algorithm for active contours")
    print("2. Watershed algorithm for image segmentation")
    print("3. K-means for segmentation")
    print("4. Mean shift algorithm for segmentation")
    print("5. Exit")

    # Get user input
    choice = input("Enter your choice (1-5): ")

    if choice == "1":
        # Snakes algorithm
        if 'image' not in locals():
            print("Error: Image not loaded")
        else:
            snakes(image)
    elif choice == "2":
        # Watershed algorithm
        if 'image' not in locals():
            print("Error: Image not loaded")
        else:
            watershed_segmentation(img)
    elif choice == "3":
        # K-means algorithm
        if 'image' not in locals():
            print("Error: Image not loaded")
        else:
            kmeans_segmentation(img)
    elif choice == "4":
        # Mean shift algorithm
        if 'image' not in locals():
            print("Error: Image not loaded")
        else:
            print("select with mouse at screen and enter input at keyboard")
            mean_shift_segmentation()

    elif choice == "5":
        # Exit program
        break
    else:
        print("Invalid choice")
