import numpy as np
import cv2


def kmeans_segmentation(image):
        # Change color to RGB (from BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshape image into a 2D array of pixels and 3 color values (RGB)
        pixel_vals = image.reshape((-1,3))

        # Convert to float type
        pixel_vals = np.float32(pixel_vals)
        # define stopping criteria
        # you can change the number of max iterations for faster convergence!
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        ## TODO: Select a value for k
        # then perform k-means clustering
        k = 4  # set k to 4 to include red color
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # convert data into 8-bit values
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]

        # reshape data into the original image dimensions
        segmented_image = segmented_data.reshape((image.shape))
        labels_reshape = labels.reshape(image.shape[0], image.shape[1])

        # change the fourth cluster (red color) to all red pixels
        segmented_image[labels_reshape == 3] = [255, 0, 0]

        # Convert to grayscale
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

        # Threshold to create a binary image
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a rectangle around each contour that is large enough
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:
                cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the segmented image with rectangles around the red color region using cv2.imshow
        cv2.imshow('Segmented Image with Rectangles', segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
