import cv2 as cv
import pytesseract as pt
import numpy as np

pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

def preprocess_image(image):
    """
    Preprocess the image by converting to grayscale and applying Gaussian blur.
    
    Args:
    image (ndarray): Input image.
    
    Returns:
    blurred_img (ndarray): Preprocessed grayscale and blurred image.
    """
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Apply Gaussian blur with a larger kernel size for more aggressive smoothing
    blurred_img = cv.GaussianBlur(gray_img, (15, 15), 0)
    return blurred_img

def adjust_points(rect, buffer=10):
    """
    Adjust points by bringing them in by a specified buffer.
    
    Args:
    rect (ndarray): Ordered points.
    buffer (int): Number of pixels to bring the points in by.
    
    Returns:
    adjusted_rect (ndarray): Adjusted points.
    """
    tl, tr, br, bl = rect
    tl[0] += buffer  # Move right
    tl[1] += buffer  # Move down
    tr[0] -= buffer  # Move left
    tr[1] += buffer  # Move down
    br[0] -= buffer  # Move left
    br[1] -= buffer  # Move up
    bl[0] += buffer  # Move right
    bl[1] -= buffer  # Move up
    return np.array([tl, tr, br, bl], dtype="float32")

def get_contours(image):
    """
    Get contours from the preprocessed image.
    
    Args:
    image (ndarray): Preprocessed image.
    
    Returns:
    contours (list): List of contours found in the image.
    hierarchy (ndarray): Hierarchy of contours.
    """
    edges = cv.Canny(image, 50, 110)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def order_points(pts):
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
    pts (ndarray): Array of points.
    
    Returns:
    rect (ndarray): Ordered points.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_sign1(img):
    """
    Process the image to correct perspective, display original and corrected images, and print extracted text.
    
    Args:
    img (ndarray): Input image.
    """
    # Display the original image
    cv.imshow('Original Image', img)
    cv.waitKey(0)

    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Get contours for the image
    contours, hierarchy = get_contours(preprocessed_img)

    # Get the largest contour based on area
    largest_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]

    # Approximate the largest contour to a polygon
    peri = cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, 0.02 * peri, True)

    # Ensure the approximated polygon has 4 points
    if len(approx) == 4:
        # Order the points
        rect = order_points(approx.reshape(4, 2))

        # Bring all sides of rect in by 10 pixels
        buffer = 10
        tl, tr, br, bl = rect
        tl[0] += buffer  # Move right
        tl[1] += buffer  # Move down
        tr[0] -= buffer  # Move left
        tr[1] += buffer  # Move down
        br[0] -= buffer  # Move left
        br[1] -= buffer  # Move up
        bl[0] += buffer  # Move right
        bl[1] -= buffer  # Move up

        # Update rect with the new points
        rect = np.array([tl, tr, br, bl], dtype="float32")

        # Set the destination points to fill the screen
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))

        # Turn the black parts of the image white and all else black
        warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

        # Threshold the image
        _, warped_thresh = cv.threshold(warped_gray, 150, 255, cv.THRESH_BINARY_INV)

        # Get rid of small white artefacts
        kernel = np.ones((3, 3), np.uint8)
        warped_thresh = cv.erode(warped_thresh, kernel, iterations=1)

        # Compute the inverse perspective transform matrix
        Minv = cv.getPerspectiveTransform(dst, rect)
        unwarped = cv.warpPerspective(warped_thresh, Minv, (img.shape[1], img.shape[0]))

        # Extract text from the unwarped image
        text_from_unwarped = pt.image_to_string(unwarped)

        # Display the corrected image
        cv.imshow('Corrected Image', unwarped)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Print the extracted text
        return text_from_unwarped
    else:
        print("Could not find a suitable contour for perspective correction.")

def process_sign_2(sign2):
    """
    Process the sign image to extract text.
    
    Args:
    sign2 (ndarray): Input image of the sign.
    
    Returns:
    text (str): Extracted text from the sign.
    """
    # Convert to grayscale, blur, and threshold for text detection
    gray = cv.cvtColor(sign2, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Do thresholding on the edge detected image
    edges = cv.Canny(blur, 100, 200)
    _, thresh = cv.threshold(edges, 100, 255, cv.THRESH_BINARY)

    # Find rectangular polygon contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter for vaguely rectangular contours
    rectangular_contours = []
    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            rectangular_contours.append(contour)

    # Keep only the biggest contour
    if rectangular_contours:
        biggest_contour = max(rectangular_contours, key=cv.contourArea)
        rectangular_contours = [biggest_contour]

        # Get rectangular approximation of biggest contour
        peri = cv.arcLength(biggest_contour, True)
        approx = cv.approxPolyDP(biggest_contour, 0.02 * peri, True)

        approx = approx.reshape(4, 2).astype(np.float32)
        ordered_pts = order_points(approx)

        # Adjust the points by bringing them in by 10 pixels
        adjusted_pts = adjust_points(ordered_pts, buffer=10)

        # Define the destination points for the warp
        width, height = 250, 50
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)

        # Compute the perspective transform matrix and warp the image
        matrix = cv.getPerspectiveTransform(adjusted_pts, dst)
        warp = cv.warpPerspective(gray, matrix, (width, height))

        # Use pytesseract to extract text from the warped image
        text = pt.image_to_string(warp)
        text = text.strip('\n')
        return text
    else:
        return "No rectangular contours found."

def process_sign3(sign3):
    # Convert to grayscale
    sign3_gray = cv.cvtColor(sign3, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv.GaussianBlur(sign3_gray, (21, 21), 0)

    # Apply binary threshold
    threshold_value = 100  # Adjust this value as needed
    _, thresh = cv.threshold(blur, threshold_value, 255, cv.THRESH_BINARY)

    # Detect edges
    edges = cv.Canny(thresh, 10, 120)

    # Find contours with hierarchy
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Filter out inner contours (those with a parent contour)
    inner_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] != -1]

    # Get largest contour and its rectangle approximation
    largest_contour = max(contours, key=cv.contourArea)

    # Approximate the contour to a polygon
    peri = cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, 0.02 * peri, True)

    approx = approx.reshape(4, 2).astype(np.float32)
    ordered_pts = order_points(approx)

    width, height = 500, 500
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)

    matrix = cv.getPerspectiveTransform(approx, dst)
    warp = cv.warpPerspective(sign3, matrix, (width, height))

    # Convert the warp image to grayscale
    warp_gray = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)

    # Apply a binary threshold to make the warp image strictly black and white, and invert it
    _, warp_bw_inv = cv.threshold(warp_gray, 50, 255, cv.THRESH_BINARY_INV)

    # Reverse the warp to get the thresholded image back to original orientation
    Minv = cv.getPerspectiveTransform(dst, approx)
    unwarp = cv.warpPerspective(warp_bw_inv, Minv, (sign3.shape[1], sign3.shape[0]))

    # Clip 50 pixels from each side of the image
    clip_amount = 50
    height, width = unwarp.shape[:2]
    cropped_unwarp = unwarp[clip_amount:height-clip_amount, clip_amount:width-clip_amount]

    # Display the cropped unwarped image
    # plt.imshow(cropped_unwarp, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()

    # Get text using pytesseract
    text = pt.image_to_string(cropped_unwarp)
    
    # replace newline characters with spaces
    text = text.replace('\n', ' ')

    return text

def process_sign4(sign4):
    # Convert the image from BGR to RGB
    sign4_rgb = cv.cvtColor(sign4, cv.COLOR_BGR2RGB)
    sign4_gray = cv.cvtColor(sign4, cv.COLOR_BGR2GRAY)

    # Gaussian blur
    sign4_blur = cv.GaussianBlur(sign4_gray, (21, 21), 0)

    # Thresholding
    _, sign4_thresh = cv.threshold(sign4_blur, 47, 255, cv.THRESH_BINARY)

    # Edge detection
    sign4_edges = cv.Canny(sign4_thresh, 50, 150)

    # Create contours
    contours, hierarchy = cv.findContours(sign4_edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Sort by area, and take the top. Draw it in red.
    top_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]

    # Let's convert the polygon represented by the biggest contour to a triangular polygon
    epsi = 0.02 * cv.arcLength(top_contour, True)
    polygon = cv.approxPolyDP(top_contour, epsi, True)

    # Adjust the polygon by bringing all triangle points in by 10
    adjustment_factor = 15

    # The top corner is the first sublist
    top = polygon[0][0]
    top_x = top[0]
    top_y = top[1]

    # The rightmost corner is the second sublist
    bottom = polygon[1][0]
    bottom_x = bottom[0]
    bottom_y = bottom[1]

    # The bottom corner is the third sublist
    right = polygon[2][0]
    right_x = right[0]
    right_y = right[1]

    # Now we need to take the top corner and increase the x by 10
    top_x += 10
    top_y += 10

    # The rightmost corner needs to decrease the x by 10 and increase the y by 10
    right_x -= 15
    right_y -= 15

    # The bottom corner needs to increase x by 10
    bottom_x += 10
    bottom_y -= 20

    # Update the polygon
    polygon[0][0] = [top_x, top_y]
    polygon[1][0] = [right_x, right_y]
    polygon[2][0] = [bottom_x, bottom_y]

    # First, we create a blank image.
    mask = np.zeros_like(sign4_gray)

    # Then, we draw the contours on the mask image
    cv.drawContours(mask, [polygon], -1, 255, thickness=cv.FILLED)

    # Apply the mask to the grayscale image to get the masked grey image
    masked_grey = cv.bitwise_and(sign4_gray, sign4_gray, mask=mask)

    # Invert the mask
    inverted_mask = cv.bitwise_not(mask)

    # Create a white background
    white_background = np.ones_like(sign4_gray) * 255

    # Combine the white background with the inverted mask
    outer_white = cv.bitwise_and(white_background, white_background, mask=inverted_mask)

    # Combine the masked grey image with the outer white image
    final_image = cv.bitwise_or(masked_grey, outer_white)

    # For final image, we threshold to just get black text on white background
    _, final_image_thresh = cv.threshold(final_image, 80, 255, cv.THRESH_BINARY)

    # Now finally use pytesseract to extract text
    text = pt.image_to_string(final_image_thresh)

    return text


# Example usage
if __name__ == "__main__":
    # Path to the image
    sign1 = 'sign1.jpg'

    # Load image
    sign1 = cv.imread(sign1)

    # Process and display the image
    imgtext = process_sign1(sign1)
    print(imgtext)
    
    # Read sign 2
    sign2 = cv.imread('sign2.jpg')

    # Process sign 2
    sign2_text = process_sign_2(sign2)
    print(sign2_text)

    # Process sign 3
    sign3 = cv.imread('sign3.jpg')
    sign3_text = process_sign3(sign3)
    print(sign3_text)

    # Process sign 4
    sign4 = cv.imread('sign4.jpg')
    sign4_text = process_sign4(sign4)
    print(sign4_text)