import cv2
import numpy as np
import os

# Step 3: Define the function to perform template matching and categorize
def match_templates_in_boxes(image, bounding_boxes, template_folder):
    categorized_templates = {i: [] for i in range(len(bounding_boxes))}  # To store matched templates by row
    
    # Step 4: Read template images from the folder
    for template_file in os.listdir(template_folder):
        template_path = os.path.join(template_folder, template_file)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # Load template in grayscale
        
        if template is None:
            continue  # Skip if the template is not readable
        
        template_h, template_w = template.shape
        
        # Step 5: Match the template in each bounding box
        for idx, (top_left, bottom_right) in enumerate(bounding_boxes):
            # Crop the area corresponding to the current bounding box
            cropped_region = gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # Perform template matching
            result = cv2.matchTemplate(cropped_region, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Step 6: Check if the match is good enough (use a threshold)
            match_threshold = 0.8  # Adjust this value as needed
            if max_val >= match_threshold:
                # If a good match is found, save the template's filename and row index (idx)
                categorized_templates[idx].append(template_file)
                
                # Optionally draw the match on the image for visualization
                match_top_left = (max_loc[0] + top_left[0], max_loc[1] + top_left[1])
                match_bottom_right = (match_top_left[0] + template_w, match_top_left[1] + template_h)
                cv2.rectangle(image, match_top_left, match_bottom_right, (0, 255, 255), 2)
    
    return categorized_templates

# Step 1: Load the image
image = cv2.imread('input/testuser1.png')

# Step 2: Convert to grayscale (optional if working with a grayscale image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Get the image dimensions
image_height, image_width = gray.shape

# Step 4: Known color of the black line (this is an assumption, adjust accordingly)
# Assuming the black line has a grayscale intensity of 0 (pure black) or a specific RGB value.
line_color_value = 0  # Change this to the correct value if needed

# Step 5: Tolerance for detecting "black" (to account for noise)
tolerance = 10  # Adjust this based on the variation of the black line

# Step 6: Detect horizontal lines by scanning each row for uniform black pixels
line_positions = []
for y in range(image_height):
    row = gray[y, :]  # Get all pixels in the row (one row at a time)
    
    # Check if all or most pixels in the row match the black line color
    if np.all(np.abs(row - line_color_value) <= tolerance):
        line_positions.append(y)
line_positions.append(image_height - 1)

# Step 7: Group line positions (in case multiple consecutive rows match the line color)
filtered_lines = []
for i in range(1, len(line_positions)):
    if line_positions[i] != line_positions[i-1] + 1:  # New line detected
        filtered_lines.append(line_positions[i-1])
filtered_lines.append(line_positions[-1])  # Append the last line

# Step 8: Define bounding boxes between the detected lines
bounding_boxes = []
for i in range(len(filtered_lines) - 1):
    y_top = filtered_lines[i]
    y_bottom = filtered_lines[i+1]
    
    # Bounding box coordinates (top left and bottom right corners)
    top_left = (0, y_top)
    bottom_right = (image_width, y_bottom)
    
    bounding_boxes.append((top_left, bottom_right))
# Step 7: Specify the folder with template images
template_folder = 'output/'

# Step 8: Call the function to match templates and categorize by rows
categorized_results = match_templates_in_boxes(gray, bounding_boxes, template_folder)

# Step 9: Print the categorized results
for row_idx, templates in categorized_results.items():
    print(f"Row {row_idx}: {templates}")

# Step 10: Display the result (with matched templates highlighted)
cv2.imshow('Image with Matched Templates', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
