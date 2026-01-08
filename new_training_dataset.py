import os
import cv2

# Ensure the directories exist before starting
base_path = "own_data"
categories = [["Gelber Sack", 1], ["Biomuell", 1], ["Restmuell", 1], ["Papiermuell", 1]]

for cat_name, _ in categories:
    # Creating a path friendly name (no spaces)
    folder_name = cat_name.replace(" ", "_")
    os.makedirs(os.path.join(base_path, folder_name), exist_ok=True)

# 1. Initialize camera OUTSIDE the loop
cap = cv2.VideoCapture(1)  # Changed to 0 as default

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    display_frame = frame.copy()
    cv2.imshow("Trash Detection", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Mapping keys to indices to keep the code DRY (Don't Repeat Yourself)
    key_map = {ord('g'): 0, ord('b'): 1, ord('r'): 2, ord('p'): 3}

    if key in key_map:
        idx = key_map[key]
        folder = categories[idx][0].replace(" ", "_")
        count = categories[idx][1]

        filename = os.path.join(base_path, folder, f"{count}.jpg")
        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")
        categories[idx][1] += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()