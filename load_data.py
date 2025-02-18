import cv2

def start_load_data(labels_file):
    # Loads image paths and labels from "labels.txt"
    images = []
    labels = []
    with open(labels_file, "r") as f:
        for line in f:
            image_path, label = line.strip().split(" ")
            images.append(cv2.imread(image_path))
            labels.append(label)
    return images, labels