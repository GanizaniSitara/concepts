import os
import time
from datetime import datetime
from PIL import ImageGrab, ImageChops

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def images_are_identical(img1, img2):
    return ImageChops.difference(img1, img2).getbbox() is None

    # Alternative (potentially faster) method using byte hash comparison:
    # return hashlib.md5(img1.tobytes()).digest() == hashlib.md5(img2.tobytes()).digest()


def take_screenshot():
    return ImageGrab.grab()

def save_screenshot(image, path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(path, f'screenshot_{timestamp}.png')
    image.save(filename)
    print(f"Saved: {filename}")
    return filename

def main():
    save_dir = './screenshots'
    ensure_directory(save_dir)

    last_image = None
    last_path = None

    try:
        while True:
            current_image = take_screenshot()

            if last_image and images_are_identical(last_image, current_image):
                # Remove old file if it exists
                if last_path and os.path.exists(last_path):
                    os.remove(last_path)
                    print(f"Removed duplicate: {last_path}")

            # Save new screenshot regardless, to preserve the new timestamp
            last_path = save_screenshot(current_image, save_dir)
            last_image = current_image

            time.sleep(3)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()
