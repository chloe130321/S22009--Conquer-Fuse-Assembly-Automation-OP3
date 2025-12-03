from harvesters.core import Harvester
import cv2
import os
import datetime

class PictureTaker:
    def __init__(self, harvester, base_dir, serial_number):
        """
        Initialize PictureTaker with a shared Harvester instance.

        Args:
            harvester (Harvester): Shared Harvester instance.
            base_dir (str): Directory to save images.
            serial_number (str): Serial number of the camera to initialize.
        """
        self.base_dir = base_dir
        self.h = harvester
        self.serial_number = serial_number
        os.makedirs(self.base_dir, exist_ok=True)
        self.ia = None
        self.reinitialize()


    def reinitialize(self):
        """Reinitialize the camera if it was destroyed or lost connection."""
        print(f"Looking for camera with serial number: {self.serial_number}")
        self.ia = None  # Reset the image acquirer
        for idx, device_info in enumerate(self.h.device_info_list):
            if device_info.serial_number == self.serial_number:
                print(f"Matched serial number: {self.serial_number}")
                self.ia = self.h.create(idx)
                break
        if not self.ia:
            raise ValueError(f"Camera with serial number {self.serial_number} not found.")
        print(f"PictureTaker initialized successfully for {self.serial_number}.")

    def start_acquisition(self):
        """Start image acquisition."""
        if self.ia:
            self.ia.start()
            print(f"Acquisition started for camera {self.serial_number}.")
        else:
            raise RuntimeError(f"No image acquirer available for camera {self.serial_number}.")

    def stop_acquisition(self):
        """Stop image acquisition without destroying the camera instance."""
        if self.ia:
            self.ia.stop()
            print(f"Acquisition stopped for camera {self.serial_number}.")
        else:
            print(f"No image acquirer to stop for camera {self.serial_number}.")

    def fetch_image(self):
        """
        Fetch and process an image from the camera.

        Returns:
            numpy.ndarray: Processed image in OpenCV-compatible format.
        """
        if not self.ia:
            raise RuntimeError(f"No image acquirer available for camera {self.serial_number}.")

        with self.ia.fetch() as buffer:
            component = buffer.payload.components[0]
            height, width = component.height, component.width
            image_data = component.data.reshape(height, width)  # Ensure correct shape

            # Convert to OpenCV format to avoid black screen issue
            image_np = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR) if len(image_data.shape) == 2 else image_data

            return image_np

    def save_image(self, image_np, base_name='captured_image', extension='png'):
        """
        Save the image inside a date-based folder.

        Args:
            image_np (numpy.ndarray): Processed image.
            base_name (str): Base name for the saved image.
            extension (str): File extension for the saved image.

        Returns:
            str: Full path to the saved image.
        """
        # Generate today's date folder (YYYY-MM-DD)
        today_folder = datetime.datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_dir, today_folder)

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique file name
        file_name = f"{base_name}_{self.serial_number}.{extension}"
        image_path = os.path.join(save_dir, file_name)

        # Save the image only once
        cv2.imwrite(image_path, image_np)
        print(f"Image saved: {image_path}")

        return image_path
