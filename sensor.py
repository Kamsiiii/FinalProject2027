import leap
import time

class SensorConnection:
    """Manages the connection to the Leap Motion Controller."""

    def __init__(self):
        self.controller = leap.Controller()
        self._wait_for_connection()

    def _wait_for_connection(self, timeout=10):
        print("Connecting to Leap Motion...")
        start = time.time()
        while not self.controller.is_connected:
            if time.time() - start > timeout:
                raise RuntimeError(
                    "Leap Motion not found. Check the device is plugged in "
                    "and the Ultraleap service is running."
                )
            time.sleep(0.1)
        print("Connected.")

    def get_frame(self):
        """Returns the most recent frame, or None if unavailable."""
        frame = self.controller.frame()
        return frame if frame.is_valid else None

    def get_hand(self, frame=None):
        """Returns the first detected Hand object, or None."""
        if frame is None:
            frame = self.get_frame()
        if frame and frame.hands:
            return frame.hands[0]
        return None

    def is_connected(self):
        return self.controller.is_connected