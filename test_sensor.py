from sensor import SensorConnection
import time

s = SensorConnection()
print("Waiting for hand data... put your hand over the sensor")

for i in range(20):
    hand = s.get_hand()
    print("Hand data received: " + str(hand is not None))
    time.sleep(0.5)