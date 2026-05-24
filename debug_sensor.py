from sensor import SensorConnection
import time

s = SensorConnection()
print("Show your hand and hold a letter...")
time.sleep(2)

for i in range(3):
    hand = s.get_hand()
    if hand:
        fingers = hand["fingers"]
        print("\n--- Frame", i, "---")
        print("Palm position:", [round(x,1) for x in hand["palm_position"]])
        print("Palm normal:  ", [round(x,3) for x in hand["palm_normal"]])
        print("Direction:    ", [round(x,3) for x in hand["direction"]])
        for j, f in enumerate(fingers):
            print(f"Finger {j}: extended={f['is_extended']}")
            print(f"  tip={[round(x,1) for x in f['tip_position']]}")
            print(f"  mcp={[round(x,1) for x in f.get('mcp_position', [0,0,0])]}")
            print(f"  pip={[round(x,1) for x in f.get('pip_position', [0,0,0])]}")
    time.sleep(0.5)