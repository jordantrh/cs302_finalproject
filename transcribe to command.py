#import tellopy
from djitellopy import Tello
from queue import Queue
import time

tello = Tello()
tello.connect()
queue = Queue()
print(tello.get_battery())
text = input("Enter text: ")

def commands(command, command2):
    texts = command
    if "takeoff" == texts:
        tello.takeoff()
        time.sleep(3)
        print("takeoff")

    if "forward" == texts:
        tello.move_forward(command2)
        print("forward")
        print(command2)
    if "back" == texts:
        print("back")
        print(command2)
    if "follow" == texts:
        print("follow")

    if "land" == texts:
        tello.land()
        print("land")

    if "counter" == texts:
        tello.rotate_counter_clockwise(command2)
        print("counter")

    if "clockwise" == texts:
        print("clockwise")
        print(command2)
        tello.rotate_clockwise(command2)

    if "follow" == texts:
        #





def find_number(text):

    number_str = ""

    for char in text:
        # Check if the current character is a number
        if char.isdigit():
            number_str += char

    # Check if a number was found in the text
    if number_str:
        number = int(number_str)
        return number
    else:
        return None


def direction(text, index):

    if "forward" == text[index + 1] or "back" == text[index + 1] or "clockwise" == text[index + 1] or "counter" == text[index + 1]:
        queue.put(text[index + 1])
        number = find_number(text[index + 2])
        queue.put(number)
    elif "forward" == text[index + 2] or "back" == text[index + 2] or "clockwise" == text[index + 2] or "counter" == text[index + 2]:
        queue.put(text[index + 2])
        number = find_number(text[index + 3])
        queue.put(number)
    elif "forward" == text[index + 3] or "back" == text[index + 3] or "clockwise" == text[index + 3] or "counter" == text[index + 3]:
        queue.put(text[index + 3])
        number = find_number(text[index + 4])
        queue.put(number)



def queue_text(text):
    # Find the next command(forward, back, takeoff, & land)
    search = text.split()

    for i in range(len(search)):
        if "move" == search[i]:
            direction(search, i)
        elif "takeoff" == search[i]:
            queue.put(search[i])
        elif "land" == search[i]:
            queue.put(search[i])
        elif "rotate" == search[i]:
            direction(search, i)
        elif "follow" == search[i]:
            queue.put(search[i])

    print(str(search))


queue_text(text)


#negative counterclockwise
def rotation(angle):
    if angle < 0:
        commands('counter', angle)
    else:
        commands('clockwise', angle)


def move(distance):
    commands('forward', distance)


while not queue.empty():
    j = queue.get(0)
    k = 0
    if j == "forward" or j == "back" or j == "clockwise" or j == "counter":
        k = queue.get(0)
    commands(j, k)


tello.disconnect()
