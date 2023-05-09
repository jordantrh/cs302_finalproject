# import tellopy
from queue import Queue
import time


def commands(command, command2, tello):
    texts = command
    if "takeoff" == texts:
        if tello:
            tello.takeoff()
        time.sleep(3)
        print("takeoff")

    if "forward" == texts:
        if tello:
            tello.forward(command2)
        print("forward")
        print(command2)
    
    if "back" == texts:
        if tello:
            tello.backward(command2)
        print("back")
        print(command2)

    if "right" == texts:
        if tello:
            tello.right(command2)
        print("right")
        print(command2)

    if "left" == texts:
        if tello:
            tello.left(command2)
        print("left")
        print(command2)

    if "follow" == texts:
        print("follow")

    if "land" == texts:
        if tello:
            tello.land()
        print("land")

    if "counterclockwise" == texts:
        if tello:
            tello.counter_clockwise(command2)
        print("counterclockwise")

    if "clockwise" == texts:
        print("clockwise")
        print(command2)
        if tello:
            tello.clockwise(command2)

    if "flip" == texts:
        if tello:
            tello.flip(command2)
        print("flip")


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


def direction(text, index, queue):

    if "forward" == text[index + 1] or "back" == text[index + 1] or "right"  == text[index + 1] or "left"  == text[index + 1] or "clockwise" == text[index + 1] or "counterclockwise" == text[index + 1]:
        queue.put(text[index + 1])
        number = find_number(text[index + 2])
        queue.put(number)


def queue_text(text, queue):
    # Find the next command(forward, back, takeoff, & land)
    text = text.replace('.','')
    search = text.split()
    search = [x.lower() for x in search]

    for i in range(len(search)):
        if "move" == search[i]:
            queue = direction(search, i, queue)
        elif "take" == search[i] and "off" == search[i+1]:
            queue.put('takeoff')
        elif "takeoff" == search[i]:
            queue.put(search[i])
        elif "land" == search[i]:
            queue.put(search[i])
        elif "rotate" == search[i]:
            queue = direction(search, i, queue)
        elif "flip" == search[i]:
            queue.put(search[i])

    print(str(search))
    return queue
