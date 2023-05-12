'''
Functions for parsing text commands and executing them.
'''

# execute commands from queue
def commands(command, command2, tello):
    texts = command
    if "takeoff" == texts:
        if tello:
            tello.takeoff()
        print("Executing command: takeoff")

    if "forward" == texts:
        if tello:
            tello.forward(command2)
        print(f'Executing command: move forward {command2}')
    
    if "back" == texts:
        if tello:
            tello.backward(command2)
        print(f'Executing command: move backward {command2}')

    if "right" == texts:
        if tello:
            tello.right(command2)
        print(f'Executing command: move right {command2}')

    if "left" == texts:
        if tello:
            tello.left(command2)
        print(f'Executing command: move left {command2}')

    if "follow" == texts:
        print("follow")

    if "land" == texts:
        if tello:
            tello.land()
        print(f'Executing command: land')

    if "counterclockwise" == texts:
        if tello:
            tello.counter_clockwise(command2)
        print(f'Executing command: rotate counter clockwise {command2}')

    if "clockwise" == texts:
        if tello:
            tello.rotate_clockwise(command2)
        print(f'Executing command: rotate clockwise {command2}')

    if "flip" == texts:
        if tello:
            tello.flip_forward()
        print(f'Executing command: flip')

    if "up" == texts:
        if tello:
            tello.up(command2)
        print(f'Executing command: move up {command2}')

    if "down" == texts:
        if tello:
            tello.down(command2)
        print(f'Executing command: move down {command2}')

# parse a number from a string if we need to e.g. for move and rotate commands
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


# parse text for to find direction and quantity for movement and rotation commands
def direction(text, index, queue):

    if "forward" == text[index + 1] or "back" == text[index + 1] or "right"  == text[index + 1] or "left"  == text[index + 1] or "clockwise" == text[index + 1] or "counterclockwise" == text[index + 1] or "up" == text[index + 1] or "down" == text[index + 1] :
        queue.put(text[index + 1])
        number = find_number(text[index + 2])
        queue.put(number)
    return queue

# parse text for commands and add them to the queue
def queue_text(text, queue):
    
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

    return queue
