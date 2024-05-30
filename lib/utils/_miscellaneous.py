import random
import math


def is_overlap(new_circle, circles, r):
    for circle in circles:
        dist = math.sqrt((new_circle[0] - circle[0]) ** 2 + (new_circle[1] - circle[1]) ** 2)
        if dist < 2 * r:
            return True
    return False


def generate_circles(n, r, area):
    circles = []
    attempts = 0
    max_attempts = 10

    while len(circles) < n and attempts < max_attempts:
        angle = random.uniform(0, 2 * math.pi)
        direction = random.uniform(0, 360)
        distance = random.uniform(0, area - r)
        new_circle = (distance * math.cos(angle), distance * math.sin(angle), direction)

        if not is_overlap(new_circle, circles, r):
            circles.append(new_circle)
            attempts = 0
            continue
        attempts += 1

    if attempts == max_attempts:
        print(f"Warning: Maximum attempts reached with {len(circles)} circles placed.")

    return circles
