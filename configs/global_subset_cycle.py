def get_subset_size_for_cycle(cycle): # modified this to change how the schedule work
    if cycle < 5:
        return 8
    elif cycle < 10:
        return 16
    elif cycle < 15:
        return 24
    elif cycle < 20:
        return 32
    else:
        return 64