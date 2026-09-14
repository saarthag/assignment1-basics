#! /usr/bin/env python
from cs336_basics import linear

if __name__ == "__main__":
    linear_layer = linear.Linear(2, 2, None, None)
    print(linear_layer.state_dict())
