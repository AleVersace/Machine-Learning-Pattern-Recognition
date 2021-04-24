# Every bus has a unique line
# Every line may have multiple buses
# DB record format:
# 2187 13 10 1003 18000     => bus line x y time
# Command line format:
# db.txt -b busId           => Get total distance traveled from a bus
# db.txt -l lineId          => Get avg speed on the line

import sys
import math

# Check class with its information (a check updated for every bus, sort of bus status)
class Check():
    def __init__(self, busId, line, x, y, time):
        self.busId = busId
        self.line = line
        self.x = x
        self.y = y
        self.time = time
        self.totalTime = 0
        self.distance = 0

    def getTotalTime(self):
        return self.totalTime

    def getBusId(self):
        return self.busId

    def getDistance(self):
        return self.distance

    def updateDistAndSpeed(self, x, y, time):
        self.distance += distance(self.x, self.y, x, y)
        self.x = x
        self.y = y
        self.totalTime += (time - self.time)
        self.time = time


# Line class that defines a route
class Line():
    def __init__(self, lineId):
        self.lineId = lineId
        self.buses = set()

    def addBus(self, busId):
        self.buses.add(busId)

    def getLineId(self):
        return self.lineId

    def getLineBuses(self):
        return self.buses


# Find bus by busId in a list
def retreiveBus(busId, checks):
    for c in checks:
        if c.getBusId() == busId:
            return c
    return -1

# Find line/route by lineId in a list
def retreiveLine(lineId, lines):
    for l in lines:
        if l.getLineId() == lineId:
            return l
    return -1

# Calculate distance from 2 points
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Calculate distance traveled by a bus given the busId
def totalDistanceBus(busId, checks):
    # Get the latest check of the bus with all updated info
    check = retreiveBus(busId, checks)
    if check != -1:
        print("\n{} - Total Distance: {}".format(check.getBusId(), check.getDistance()))
    else:
        print("\nThis bus does not exist!")
 
# Calculate AVG Speed of the route, based on all the buses of the route
def avgSpeedLine(lineId, lines, checks):
    l = retreiveLine(lineId, lines)
    if l != -1:   
        buses = l.getLineBuses()    # Retreive all busesId on the line
        d = 0
        totalTime = 0
        for b in buses:     # Retreive total distance and time spent
            bus = retreiveBus(b, checks)
            d += bus.getDistance()
            totalTime += bus.getTotalTime()
        print("\n{} - Avg Speed: {}".format(lineId, d/totalTime))
    else:
        print("\nThis line does not exist!")


if __name__ == "__main__":

    # Command line check
    if len(sys.argv) != 4:
        print("Wrong command line arguments.\n")
        exit(1)
    
    checks = []
    lines = []
    flag = sys.argv[2]

    # File with buses information handling
    with open(sys.argv[1], "r") as f:
        for line in f.readlines():
            l = line.split()
            
            # Create new "Checkpoint"
            c = retreiveBus(int(l[0]), checks)
            if c == -1:
                checks.append(Check(int(l[0]), int(l[1]), int(l[2]), int(l[3]), int(l[4])))    # Add to the list if there is a new bus
            else:
                c.updateDistAndSpeed(int(l[2]), int(l[3]), int(l[4]))   # Update already existing bus stat

            newLine = retreiveLine(int(l[1]), lines)
            if newLine == -1:         # Add a new line/route
                lines.append(Line(int(l[1])))

            # Add the bus on the route
            route = retreiveLine(int(l[1]), lines)
            route.addBus(int(l[0]))
            

    if flag == '-b':
        totalDistanceBus(int(sys.argv[3]), checks)
    elif flag == '-l':
        avgSpeedLine(int(sys.argv[3]), lines, checks)
    else:
        print("\nFlag not allowed")
        exit(2)