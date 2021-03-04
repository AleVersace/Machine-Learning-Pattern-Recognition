import sys

# Describe the competitor
class Competitor():
    def __init__(self, name, surname, country, scores):
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = [float(i) for i in scores]
        self.avgScore = self.computeAvgScore()

    def computeAvgScore(self):
        self.scores.sort()
        self.scores = self.scores[1:(len(self.scores)-1)]
        return sum(self.scores)


    def getCountry(self):
        return self.country

    def getAvgScore(self):
        return self.avgScore

    def getName(self):
        return self.name + " " + self.surname


# Get three competitors with the best scores
def computeThreeBest(comps):
    comps.sort(key=lambda x: x.getAvgScore(), reverse=True)
    return comps[0:3]

# Fill a dict with all the countries and sum scores of their competitors
def computeBestCountry(comps):
    country = {}
    for c in comps:
        if c.getCountry() in country.keys():
            avg = country["{}".format(c.getCountry())]
            country["{}".format(c.getCountry())] = avg + c.getAvgScore()
        else:
            country["{}".format(c.getCountry())] = c.getAvgScore() 
    return max(country, key=country.get), country




if __name__ == "__main__":

    # File.txt passed as arg from command line
    if len(sys.argv) != 2:
        exit(1)

    # Read file
    comps = []
    with open(sys.argv[1], "r") as f:
        for line in f.readlines():
            l = line.split()
            comps.append(Competitor(l[0], l[1], l[2], l[3:]))
    
    treeBest = computeThreeBest(comps)
    bestCountry, countries = computeBestCountry(comps)

    print("\nFinal Ranking:\n1: {} Score: {}\n2: {} Score: {}\n3: {} Score: {}\n".format(treeBest[0].getName(), round(treeBest[0].getAvgScore(), 2), treeBest[1].getName(), round(treeBest[1].getAvgScore(), 2), treeBest[2].getName(), round(treeBest[2].getAvgScore(), 2)))
    print("\nBest Country:\n{} Total Score: {}".format(bestCountry, countries['{}'.format(bestCountry)]))