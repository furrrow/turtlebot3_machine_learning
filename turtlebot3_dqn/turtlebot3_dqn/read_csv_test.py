import csv

with open('rac.csv', newline='') as csvfile:

    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

    for row in spamreader:

        print(', '.join(row))