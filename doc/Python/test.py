import numpy as np
import csv

with open('../data/test.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['John Smith', 'Accounting', 'November'])
    writer.writerow(['Erica Meyers', 'IT', 'March'])


