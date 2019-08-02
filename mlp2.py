# Function which convert the word number to number
def text2int(textnum, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
          raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current

print text2int("seven billion one hundred million thirty one thousand three hundred thirty seven")
#7100031337
print text2int("seven")

import numpy as np
import pandas as pd

# Load dataset
dataset = pd.read_csv('hiring.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Convert word number to actual number
for item in x:
    if item[0]== item[0]:
        item[0] = text2int(item[0])

#  Fill the missing values

# Take care  of the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(x[:,0:2])
x[:,0:2] = imputer.fit_transform(x[:,0:2])

# Create model
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(x,y)
r_sq = model.score(x,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

model.predict([[2,9,6]])

model.predict([[12,10,10]])

     