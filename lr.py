import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv('HR_comma_sep.csv')

# Load the dataset in x and y
x = dataset.iloc[:,:-4].values
y = dataset.iloc[:,6].values


# Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state=0)

# Create model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

model.score(x_test,y_test)


# Building optimal model using Backward elimination
# import statsmodels.api as sm

# x= np.append(arr= np.ones((14999,1)).astype(int), values = x, axis=1)
# x = sm.add_constant(x)


# x_opt = x[:,[0,1,2,3,4,5,6]] 
# model_ols = sm.Logit(y, x_opt)
# results = model_ols.fit()
# print(results.summary())

# Arrange the array
departments = list(set(dataset.Department))
departmentsValues_Left = []
departmentsValues_Stayed = []

# initialize the array
for i in range(len(departments)):
    departmentsValues_Left.append(0)
    departmentsValues_Stayed.append(0)
   
# Get the values of the departments
for i in range(len(departments)):
    print(departments[i])
    for j in range(len(dataset)):
        if dataset.iloc[j].Department == departments[i]:
            if dataset.iloc[j].left == 0:
                departmentsValues_Left[i] = departmentsValues_Left[i] + 1
            else:
                departmentsValues_Stayed[i] = departmentsValues_Stayed[i] + 1



# Generate Bar chart
import matplotlib.pyplot as plt
    
x_axis = np.arange(len(departments))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x_axis - width/2, departmentsValues_Left, width, label='Left')
rects2 = ax.bar(x_axis + width/2, departmentsValues_Stayed, width, label='Stayed')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Department wise Retention')
ax.set_xticks(x_axis)
ax.set_xticklabels(departments)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# Arrange the array
salaries = list(set(dataset.salary))
salariesValues_Left = []
salariesValues_Stayed = []


# initialize the array
for i in range(len(salaries)):
    salariesValues_Left.append(0)
    salariesValues_Stayed.append(0)


    
# Get the values of the departments
for i in range(len(salaries)):
    print(salaries[i])
    for j in range(len(dataset)):
        if dataset.iloc[j].salary == salaries[i]:
            if dataset.iloc[j].left == 0:
                salariesValues_Left[i] = salariesValues_Left[i] + 1
            else:
                salariesValues_Stayed[i] = salariesValues_Stayed[i] + 1


# Generate Bar chart
import matplotlib.pyplot as plt
    
x_axis = np.arange(len(salaries))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x_axis - width/2, salariesValues_Left, width, label='Left')
rects2 = ax.bar(x_axis + width/2, salariesValues_Stayed, width, label='Stayed')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Salary wise Retention')
ax.set_xticks(x_axis)
ax.set_xticklabels(salaries)
ax.legend()

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
