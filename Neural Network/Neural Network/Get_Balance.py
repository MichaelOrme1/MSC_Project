import numpy as np

Training = np.load("TrainingData_IMG_1session.npy",allow_pickle=True)
Xtrain = Training[0]
ytrain = Training[1]

ytrain =  np.array([np.array(val) for val in ytrain])#Fixes issues with numpy loading

ytrain =  np.array([val.reshape(1,12) for val in ytrain])#Reshape to fit model
B_Value = 0
Y_Value = 0
SELECT_Value = 0
START_Value = 0
UP_Value = 0
DOWN_Value = 0
LEFT_Value = 0
RIGHT_Value = 0
A_Value = 0
X_Value = 0
L_Value = 0
R_Value = 0
total = len(ytrain)
print(total)

for inputs in ytrain:
    
    for values in inputs:
        if values[0] == 1:
            B_Value = B_Value +1

        if values[1] == 1:
            Y_Value = Y_Value +1

        if values[2] == 1:
            SELECT_Value = SELECT_Value +1

        if values[3] == 1:
            START_Value = START_Value +1

        if values[4] == 1:
            UP_Value = UP_Value +1

        if values[5] == 1:
            DOWN_Value = DOWN_Value +1

        if values[6] == 1:
            LEFT_Value = LEFT_Value +1

        if values[7] == 1:
            RIGHT_Value = RIGHT_Value +1

        if values[8] == 1:
            A_Value = A_Value +1

        if values[9] == 1:
            X_Value = X_Value +1

        if values[10] == 1:
            L_Value = L_Value +1

        if values[11] == 1:
            R_Value = R_Value +1
       
values = [B_Value,Y_Value,SELECT_Value,START_Value,UP_Value,DOWN_Value,LEFT_Value,RIGHT_Value,A_Value,X_Value,L_Value,R_Value]
print(["B",B_Value/total*100, "Y",Y_Value/total*100, "SELECT",SELECT_Value/total*100, "START",START_Value/total*100, "UP",UP_Value/total*100, "DOWN",DOWN_Value/total*100, "LEFT",LEFT_Value/total*100, "RIGHT",RIGHT_Value/total*100, "A",A_Value/total*100, "X",X_Value/total*100, "L",L_Value/total*100, "R",R_Value/total*100])


#print(values)
