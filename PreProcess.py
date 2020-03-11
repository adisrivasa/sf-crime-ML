import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import warnings
from sklearn.preprocessing import LabelEncoder, scale
warnings.filterwarnings("ignore")

def address(data):
    data['StreetNo'] = data['Address'].apply(lambda x: 1 if x.split(' ', 1)[0].isdigit() else 0)
    data["Intersection"] = data["Address"].apply(lambda x: 1 if "/" in x else 0)
    return data

def multicrime(data):
    multCount = data.groupby(['Dates', 'X', 'Y']).size()
    multCount = pd.DataFrame(multCount)
    multCount = multCount.reset_index()
    data = data.merge(multCount, how='right')
    data.rename(columns={0:'#MultCrimes'}, inplace=True)
    data["#MultCrimes"] = data["#MultCrimes"].astype(int)
    return data


def datesplit(data):
    data["Dates"] = data["Dates"].astype('datetime64')
    data['Date'] = pd.to_datetime(data['Dates'].dt.date)
    data["Year"] = data["Dates"].dt.year
    data["Month"] = data["Dates"].dt.month
    data["Time"] = data["Dates"].dt.hour + data["Dates"].dt.minute/60
    data["Hour"] = data["Dates"].dt.hour
    del data["Dates"]
    return data

def cleancoord(data):
    listOfPd = data["PdDistrict"].unique()
    for i in listOfPd:
        initial = data[data['PdDistrict']==i]
        data_i = initial[initial['Y']!=90]
        data.loc[(data['Y'] > 38) & (data['PdDistrict'] == i), 'Y']= data_i.Y.mean()
        data.loc[(data['X'] > (-120.6)) & (data['PdDistrict'] == i), 'X'] = data_i.X.mean()
    return data

def workdays(data):
    #Considering certain government declared holidays
    # - New Year
    # - Independence day
    # - Christmas
    # - Memorial Day
    # - Thanksgiving
    cal = calendar()
    print(cal.rules)
    # cal.rules.pop(7)
    # cal.rules.pop(6)
    # cal.rules.pop(5)
    # cal.rules.pop(2)
    # cal.rules.pop(1)
    holidays = cal.holidays(start='2003-01-01', end='2015-05-13')
    data['WorkDay'] = ((data['DayOfWeek'].isin(['Saturday', 'Sunday'])==False) & (data['Date'].isin(holidays)==False))
    return data

def corrmap(data):
    data.corr()
    heat = plt.subplots(figsize=(20,20))
    sns.heatmap(data.corr(),annot=True)
    plt.show()

def nightime(data):
    data["Nightime"] = True
    data.loc[(data['Month']==1) & (data['Time'] > 7.0) & (data['Time'] < 17.5), 'Nightime'] = False
    data.loc[(data['Month']==2) & (data['Time'] > 7.0) & (data['Time'] < 17.5), 'Nightime'] = False
    data.loc[(data['Month']==3) & (data['Time'] > 6.5) & (data['Time'] < 18.0), 'Nightime'] = False
    data.loc[(data['Month']==4) & (data['Time'] > 6.5) & (data['Time'] < 18.0), 'Nightime'] = False
    data.loc[(data['Month']==5) & (data['Time'] > 6.0) & (data['Time'] < 18.5), 'Nightime'] = False
    data.loc[(data['Month']==6) & (data['Time'] > 6.25) & (data['Time'] < 18.5), 'Nightime'] = False
    data.loc[(data['Month']==7) & (data['Time'] > 6.0) & (data['Time'] < 18.75), 'Nightime'] = False
    data.loc[(data['Month']==8) & (data['Time'] > 6.25) & (data['Time'] < 19.0), 'Nightime'] = False
    data.loc[(data['Month']==9) & (data['Time'] > 6.25) & (data['Time'] < 18.75), 'Nightime'] = False
    data.loc[(data['Month']==10) & (data['Time'] > 6.5) & (data['Time'] < 18.0), 'Nightime'] = False
    data.loc[(data['Month']==11) & (data['Time'] > 6.5) & (data['Time'] < 18.0), 'Nightime'] = False
    data.loc[(data['Month']==12) & (data['Time'] > 7.0) & (data['Time'] < 17.5), 'Nightime'] = False
    return data

def labencd(data):
    preProc = LabelEncoder()
    data['PdDistrict'] = preProc.fit_transform(data.PdDistrict)
    data.loc[data['DayOfWeek'] == 'Monday', 'DOW'] = 1
    data.loc[data['DayOfWeek'] == 'Tuesday', 'DOW'] = 2
    data.loc[data['DayOfWeek'] == 'Wednesday', 'DOW'] = 3
    data.loc[data['DayOfWeek'] == 'Thursday', 'DOW'] = 4
    data.loc[data['DayOfWeek'] == 'Friday', 'DOW'] = 5
    data.loc[data['DayOfWeek'] == 'Saturday', 'DOW'] = 6
    data.loc[data['DayOfWeek'] == 'Sunday', 'DOW'] = 7
    data["DOW"] = data["DOW"].astype(int)
    return data

def scaleXY(data):
    data['X'] = scale(list(map(lambda x: x+122.4194, data.X)))
    data['Y'] = scale(list(map(lambda x: x-37.7749, data.Y)))
    data["rot45_X"] = 0.707* data["Y"] + 0.707* data["X"]
    data["rot45_Y"] = 0.707* data["Y"] - 0.707* data["X"]
    data["rot30_X"] = 0.866* data["X"] + 0.5* data["Y"]
    data["rot30_Y"] = 0.866* data["Y"] - 0.5* data["X"]
    data["rot60_X"] = 0.5* data["X"] + 0.866* data["Y"]
    data["rot60_Y"] = 0.5* data["Y"] - 0.866* data["X"]
    data["radial_r"] = np.sqrt( np.power(data["Y"],2) + np.power(data["X"],2))
    return data

def delcols(data):
    del data["Descript"]
    del data["Resolution"]
    del data["Date"]
    #del data["Time"]
    del data["Address"]
    del data["DayOfWeek"]
    return data

def main():
    #Loading the datasets and parsing the dates
    train = pd.read_csv("train.csv", parse_dates=["Dates"])
    test = pd.read_csv("test.csv", parse_dates=["Dates"])
    #Rearranging the columns of the datasets
    train = train[['Id', 'Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']]
    test = test[['Id', 'Dates', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']]


    #Incorrect values of X, Y coordinates corrected
    train = cleancoord(train)
    test = cleancoord(test)
    print("Coordinates cleaned")

    #Adding Simultaneous crimes.
    #train = multicrime(train)
    #test = multicrime(test)
    print("MultCrimes split")

    #Splitting the dates feature.
    train = datesplit(train)
    test = datesplit(test)
    print("Dates split")


    #Checking workingdays across the timeline
    train = workdays(train)
    test = workdays(test)
    print("Workdays checked")

    #Checking nightime with respect to different seasons, PFA data used for sunrise and sunset times.
    train = nightime(train)
    test = nightime(test)
    print("Night time updated")

    #Cleaning Address
    train = address(train)
    test = address(test)
    print("Address updated into Intersection and StreetNo")

    #LabelEncoding the values of PdDistrict and DayOfWeek.
    train = labencd(train)
    test = labencd(test)
    print("Label Encoding done")

    #Scaling the X, Y coordinates for easy computation
    train = scaleXY(train)
    test = scaleXY(test)
    print("X,Y scaled")

    #Delete columns
    train = delcols(train)
    test = delcols(test)

    #corrmap(train)
    #corrmap(test)

    print(train.info())
    print(test.info())
    train.to_csv('train_final.csv', index=False)
    test.to_csv('test_final.csv', index=False)

if __name__ =="__main__":
    main()
