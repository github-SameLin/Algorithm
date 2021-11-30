import os
import csv
import pandas as pd

def main():
    pd.set_option('display.max_columns', None)
    rootPath = r"C:\Users\86469\Desktop\fsdownload\data"
    paths = os.listdir(rootPath)
    idx, data, csvPath = 0, None, ''
    year, week = 0, 0
    for pathdir in paths:
        pathdir = rootPath + '/' + pathdir
        files = os.listdir(pathdir)
        for file in files:
            rawData = pd.read_csv(pathdir + '/' + file).iloc[:, [0,1,2,3,4,5,6,7,11,12]]

            print(rawData)
            print(rawData.iloc[:, :8])
            print(rawData.add(rawData.iloc[:, :8], fill_value=0))
            return
    #         if idx == 0:
    #             data = rawData
    #             if file[20:24] != year:
    #                 year = file[20:24]
    #                 week = 0
    #             week += 1
    #             csvPath = r'%s\%s_%d.csv' % (rootPath, year, week)
    #             print(csvPath)
    #         else:
    #             print(data)
    #             data = data.add(rawData.iloc[:,:8], fill_value=0)
    #             print(data)
    #         idx = (idx + 1) % 7
    #         if idx == 0:
    #             data = data.applymap(lambda x: x/7)
    # #             data.to_csv(csvPath)
    if idx:
        data = data.applymap(lambda x: x/idx)
# data.to_csv(csvPath)

main()
