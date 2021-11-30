import os
import pandas as pd

if __name__ == "__main__":
    csv_path = './dataset/CN-Reanalysis2016010108.csv'
    pd_data = pd.read_csv(csv_path)
    # print(pd_data)
    # print(str(pd_data.iloc[:, :-1].values.tolist()))
    # print(max(pd_data.iloc[:,0]))
    with open(csv_path[:-4]+"_.txt", 'w') as f:
        f.write(str(pd_data.iloc[:, [-2,-3,0]].values.tolist()))
    print("done")
