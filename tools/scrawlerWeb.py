import csv
import re
import time
import webbrowser

import requests
BASE_URL = "http://scuvis.org"
urlList = []
for i in range(1,56):
    print(i)
    url = BASE_URL+'/page/'+str(i)
    req = requests.get(url)
    req.encoding = req.apparent_encoding
    urlList.extend(re.findall(r'<a.*?href="(.*?)".*?>*?论文研读预告</a>',req.text))

urlList = list(set(urlList))
urlList = [X for X in urlList if X[1:5] in ["2020", "2019"]]

chromePath = r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chromePath))
for url in urlList:
    time.sleep(1)
    webbrowser.get('chrome').open(BASE_URL+url,new=1,autoraise=True)
# csvfile = open(r"C:\Users\86469\Desktop\vcl.csv","w", newline='', encoding = 'utf-8-sig')
# writer = csv.writer(csvfile)
# for url in urlList:
#     writer.writerow([BASE_URL+url])
# csvfile.close()
print("done")