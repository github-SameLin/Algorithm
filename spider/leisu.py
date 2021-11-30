import re

import requests
# from requests import request

if __name__ == "__main__":
    rsp = requests.get(r'https://live.leisu.com/lanqiu')
    rsp.encoding = "utf-8"
    print(rsp.content)
    print(re.search('TEAMS=(.*?)$',str(rsp.content)).group(0))