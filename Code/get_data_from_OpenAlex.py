import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# read the doi from Wikipedia Citation dataset, wchih you can find here: https://github.com/Harshdeep1996/cite-classifications-wiki
doi_list = []
with open('doi_list.txt', 'r',encoding='utf-8') as f:
    for i in f.readlines():
        doi_list.append(i.strip())

target_list = [doi[i:i+50] for i in range(0,len(doi_list),50)] # request 50 dois per time, please find more information here: https://docs.openalex.org/

# set your own headers and follow the rules on openalex.
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}


result=[]
error =[]
for i in tqdm(target_list):
    url = 'https://api.openalex.org/works?filter=doi:'+'|'.join(i)+'&per-page=200'
    try:
        res = json.loads(requests.get(url,headers=headers,timeout=30).text)
        d = [{k:v for k,v in dic_t.items() if k in ['cited_by_count','title',
                    'doi','concepts','host_venue','open_access',
                    'publication_date','referenced_works']} for dic_t in res['results']]
        if 'error' in res:
            error.append(i)
            continue
        result.extend(d)
    except:
        error.append(i)
        continue

data = pd.DataFrame(result)
data.to_parquet('openalex_data.parquet')
