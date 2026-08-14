import requests
from bs4 import BeautifulSoup

HEAD={'User-Agent':'Mozilla/5.0'}
for ds in ['2000-12-29','2001-12-31','2005-12-30','2010-12-31','2015-12-31','2020-12-31','2025-12-31']:
    url=f'https://companiesmarketcap.com/time-machine/{ds}/'
    r=requests.get(url,headers=HEAD,timeout=30,allow_redirects=True)
    print('\nDATE',ds,'status',r.status_code,'final',r.url,'len',len(r.text))
    soup=BeautifulSoup(r.text,'html.parser')
    print('title',soup.title.get_text(' ',strip=True) if soup.title else None)
    rows=[]
    for tr in soup.select('table tbody tr')[:25]:
        txt=' | '.join(td.get_text(' ',strip=True) for td in tr.find_all('td'))
        if txt: rows.append(txt)
    print('rows',len(rows))
    for x in rows[:5]: print(x[:300])
