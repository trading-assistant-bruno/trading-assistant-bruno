import io, requests, pandas as pd
url='https://coinmarketcap.com/historical/20190106/'
r=requests.get(url,headers={'User-Agent':'Mozilla/5.0 AppleWebKit/537.36 Chrome/128 Safari/537.36','Accept-Language':'en-US,en;q=0.9'},timeout=30)
print('status',r.status_code,'len',len(r.text))
print('contains BTC', 'Bitcoin' in r.text, 'contains $4,076', '4,076' in r.text)
tables=pd.read_html(io.StringIO(r.text))
print('tables',len(tables))
for i,t in enumerate(tables):
 print('\nTABLE',i,'shape',t.shape,'columns',repr(list(t.columns)))
 print(t.head(25).to_string())
