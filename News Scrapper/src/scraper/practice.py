from bs4 import BeautifulSoup
import requests

url="https://www.theguardian.com"

session=requests.Session()

response=session.get(url)
response.raise_for_status()

html=response.text
soup=BeautifulSoup(html,'html.parser')
link=soup.select_one("a[data-link-name='article']")
print(link.get('href'))
