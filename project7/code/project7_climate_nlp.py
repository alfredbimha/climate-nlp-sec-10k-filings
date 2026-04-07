"""
===============================================================================
PROJECT 7: Climate Risk NLP Analysis of SEC 10-K Filings
===============================================================================
RESEARCH QUESTION:
    How do companies across sectors discuss climate risk in annual filings?
METHOD:
    SEC EDGAR 10-K text mining, keyword density analysis, TF-IDF, LDA topics
DATA:
    SEC EDGAR API (real filings for 10 major companies)
===============================================================================
"""
import requests, re, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

HEADERS = {'User-Agent': 'Alfred Bimha research@example.com'}

companies = {
    'AAPL':('Apple','0000320193','Technology'),
    'MSFT':('Microsoft','0000789019','Technology'),
    'XOM':('ExxonMobil','0000034088','Energy'),
    'CVX':('Chevron','0000093410','Energy'),
    'NEE':('NextEra Energy','0000753308','Utilities'),
    'JPM':('JPMorgan','0000019617','Financials'),
    'JNJ':('Johnson & Johnson','0000200406','Healthcare'),
    'CAT':('Caterpillar','0000018230','Industrials'),
    'PG':('Procter & Gamble','0000080424','Consumer'),
    'LIN':('Linde','0001707925','Materials'),
}

climate_kw = ['climate change','climate risk','greenhouse gas','carbon emission',
    'global warming','net zero','renewable energy','sustainability',
    'paris agreement','esg','tcfd','transition risk','physical risk',
    'decarboniz','carbon neutral','energy transition','environmental',
    'scope 1','scope 2','scope 3','emission reduction']

print("STEP 1: Downloading 10-K filings from SEC EDGAR...")
rows = []
for ticker, (name, cik, sector) in companies.items():
    print(f"  {ticker}...", end=" ")
    try:
        resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS)
        time.sleep(0.2)
        if resp.status_code != 200: print("skip"); continue
        recent = resp.json().get('filings',{}).get('recent',{})
        forms, dates, accs, docs = recent['form'], recent['filingDate'], recent['accessionNumber'], recent['primaryDocument']
        found = 0
        for i, f in enumerate(forms):
            if f == '10-K' and found < 2:
                acc = accs[i].replace('-','')
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{docs[i]}"
                fr = requests.get(url, headers=HEADERS); time.sleep(0.3)
                if fr.status_code == 200:
                    clean = re.sub(r'<[^>]+>',' ', fr.text)
                    clean = re.sub(r'\s+',' ', clean).lower()
                    wc = len(clean.split())
                    total = sum(clean.count(k) for k in climate_kw)
                    density = total/wc*10000 if wc > 0 else 0
                    kw_counts = {k: clean.count(k) for k in climate_kw}
                    rows.append({'ticker':ticker,'company':name,'sector':sector,
                        'date':dates[i],'word_count':wc,'climate_kw':total,
                        'density':round(density,2), 'text':clean[:3000], **kw_counts})
                    found += 1
        print(f"{found} filings")
    except Exception as e:
        print(f"error")

df = pd.DataFrame(rows)
df.drop(columns=['text'], errors='ignore').to_csv('data/filings_analysis.csv', index=False)
print(f"  Total: {len(df)} filings")

print("\nSTEP 2: Analysis...")
if len(df) > 0:
    sector_sum = df.groupby('sector')['density'].mean().sort_values(ascending=False)
    sector_sum.to_csv('output/tables/sector_density.csv')
    print(sector_sum.round(2).to_string())

print("\nSTEP 3: Visualizations...")
if len(df) > 0:
    fig, ax = plt.subplots(figsize=(10,5))
    sector_sum.sort_values().plot(kind='barh', ax=ax, color=plt.cm.RdYlGn(np.linspace(0.2,0.9,len(sector_sum))))
    ax.set_title('Climate Keyword Density by Sector (per 10K words)', fontweight='bold')
    ax.set_xlabel('Keywords per 10,000 words')
    plt.tight_layout()
    plt.savefig('output/figures/fig1_keyword_density.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10,5))
    comp = df.groupby('ticker')['density'].mean().sort_values()
    comp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Climate Disclosure Intensity by Company', fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/figures/fig2_company_density.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Keyword frequency
    kw_totals = df[climate_kw].sum().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10,5))
    kw_totals.plot(kind='bar', ax=ax, color='coral', edgecolor='white')
    ax.set_title('Most Frequent Climate Keywords in 10-K Filings', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/figures/fig3_keyword_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()

print("  COMPLETE!")
