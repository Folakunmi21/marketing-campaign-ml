import requests

#url = 'http://localhost:9696/response'
url = 'https://aged-dew-7973.fly.dev/response'

customer = {
    "id": 202,
    "education": "master",
    "marital_status": "single",
    "income": 82032.0,
    "kidhome": 0,
    "teenhome": 0,
    "recency": 54,
    "mntwines": 332,
    "mntfruits": 194,
    "mntmeatproducts": 377,
    "mntfishproducts": 149,
    "mntsweetproducts": 125,
    "mntgoldprods": 57,
    "numdealspurchases": 0,
    "numwebpurchases": 4,
    "numcatalogpurchases": 6,
    "numstorepurchases": 7,
    "numwebvisitsmonth": 3,
    "acceptedcmp3": 0,
    "acceptedcmp4": 0,
    "acceptedcmp5": 1,
    "acceptedcmp1": 0,
    "acceptedcmp2": 0,
    "complain": 0,
    "age": 25,
    "customer_days": 4245,
    "total_purchases": 17,
    "total_spending": 1234,
    "previous_response_rate": 0.0
}

try:
    response = requests.post(url, json=customer, timeout=5)
    response.raise_for_status()
    
    result = response.json()
    print('Response:', result)
    print(f"Probability: {result['response_probability']:.1%}")
    
    if result['will_respond']:
        print('Customer is likely to respond to the offer')
    else:
        print('Customer is not likely to respond to the offer')
        
except requests.exceptions.ConnectionError:
    print('ERROR: Cannot connect to API. Is it running at http://localhost:9696?')
except requests.exceptions.Timeout:
    print('ERROR: Request timed out.')
except Exception as e:
    print(f'ERROR: {e}')