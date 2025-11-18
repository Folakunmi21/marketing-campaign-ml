import requests

url = 'http://localhost:9696/response'

customer = {
    "id": 202,
    "education": "masters",
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

response = requests.post(url, json=customer)
response_predictions = response.json()

print('response:', response_predictions)

if response_predictions['response'] > 0.5:
    print('customer is likely to respond to offer')
else:
    print('customer is not likely to respond to offer')