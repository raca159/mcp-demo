import requests
import urllib.parse


prompt = 'give be the result of a 2+2 operation'

encoded_prompt = urllib.parse.quote(prompt)
result = requests.post(
    f"http://localhost:8080/research",
    headers={'accept': 'application/json'},
    json={'prompt': prompt}
)
print(result.json())