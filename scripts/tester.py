import requests
r = requests.post("http://127.0.0.1:5000/predict",
                  json={"comments": [
                      "this video was absolutely terrible",
                      "it was okay, nothing special",
                      "amazing work, i loved it",
                  ]})
print(r.status_code)
print(r.json())