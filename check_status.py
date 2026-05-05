import requests
import json

TOKEN = "YOUR_TOKEN_HERE"
SERVICE_ID = "YOUR_SERVICE_ID_HERE"
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
API = "https://backboard.railway.app/graphql/v2"

q = """
{
  deployments(input: {serviceId: "ba8be940-cd14-4137-ab8a-b9d21019cdd6"}) {
    edges {
      node {
        id
        status
        createdAt
      }
    }
  }
}
"""
r = requests.post(API, headers=headers, json={"query": q})
data = r.json()
deployments = data.get("data", {}).get("deployments", {}).get("edges", [])
if not deployments:
    print("Error:", json.dumps(data, indent=2))
else:
    print(f"Latest deployments:")
    for d in deployments[:3]:
        n = d["node"]
        print(f"  Status: {n['status']} | Created: {n['createdAt'][:19]}")
