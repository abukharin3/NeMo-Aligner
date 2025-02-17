import httpx

# Define the server IP and port (adjust with Node 1's IP)
server_ip = "NODE_1_IP"  # Replace with the actual IP of Node 1
server_port = 8000
url = f"http://{server_ip}:{server_port}/get_reward"

# Example queries
query_data = {
    "prompt": "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?",
    "response": "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
}

# Send the request to the server
async def get_reward():
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=query_data)
        print(f"Reward Score: {response.json()['reward']}")

# Run the query
import asyncio
asyncio.run(get_reward())