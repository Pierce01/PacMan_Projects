import base64

with open('reinforcement.token', 'rb') as f:
    encoded_data = f.read()

decoded_data = base64.b64decode(encoded_data)
decoded_string = decoded_data.decode('latin-1')
print(decoded_string)
# print(decoded_data)