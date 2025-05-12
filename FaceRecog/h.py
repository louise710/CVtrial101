# print("HIII")
import pickle

with open('names.pkl', 'rb') as f:
    name = pickle.load(f)

print(name)  # Shoul