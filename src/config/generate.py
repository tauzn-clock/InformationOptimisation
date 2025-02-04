import csv
import os


data = [518.8579,518.8579,325.5824,253.7362,1000,10]

store = []

for i in range(1449):
    tmp = [f"rgb/{i}.png", f"depth/{i}.png"] + data
    tmp += [f"original_gt/{i}.png", f"original_gt/{i}.csv"]
    store.append(tmp)

with open("/HighResMDE/get_planes/ransac/config/nyu.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(store)