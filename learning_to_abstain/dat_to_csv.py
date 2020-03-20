import scipy.io as sio
import pandas as pd


path = "imagelabels.mat"

data = sio.loadmat(path)

print(data)

labels = data['labels'][0].tolist()
images = ['image_%05d'%(i+1) for i in range(len(labels))]

df = pd.DataFrame({"ImageID": images, "label": labels})

df.to_csv("imagelabels.csv", index=False)