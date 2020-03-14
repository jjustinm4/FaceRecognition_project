import shutil
import facenet
import os

dataset=facenet.get_dataset('dataset')
class_names = [ cls.name.replace('_', ' ') for cls in dataset]

for i,name in enumerate(class_names):
    if (name!='Justin' and name!='Arun' and name!='Nijo'):
        shutil.rmtree('dataset\\'+name)

os.system("python create_emb.py")

print("finish reset")
