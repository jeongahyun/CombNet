import pandas as pd
import os
import shutil


list = os.listdir('../SkinCancerMNIST')

base_dir = '../datasets'
os.mkdir(base_dir)

"""
nv: Melanocytic nevi
mel: Melanoma
bkl: Benign keratosis
bcc: Basal cell carcinoma
akiec: Actinic keratoses(solar keratoses)
vasc: Vascular
df: Dermatofibroma
"""

nv = os.path.join(base_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(base_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(base_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(base_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(base_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(base_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(base_dir, 'df')
os.mkdir(df)

df_data = pd.read_csv('../SkinCancerMNIST/HAM10000_metadata.csv')

folder1 = os.listdir('../SkinCancerMNIST/HAM10000_images_part_1')
folder2 = os.listdir('../SkinCancerMNIST/HAM10000_images_part_2')

image_list = df_data['image_id'].to_numpy().tolist()
df_data.set_index('image_id', inplace=True)

for image in image_list:
    fname = image + '.jpg'
    label = df_data.loc[image, 'dx']

    if fname in folder1:
        src = os.path.join('../SkinCancerMNIST/HAM10000_images_part_1', fname)
        dst = os.path.join(base_dir, label, fname)
        shutil.copyfile(src, dst)

    if fname in folder2:
        src = os.path.join('../SkinCancerMNIST/HAM10000_images_part_2', fname)
        dst = os.path.join(base_dir, label, fname)
        shutil.copyfile(src, dst)

# print(len(os.listdir('datasets/nv')))
# print(len(os.listdir('datasets/mel')))
# print(len(os.listdir('datasets/bkl')))
# print(len(os.listdir('datasets/bcc')))
# print(len(os.listdir('datasets/akiec')))
# print(len(os.listdir('datasets/vasc')))
# print(len(os.listdir('datasets/df')))
