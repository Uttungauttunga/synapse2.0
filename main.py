import cv2
import json
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.wkt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image
from tqdm import tqdm


class XView2:
    def __init__(self, img_dir, lbl_dir):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir

        self.imgs = sorted([os.path.join(img_dir, im) for im in os.listdir(self.img_dir)])
        self.jsons = sorted([os.path.join(lbl_dir, lbl) for lbl in os.listdir(self.lbl_dir) if lbl[-5:] == ".json"])

        # Load annotations into memory
        print('Loading annotations into memory...')
        tic = time.time()
        self.anns = [json.load(open(ann, 'r')) for ann in self.jsons]
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        # Create annotation dictionary
        self.anndict = dict(zip(self.jsons, self.anns))
        # Create annotation dataframe
        print('Creating annotation dataframe...')
        tic = time.time()
        self.anndf = self.generate_dataframe()
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        self.colordict = {'none': 'c',
                          'no-damage': 'w',
                          'minor-damage': 'darkseagreen',
                          'major-damage': 'orange',
                          'destroyed': 'red',
                          'un-classified': 'b'}

    def generate_dataframe(self):
        """
        Generate main annotation dataframe.

        :return: anndf (pandas DataFrame)
        """
        ann_list = []
        for k, ann in self.anndict.items():
            if ann['features']['xy']:
                # Get features
                feature_type = []
                uids = []
                pixwkts = []
                dmg_cats = []
                imids = []
                types = []

                for i in ann['features']['xy']:
                    feature_type.append(i['properties']['feature_type'])
                    uids.append(i['properties']['uid'])
                    pixwkts.append(i['wkt'])
                    if 'subtype' in i['properties']:
                        dmg_cats.append(i['properties']['subtype'])
                    else:
                        dmg_cats.append("none")
                    imids.append(ann['metadata']['img_name'].split('_')[1])
                    types.append(ann['metadata']['img_name'].split('_')[2])

                geowkts = [i['wkt'] for i in ann['features']['lng_lat']]
                # Get Metadata
                cols = list(ann['metadata'].keys())
                vals = list(ann['metadata'].values())

                newcols = ['obj_type', 'img_id', 'type', 'pixwkt', 'geowkt', 'dmg_cat', 'uid'] + cols
                newvals = [[f, _id, t, pw, gw, dmg, u] + vals for f, _id, t, pw, gw, dmg, u in
                           zip(feature_type, imids, types, pixwkts, geowkts, dmg_cats, uids)]
                df = pd.DataFrame(newvals, columns=newcols)
                ann_list.append(df)
        return pd.concat(ann_list, ignore_index=True)

    def view_pre_post(self, disaster, imid):
        assert disaster in self.anndf.disaster.unique()

        predf, postdf = self.pre_post_split(imid, disaster)

        fig, axes = plt.subplots(1, 4, figsize=(20, 10))

        # Get pre and post disaster images
        pre_im = plt.imread(os.path.join(self.img_dir, predf.img_name.unique()[0]))
        post_im = plt.imread(os.path.join(self.img_dir, postdf.img_name.unique()[0]))

        # Plot pre-disaster image
        axes[0].imshow(pre_im)
        axes[0].set_title('Pre Disaster')
        axes[0].axis('off')

        # Plot heatmap on pre-disaster image
        pre_binary_heatmap = self.generate_binary_heatmap(predf, pre_im.shape)
        axes[1].imshow(pre_binary_heatmap, cmap='gray')
        axes[1].set_title('Pre Disaster Binary Heatmap')
        axes[1].axis('off')

        # Plot post-disaster image
        axes[2].imshow(post_im)
        axes[2].set_title('Post Disaster')
        axes[2].axis('off')

        # Plot heatmap on post-disaster image
        post_binary_heatmap = self.generate_binary_heatmap(postdf, post_im.shape)
        axes[3].imshow(post_binary_heatmap, cmap='gray')
        axes[3].set_title('Post Disaster Binary Heatmap')
        axes[3].axis('off')

        plt.suptitle(f'{disaster}_{imid}', fontsize=14, fontweight='bold')
        plt.show()

        self.pre_binary_heatmap = pre_binary_heatmap
        self.post_binary_heatmap = post_binary_heatmap

    def generate_binary_heatmap(self, df, image_shape):
        """
        Generate a binary heatmap based on the object coordinates in the dataframe.

        :param df: DataFrame containing object coordinates
        :param image_shape: Shape of the image
        :return: Binary heatmap image
        """
        heatmap = np.zeros(image_shape[:2], dtype=np.uint8)

        for _, row in df.iterrows():
            poly = shapely.wkt.loads(row['pixwkt'])
            pts = np.array(poly.exterior.coords)
            pts = np.expand_dims(pts, axis=1).astype(np.int32)
            cv2.fillPoly(heatmap, [pts], (255))

        # Convert to binary (0 or 255)
        binary_heatmap = np.where(heatmap > 0, 255, 0)

        return binary_heatmap

    def pre_post_split(self, imid, disaster):
        predf = self.anndf[(self.anndf["type"] == 'pre') & (self.anndf["img_id"] == imid) & (self.anndf["disaster"] == disaster)]
        print("Predf:", predf)
        postdf = self.anndf[(self.anndf["type"] == 'post') & (self.anndf["img_id"] == imid) & (self.anndf["disaster"] == disaster)]
        print("Postdf:", postdf)
        return predf, postdf

    def calculate_similarity(self, pre_heatmap, post_heatmap):
        # Convert heatmaps to grayscale
        pre_gray = cv2.cvtColor(pre_heatmap, cv2.COLOR_BGR2GRAY)
        post_gray = cv2.cvtColor(post_heatmap, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        ssim_score = ssim(pre_gray, post_gray)
        
        # Calculate MSE
        mse_score = mean_squared_error(pre_gray, post_gray)
        
        return ssim_score, mse_score


# Example usage
data_dir = ""
folder = "train"
img_dir = os.path.join(data_dir, folder, 'images')
lbl_dir = os.path.join(data_dir, folder, 'labels')
xview = XView2(img_dir, lbl_dir)
xview.view_pre_post(disaster='santa-rosa-wildfire', imid='00000060')

ssim_score, mse_score = xview.calculate_similarity(xview.pre_binary_heatmap, xview.post_binary_heatmap)
print("SSIM Score:", ssim_score)
print("MSE Score:", mse_score)
