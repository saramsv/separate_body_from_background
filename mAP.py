import argparse
import cv2
import numpy as np
import glob



parser = argparse.ArgumentParser()
parser.add_argument("--ground_truth_imgs", type = str)
parser.add_argument("--pred_imgs", type = str)

args = parser.parse_args()
imgs_path = args.ground_truth_imgs
preds_path = args.pred_imgs


imgs = glob.glob( imgs_path + "*.JPG"  ) + glob.glob( imgs_path + "*.jpg"  ) + glob.glob( imgs_path + "*.png"  ) +  glob.glob( imgs_path + "*.jpeg"  )
imgs.sort()
preds  = glob.glob( preds_path + "*.JPG"  ) + glob.glob( preds_path + "*.jpg"  ) + glob.glob( preds_path + "*.png"  ) +  glob.glob( preds_path + "*.jpeg"  )
preds.sort()


assert len(imgs) == len(preds)
for img, pred in zip(imgs, preds):
    assert(img.split('/')[-1].split(".")[0] == pred.split('/')[-1].split(".")[0])

zipped = zip(imgs, preds)
acc = []
num = len(imgs)

for pair in zipped:
    img = cv2.imread(pair[0])
    pred = cv2.imread(pair[1])
    intersection = np.array(img) == np.array(pred)
    intersection = np.sum(intersection)
    acc.append(intersection / float(img.size))

## Average accuracy is
print "average accuracy = ", np.sum(np.array(acc)) / num




