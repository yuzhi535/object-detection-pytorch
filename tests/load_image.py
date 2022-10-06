from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

json_path = 'instances_val2017.json'
img_path = 'val2017'

# load coco data
coco = COCO(json_path)

# get all image list info
ids = list(sorted(coco.imgs.keys()))
print(f'number of images: {len(ids)}')

# get all coco classes  (80 classes)
coco_classes = dict([(v['id'], v['name']) for k, v in coco.cats.items()])

# first three images
for img_id in id[:3]:
    annotion_ids = coco.getAnnIds(imgIds=img_id)
    targets = coco.loadAnns(annotion_ids)

    path = coco.loadImgs(img_id)[0]['file_name']

    image = Image.open(os.path.join(img_path, path))
    draw = ImageDraw.Draw(image)

    for target in targets:
        x, y, w, h = target['bbox']
        x1, y1, x3, y3 = x, y, x+w, y+h
        draw.rectangle([x1, y1, x3, y3], outline='red', width=3)
        draw.text((x1, y1), coco_classes[target['category_id']], fill='red')

    plt.imshow(image)
    plt.show()
