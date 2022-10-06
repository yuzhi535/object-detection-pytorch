from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# load coco validation annotation data
coco_true = COCO('instances_val2017.json')
# load prediction results
coco_pre = coco_true.loadRes('instances_val2017_results.json')

coco_evaluator = COCOeval(coco_true, coco_pre, 'bbox')
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()
