'''
Dictionary to map a dataset_str to the location of the data file containing
gt labels, annotations, etc.
'''

MAPPING = {
    'cocodots_train': './data/coco_dots_0_0_train2017.json',
    'cocodots_val': './data/coco_dots_0_0_val2017.json',
    'cocodots_val_mini': './data/coco_dots_0_0_val2017_mini.json',
    'mazes_train': './data/mazes_train.json',
    'mazes_val': './data/mazes_val.json'
}