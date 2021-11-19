import json

class CocoReader:
    coco_json = None
    bbs = dict()

    def __init__(self, fpath):
        with open(fpath) as json_file:
            self.coco_json = json.load(json_file)

        self.imgs_dict = dict()
        for img in self.coco_json['images']:
            hash = img['file_name'].split('_')[1]
            self.bbs[hash] = []
            self.imgs_dict[img['id']] = hash

        # iterate over annotations and write media
        lared_examples = list()
        for annot in self.coco_json['annotations']:
            hash = self.imgs_dict[annot['image_id']]
            # hash = annot_img['file_name'].split('_')[1]
            bbox = annot['bbox']
            self.bbs[hash] = [int(e) for e in bbox]

    def __len__(self):
        return len(self.bbs)

    def __getitem__(self, key):
        return self.bbs[key]

    def __contains__(self, key):
        return key in self.bbs
