#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image

import labelme
import numpy as np
from PIL import Image

#prefix = 'Jun_20_mice_1+2_3_'
prefix = ''                                         # change this according to the name
#--------------------------------------------------------

def shapes_to_label_separate(img_shape, shapes, label_name_to_value, type='class'): # by vananh
    assert type in ['class', 'instance']

    cls = np.zeros(img_shape[:2], dtype=np.int32)
    if type == 'instance':
        ins = np.zeros((img_shape[0],img_shape[1],len(shapes)), dtype=np.int32)
        instance_names = ['_background_']
    for count, shape in enumerate(shapes):
        points = shape['points']
        label = shape['label']
        shape_type = shape.get('shape_type', None)
        if type == 'class':
            cls_name = label
        elif type == 'instance':
            cls_name = label.split('-')[0]
            if label not in instance_names:
                instance_names.append(label)
            ins_id = instance_names.index(label)
        cls_id = label_name_to_value[cls_name]
        mask = labelme.utils.shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        if type == 'instance':
            ins[mask,count] = ins_id

    if type == 'instance':
        return cls, ins
    return cls


#--------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)

    print('Creating dataset:', args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = labelme.utils.label_colormap(255)

    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            os.makedirs(osp.join(args.output_dir, prefix + base))
            #os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
            os.makedirs(osp.join(args.output_dir + '/' + prefix + base, 'images'))
            os.makedirs(osp.join(args.output_dir + '/' + prefix + base, 'masks'))
            out_img_file = osp.join(
                args.output_dir, prefix + base, 'images', prefix + base + '.jpg')

            data = json.load(f)

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(PIL.Image.open(img_file))
            img_new = img[:, 60:1140]                                               # change this according to image size
            PIL.Image.fromarray(img_new).save(out_img_file)

            #cls, ins = labelme.utils.shapes_to_label_separate(
            cls, ins=shapes_to_label_separate(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
                type='instance',
            )
            ins[cls == -1] = 0  # ignore it.

            # instance label
            for i in range(ins.shape[2]):
                out_insp_file = osp.join(
                    args.output_dir, prefix + base, 'masks', prefix + base + '_' + str(i) + '.png')

                array = (ins[:, 60:1140, i]*255).astype(np.uint8)                   # change this according to image size

                img = Image.fromarray(array)
                img.save(out_insp_file)

if __name__ == '__main__':
    main()
