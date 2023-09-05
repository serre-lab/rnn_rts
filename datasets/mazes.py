import os
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision


class Mazes(Dataset):

    def __init__(self, anns_file, maze_dir, subset=1.0, shuffle=False):

        # Init arguments
        self.anns_file = anns_file
        self.maze_dir = maze_dir  # folder containing the base mazes
        self.subset = subset
        self.shuffle = shuffle
        self.frame_thickness = 3
        self.cue_size = 6  # cues are 6x6 squares in the final mazes

        # Read list of stimulus records
        self.data = self.read_anns(anns_file, shuffle=self.shuffle, subset=self.subset)

        # Load matrix of mazes
        self.mazes = np.load(os.path.join(self.maze_dir, self.data['base_mazes_fn']))

    def read_anns(self, file, shuffle, subset):
        with open(file, 'rb') as f:
            data = json.load(f)

        serrelab_anns = data["serrelab_anns"]
        if subset != 1:
            pos = [x for x in serrelab_anns if x["same"] == 1]
            neg = [x for x in serrelab_anns if x["same"] == 0]
            serrelab_anns = random.sample(pos, int(subset * len(pos))) + random.sample(neg, int(subset * len(neg)))

        if shuffle:
            random.shuffle(serrelab_anns)

        data["serrelab_anns"] = serrelab_anns

        return data

    def __len__(self):
        return len(self.data["serrelab_anns"])

    def __getitem__(self, index):
        record = self.data['serrelab_anns'][index]
        record['dataset_index'] = index
        return self.get(record)

    def get(self, record):

        try:

            # Get base maze
            maze = self.mazes[record['id']].copy()
            maze = np.transpose(maze, (1, 2, 0))

            # Remove red and green points from original data
            y_values_red, x_values_red = np.where(np.all(maze == (1, 0, 0), axis=-1))
            y_values_green, x_values_green = np.where(np.all(maze == (0, 1, 0), axis=-1))

            maze[y_values_red, x_values_red, :] = (1.0, 1.0, 1.0)
            maze[y_values_green, x_values_green, :] = (1.0, 1.0, 1.0)

            # Crop and pad
            maze = maze[record['row_start']:record['row_stop'], record['col_start']:record['col_stop'], :]
            maze = np.pad(maze, (
                (self.frame_thickness, self.frame_thickness), (self.frame_thickness, self.frame_thickness), (0, 0)))

            # Flips and rotations
            if record['horizontal_flip']:
                maze = np.fliplr(maze)
            if record['vertical_flip']:
                maze = np.flipud(maze)

            maze = np.rot90(maze, k=record['rotation_k'])

            # Insert blocks (extra walls)
            maze[record['blocks_y'], record['blocks_x'], :] = (0, 0, 0)  # note that they will be in original resolution

            # Resize
            maze = cv2.resize(maze * 255, dsize=(record['target_size'], record['target_size']),
                              interpolation=cv2.INTER_NEAREST)
            maze = maze / 255

            # Add cue channel
            dot_channel = np.zeros_like(maze[:, :, 0])[:, :, np.newaxis]
            dot_channel[record['fixation_top_y']:record['fixation_top_y'] + self.cue_size,
            record['fixation_left_x']:record['fixation_left_x'] + self.cue_size, :] = 1
            dot_channel[record['cue_top_y']:record['cue_top_y'] + self.cue_size,
            record['cue_left_x']:record['cue_left_x'] + self.cue_size, :] = 1
            maze = np.concatenate([maze, dot_channel], axis=2)

            # To tensor
            maze = torch.Tensor(np.transpose(maze, (2, 0, 1)))

            # Gather
            return_dict = {
                "image": maze,
                "label": record['same'],
                "id": record['id'],
                "index": record['dataset_index'],
                "cue_x": record['cue_x'],
                "cue_y": record['cue_y'],
                "fixation_x": record['fixation_x'],
                "fixation_y": record['fixation_y']

            }

            return return_dict

        except Exception as e:
            print(
                "Error in getting sample with index {}, id {}, and serre_lab sample {}: {}".format(
                    record['dataset_index'], record['id'], record["serrelab_sample"], str(e)))
            print("Sampling new random image")
            new_idx = random.choice(list(range(len(self))))
            return_dict = self.__getitem__(new_idx)
            return return_dict

    def tensor_to_image(self, img_w_dots, draw_cues=True):
        img = img_w_dots.detach().cpu().numpy()
        dots = img[3, :, :] * 255
        img = np.transpose(img[0:3, :, :], (1, 2, 0)) * 255
        img = img.astype('int32')
        if draw_cues:
            img[dots == 255] = (255, 0, 0,)
        return img

