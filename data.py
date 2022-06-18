import torch
import torchvision

from detection import transforms as T


class DisplayDataset(torch.utils.data.Dataset):
    def __init__(self, background='synthetic/empty.png', digit_map='synthetic/digits.png', augment=False):
        # This should be an image of the display with all segments turned off
        self.background = torchvision.transforms.functional.convert_image_dtype(
            torchvision.io.read_image(background)[:3], torch.float32
        )
        # For every pixel this indicated which segment of which digit it corresponds to (or if it is part of the
        # background). The first segment of the first digit (left to right) will have index 1. With seven segments
        # for each digit the second digit will have indices 8-14, the third will have indices 15-21 and so on
        self.digit_map = torchvision.io.read_image(digit_map)[0]
        self.num_digits = self.digit_map.max().item() // 7

        # taken from https://en.wikipedia.org/wiki/Seven-segment_display#Hexadecimal
        self.segment_encodings = (
            0b0111111,
            0b0000110,
            0b1011011,
            0b1001111,
            0b1100110,
            0b1101101,
            0b1111101,
            0b0000111,
            0b1111111,
            0b1101111,
        )

        # potentially add some variation to the data in terms of changing scale, jitter and overall appearance
        transforms = [T.ScaleJitter(target_size=(100, 200), scale_range=((0.8, 1.2) if augment else (1, 1)))]
        if augment:
            transforms.append(T.RandomPhotometricDistort())
        self.transform = T.Compose(transforms)

    def __len__(self):
        return 1_000_000

    def __getitem__(self, number):
        number = int(number)  # cast to int in case input is a torch.Tensor

        if not (0 <= number < 1_000_000):
            raise ValueError('only numbers in the range [0, 1_000_000) can be displayed')

        image = self.background.clone()
        height, width = image.shape[-2:]

        target = {
            'boxes': torch.zeros((self.num_digits, 4), dtype=torch.float32),
            'labels': torch.zeros(self.num_digits, dtype=torch.int64),
            'image_id': torch.tensor(number, dtype=torch.int64),
            'area': torch.zeros(self.num_digits, dtype=torch.float32),
            'iscrowd': torch.zeros(self.num_digits, dtype=torch.uint8),
            'masks': torch.zeros((self.num_digits, height, width), dtype=torch.uint8),
        }

        number_string = str(number).zfill(self.num_digits)
        for digit_index, digit in enumerate(number_string):
            target['labels'][digit_index] = int(digit) + 1

            digit_map = target['masks'][digit_index]
            for segment_index in range(7):
                visible = bool(self.segment_encodings[int(digit)] & (1 << segment_index))
                if visible:
                    digit_map[self.digit_map == (digit_index * 7 + segment_index + 1)] = 1

            coords_y, coords_x = torch.where(digit_map)
            target['boxes'][digit_index, 0] = coords_x.min()
            target['boxes'][digit_index, 1] = coords_y.min()
            target['boxes'][digit_index, 2] = coords_x.max()
            target['boxes'][digit_index, 3] = coords_y.max()

            target['area'][digit_index] = (coords_x.max() - coords_x.min()) * (coords_y.max() - coords_y.min())

        for digit_index in range(self.num_digits):
            mask = target['masks'][digit_index] == 1
            image[(mask)[None].expand(3, -1, -1)] = 0.1

        return self.transform(image, target)