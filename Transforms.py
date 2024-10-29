import numpy as np
import cv2
from PIL import Image

class BilinearWarp:
    def __init__(self, src_points, dst_points):
        self.src_points = src_points
        self.dst_points = dst_points

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        matrix = cv2.getAffineTransform(self.src_points, self.dst_points)
        warped_img = cv2.warpAffine(img_np, matrix, (img_np.shape[1], img_np.shape[0]))
        return Image.fromarray(warped_img)

class OpticalFlowTransform:
    def __init__(self, flow):
        self.flow = flow

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h))) + self.flow.reshape(-1, 2)
        flow_map = np.clip(flow_map, 0, (h-1, w-1)).astype(np.float32)
        warped_img = cv2.remap(img_np, flow_map[:, 1], flow_map[:, 0], interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(warped_img)

class ElasticTransform:
    def __init__(self, alpha_x=5, alpha_y=5, sigma_x=0.5, sigma_y=0.5):
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # Create meshgrid for coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Create fixed displacement fields for x and y
        dx = self.alpha_x * np.sin(np.linspace(-3.14, 3.14, h).reshape(-1, 1) / self.sigma_x)
        dy = self.alpha_y * np.cos(np.linspace(-3.14, 3.14, w).reshape(1, -1) / self.sigma_y)

        # Apply displacements
        x_distorted = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_distorted = np.clip(y + dy, 0, h - 1).astype(np.float32)

        # Remap the image
        distorted_img = cv2.remap(img_np, x_distorted, y_distorted, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(distorted_img)

class InvertSelectedChannels:
    def __init__(self, channels_to_invert=[0, 1, 2]):  # Default to invert R, G, and B
        self.channels_to_invert = channels_to_invert

    def __call__(self, img: Image.Image) -> Image.Image:
        channels = img.split()
        inverted_channels = []
        for i in range(len(channels)):
            if i in self.channels_to_invert:
                inverted_channel = Image.eval(channels[i], lambda x: 255 - x)
                inverted_channels.append(inverted_channel)
            else:
                inverted_channels.append(channels[i])

        return Image.merge(img.mode, inverted_channels)
class ComplexWaveDistortion:
    def __init__(self, amplitudes=[5, 2, 3], frequencies=[200, 100, 50], vertical_phases=[0, 0, 0], horizontal_phases=[0, 0, 0]):
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.vertical_phases = vertical_phases
        self.horizontal_phases = horizontal_phases

    def __call__(self, img: Image.Image) -> Image.Image:
        frame = np.array(img)
        rows, cols, _ = frame.shape

        # Calculate vertical and horizontal offsets
        y_indices = np.arange(rows)
        vertical_offsets = np.array([
            amplitude * np.sin(2 * np.pi * (y_indices / frequency + phase))
            for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies, self.vertical_phases)
        ])
        total_vertical_offset = np.sum(vertical_offsets, axis=0).astype(int)

        x_indices = np.arange(cols)
        horizontal_offsets = np.array([
            amplitude * np.sin(2 * np.pi * (x_indices / frequency + phase))
            for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies, self.horizontal_phases)
        ])
        total_horizontal_offset = np.sum(horizontal_offsets, axis=0).astype(int)

        x, y = np.meshgrid(x_indices, y_indices)
        new_x = np.clip(x + total_horizontal_offset, 0, cols - 1)
        new_y = np.clip(y + total_vertical_offset[:, np.newaxis], 0, rows - 1)

        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)

        distorted_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(distorted_frame)
class SwapChannels:
    def __init__(self, mapping=[0, 1, 2]):
        self.mapping = mapping

    def __call__(self, img: Image.Image) -> Image.Image:
        channels = img.split()
        # Rearrange the channels according to the mapping
        swapped_channels = [channels[i] for i in self.mapping]
        return Image.merge(img.mode, swapped_channels)