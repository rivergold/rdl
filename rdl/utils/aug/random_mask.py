import math
from PIL import Image, ImageDraw
import numpy as np
import cv2


# ******
# BoxMask Generator
# ******
class BoxMaskGenerator:
    def __init__(self,
                 max_num=4,
                 box_w_range=(0, 0),
                 box_h_range=(0, 0),
                 mask_value=255) -> np.ndarray:
        self.max_num = max_num
        self.box_w_range = box_w_range
        self.box_h_range = box_h_range
        self.mask_value = mask_value

    def gen_mask(self, mask_w, mask_h):
        if self.max_num == 1:
            num_parts = 1
        else:
            num_parts = np.random.randint(1, self.max_num)
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

        for _ in range(num_parts):
            bbox_w = int(np.random.uniform(*self.box_w_range))
            bbox_h = int(np.random.uniform(*self.box_h_range))
            x1 = int(np.random.uniform(0, mask_w - bbox_w))
            y1 = int(np.random.uniform(0, mask_h - bbox_h))
            mask[y1:y1 + bbox_h, x1:x1 + bbox_w] = self.mask_value
        return mask


def box_erase(img,
              max_num=1,
              box_w_range=(0, 0),
              box_h_range=(0, 0),
              fill_value=255):
    img_h, img_w = img.shape[:2]
    box_generator = BoxMaskGenerator(max_num=max_num,
                                     box_w_range=box_w_range,
                                     box_h_range=box_h_range,
                                     mask_value=255)
    mask = box_generator.gen_mask(img_w, img_h)

    img[mask == 255] = fill_value
    # cv2.imwrite('./debug-mask.jpg', img)
    return img


# ******
# BrushMask Generator
# ******
class BrushMaskGenerator:
    def __init__(self,
                 max_num=1,
                 num_vertex_range=[4, 12],
                 move_angle=2 * math.pi / 15,
                 mean_move_angle=(2 * math.pi / 5),
                 vertex_line_width_range=[12, 40]):
        """Mask has M part, each part has N vertex
        Vertex -> Part -> Mask
        Keyword Arguments:
            max_num_part {int} -- [description] (default: {4})
            num_vertex_range {list} -- [description] (default: {[4, 12]})
            move_angle_range {list} -- [description] (default: {[0, 2 * math.pi / 15]})
            move_mean_angle {tuple} -- [description] (default: {(2 * math.pi / 2)})
            vertex_line_width_range {list} -- [description] (default: {[12, 40]})
        """
        self.max_num = max_num
        self.num_vertex_range = num_vertex_range
        self.move_angle = move_angle
        self.mean_move_angle = mean_move_angle
        self.vertex_line_width_range = vertex_line_width_range

    def gen_mask(self, mask_w, mask_h):
        if self.max_num > 1:
            num_parts = np.random.randint(1, self.max_num)
        else:
            num_parts = 1
        mean_radius = math.sqrt(math.pow(mask_w, 2) + math.pow(mask_h, 2)) / 8

        mask = Image.new('L', (mask_w, mask_h), 0)
        mask_draw = ImageDraw.Draw(mask)

        for part_idx in range(num_parts):
            num_vertexes = np.random.randint(self.num_vertex_range[0],
                                             self.num_vertex_range[1])
            min_move_angle = self.mean_move_angle - np.random.uniform(
                0, self.move_angle)
            max_move_angle = self.mean_move_angle + np.random.uniform(
                0, self.move_angle)

            move_angles, vertexes = [], []
            for vertex_idx in range(num_vertexes):
                if vertex_idx % 2 == 0:
                    move_angle = 2 * math.pi - np.random.uniform(
                        min_move_angle, max_move_angle)
                else:
                    move_angle = np.random.uniform(min_move_angle,
                                                   max_move_angle)
                move_angles.append(move_angle)

            # Set start vertex
            vertexes.append(
                (np.random.randint(0, mask_w), np.random.randint(0, mask_h)))
            for vertex_idx in range(num_vertexes):
                # Calculate radius
                r = np.clip(
                    np.random.normal(loc=mean_radius, scale=mean_radius // 2),
                    0, 2 * mean_radius)
                x_next = np.clip(
                    vertexes[-1][0] + r * math.cos(move_angles[vertex_idx]), 0,
                    mask_w)
                y_next = np.clip(
                    vertexes[-1][1] + r * math.sin(move_angles[vertex_idx]), 0,
                    mask_h)
                vertexes.append((int(x_next), int(y_next)))

            # Draw line
            line_width = int(
                np.random.uniform(self.vertex_line_width_range[0],
                                  self.vertex_line_width_range[1]))
            mask_draw.line(vertexes, fill=255, width=line_width)
            # Draw ellipse
            for vertex in vertexes:
                mask_draw.ellipse(
                    (vertex[0] - line_width // 2, vertex[1] - line_width // 2,
                     vertex[0] + line_width // 2, vertex[1] + line_width // 2),
                    fill=255)

        mask = np.array(mask)
        # Horizontal flip
        if np.random.uniform(0, 1) > 0.5:
            mask = cv2.flip(mask, 1)
        # Vertical flip
        if np.random.uniform(0, 1) > 0.5:
            mask = cv2.flip(mask, 0)
        return mask  # Value [0, 255]


def brush_erase(img,
                max_num=1,
                num_vertex_range=[4, 12],
                move_angle=2 * math.pi / 15,
                mean_move_angle=(2 * math.pi / 5),
                vertex_line_width_range=[12, 40],
                fill_value=255):
    img_h, img_w = img.shape[:2]
    brush_generator = BrushMaskGenerator(
        max_num=max_num,
        num_vertex_range=num_vertex_range,
        move_angle=move_angle,
        mean_move_angle=mean_move_angle,
        vertex_line_width_range=vertex_line_width_range)
    mask = brush_generator.gen_mask(img_w, img_h)
    img[mask == 255] = fill_value
    return img


if __name__ == '__main__':
    # ******
    # Box mask
    # ******
    # box_mask_generator = BoxMaskGenerator(max_num=1,
    #                                       box_w_range=(5, 50),
    #                                       box_h_range=(5, 50),
    #                                       mask_value=255)
    # for idx in range(100):
    #     mask = box_mask_generator.gen_mask(256, 256)
    #     print(mask.shape, mask.dtype)
    #     cv2.imwrite(f"./out/tmp/mask-{idx}.jpg", mask)

    # ******
    # Brush mask
    # ******
    brush_mask_generator = BrushMaskGenerator(max_num=1,
                                              num_vertex_range=[4, 12],
                                              move_angle=2 * math.pi / 15,
                                              mean_move_angle=(2 * math.pi /
                                                               5),
                                              vertex_line_width_range=[12, 40])
    for idx in range(100):
        mask = brush_mask_generator.gen_mask(256, 256)
        print(mask.shape, mask.dtype)
        cv2.imwrite(f"./out/tmp/mask-{idx}.jpg", mask)
