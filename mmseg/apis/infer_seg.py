from mmseg.apis import MMSegInferencer
import numpy as np
import cv2
import matplotlib.pyplot as plt
mm_config = 'D:/git/sem-seg/semi-cd/configs/semi/swin-tiny-40k-mydata-512x512.py'
checkpoint_path = 'D:/git/sem-seg/semi-cd/logs/semi_seg/iter_36000.pth'

# img1 = 'E:/changeDectect/semi_seg/train/seg_u/213.tif'
# img2 = 'E:/changeDectect/semi_seg/train/seg_u/1213.tif'
# img1 = 'E:/changeDectect/semi_seg/vis_image/123.tif'
img1 = 'E:/changeDectect/train_with_seg/train/A/123.tif'
img2 = 'E:/changeDectect/train_with_seg/train/B/123.tif'

def edge_connect(segmentation_map, connectivity_threshold=0.9):
    # 识别边缘区域
    edges = cv2.Canny(segmentation_map, 0, 1)

    # 进行边缘连接
    h, w = segmentation_map.shape
    for i in range(h):
        for j in range(w):
            if edges[i, j] == 1:
                # 基于像素的边缘连接
                neighbors = []
                if i > 0 and edges[i-1, j] == 1:
                    neighbors.append(segmentation_map[i-1, j])
                if j > 0 and edges[i, j-1] == 1:
                    neighbors.append(segmentation_map[i, j-1])
                if i < h-1 and edges[i+1, j] == 1:
                    neighbors.append(segmentation_map[i+1, j])
                if j < w-1 and edges[i, j+1] == 1:
                    neighbors.append(segmentation_map[i, j+1])

                # 如果相邻像素属于不同的类别，则连接
                if neighbors:
                    if sum(neighbors) / len(neighbors) > connectivity_threshold:
                        segmentation_map[i, j] = round(sum(neighbors) / len(neighbors))

    return segmentation_map



inference = MMSegInferencer(
        model=mm_config,
        weights=checkpoint_path,
        device='cuda:0'
)

img1_data = cv2.imread(img1)

change_map = cv2.imread('E:/changeDectect/train_with_seg/vis/123.png', -1)
change_map[change_map == 255] = 1


ai_results1 = inference(img1, show=True)
pred_form = ai_results1['predictions']

ai_results2 = inference(img2, show=True)
pred_to = ai_results2['predictions']

pred_form[change_map == 0] = 0
pred_to[change_map == 0] = 0

before = np.unique(pred_form)
after = np.unique(pred_to)

print(f'before: {before}')
print(f'after: {after}')

# plt.subplot(1, 2, 1)
# plt.imshow(pred_form)
# plt.subplot(1, 2, 2)
# plt.imshow(pred_to)
# plt.show()

# pred1 = np.expand_dims(pred_form, axis=2)
# a = edge_connect(np.uint8(pred_form), connectivity_threshold=0.5)


# img1_data = cv2.imread(img1)
# pred1 = np.expand_dims(pred_form, axis=2)
# mask = pred_form.copy()
# mask[mask != 2] = 0
# three_channel_array = np.stack([mask] * 3, axis=2)
# img1_data[three_channel_array == 0] = 0
# plt.imsave("E:/changeDectect/semi_seg/vis_image/cls.png", img1_data)

# plt.subplot(1, 2, 1)
# plt.imshow(a)
# plt.subplot(1, 2, 2)
# plt.imshow(pred_to)
# plt.show()

# img1_data = cv2.imread(img1)
# img2_data = cv2.imread(img2)
# cv2.imwrite("E:/changeDectect/semi_seg/vis_image/from.png", img1_data)
# cv2.imwrite("E:/changeDectect/semi_seg/vis_image/to.png", img2_data)

