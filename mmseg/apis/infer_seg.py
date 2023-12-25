from mmseg.apis import MMSegInferencer

mm_config = 'D:/git/sem-seg/semi-cd/configs/swin/swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_mydata-512x512.py'
checkpoint_path = 'D:/git/sem-seg/semi-cd/logs/semi_seg/iter_36000.pth'

# img = dict(img_path_l=['E:/changeDectect/semi_seg/test/seg_u/2014.tif'],
#            img_path_u=['E:/changeDectect/semi_seg/test/seg_u/2014.tif'])
img = 'E:/changeDectect/seg_label_AB/A/2.tif'
infer = MMSegInferencer(
        model=mm_config,
        weights=checkpoint_path,
        device='cuda:0'
)

ai_results = infer(img, 
                   show=True)
print(ai_results)
