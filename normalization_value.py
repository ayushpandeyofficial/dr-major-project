# from src.dataset import DRDataset
# import numpy as np

# image_dataset=DRDataset(csv_path="data/train.csv",transforms=None)

# g_mean_sum=0
# g_std_sum=0
# for img,_ in image_dataset:
#     img_np=np.array(img)
#     g_mean_sum+=img_np[2].mean()
#     g_std_sum+=img_np.std()
    
#print(r_mean_sum/len(image_dataset),r_std_sum/len(image_dataset))=>124.86798522534032 19.97025197702598
#print(g_mean_sum/len(image_dataset),g_std_sum/len(image_dataset))=>130.40536511479596 19.97025197702598
#print(b_mean_sum/len(image_dataset),b_std_sum/len(image_dataset))=>130.40536511479596  19.97025197702598

# print(124.86798522534032/255.0,130.40536511479596/255.0,130.40536511479596 /255.0)
# 0.48967837343270715 0.5113935886854744 0.5113935886854744

# print(19.97025197702598/255.0)
# 0.07831471363539601