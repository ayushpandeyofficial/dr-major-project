# from src.dataset import DRDataset
# import numpy as np

# image_dataset=DRDataset(csv_path="challenge_data/train.csv",transforms=None)

# g_mean_sum=0
# g_std_sum=0
# for img,_ in image_dataset:
#     img_np=np.array(img)
#     g_mean_sum+=img_np[2].mean()
#     g_std_sum+=img_np.std()
    
# print(g_mean_sum/len(image_dataset),g_std_sum/len(image_dataset))
# 25.828141795176006 54.48457502453686
# 26.48134420719688 54.48457502453686
# 26.958662712534895 54.48457502453686

    
# print(r_mean_sum/len(image_dataset),r_std_sum/len(image_dataset))=>124.86798522534032 19.97025197702598
#print(g_mean_sum/len(image_dataset),g_std_sum/len(image_dataset))=>130.40536511479596 19.97025197702598
#print(b_mean_sum/len(image_dataset),b_std_sum/len(image_dataset))=>130.40536511479596  19.97025197702598

# print(124.86798522534032/255.0,130.40536511479596/255.0,130.40536511479596 /255.0)
# 0.48967837343270715 0.5113935886854744 0.5113935886854744

# print(19.97025197702598/255.0)
# 0.07831471363539601

# print(25.828141795176006/255.0,26.48134420719688/255.0, 26.958662712534895/255.0,54.48457502453686/255.0)
# 0.10128683056931767 0.10384840865567405 0.10572024593150939 0.21366500009622297