import PIL
import cv2
import numpy as np


def load_file_path_txt(file_path):
    img_list = []
    gt_list = []
    fov_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # read a line
            if not lines:
                break
            img,gt,fov = lines.split(' ')
            img_list.append(img)
            gt_list.append(gt)
            fov_list.append(fov)
    return img_list,gt_list,fov_list

def readImg(img_path):
    """
    When reading local image data, because the format of the data set is not uniform,
    the reading method needs to be considered.
    Default using pillow to read the desired RGB format img
    """
    img_format = img_path.split(".")[-1]
    try:
        #在win下读取tif格式图像在转np的时候异常终止，暂时没找到合适的读取方式，Linux下直接用PIl读取无问题
        img = PIL.Image.open(img_path)
    except Exception as e:
        ValueError("Reading failed, please check path of dataset,",img_path)
    return img

def load_data(data_path_list_file):
    print('\033[0;33mload data from {} \033[0m'.format(data_path_list_file))
    img_list, gt_list, fov_list = load_file_path_txt(data_path_list_file)
    imgs = None
    groundTruth = None
    FOVs = None
    for i in range(len(img_list)):
        img = np.asarray(readImg(img_list[i]))
        gt = np.asarray(readImg(gt_list[i]))
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        fov = np.asarray(readImg(fov_list[i]))
        if len(fov.shape) == 3:
            fov = fov[:, :, 0]

        imgs = np.expand_dims(img, 0) if imgs is None else np.concatenate((imgs, np.expand_dims(img, 0)))
        groundTruth = np.expand_dims(gt, 0) if groundTruth is None else np.concatenate(
            (groundTruth, np.expand_dims(gt, 0)))
        FOVs = np.expand_dims(fov, 0) if FOVs is None else np.concatenate((FOVs, np.expand_dims(fov, 0)))

    assert (np.min(FOVs) == 0 and np.max(FOVs) == 255)
    assert ((np.min(groundTruth) == 0 and (
                np.max(groundTruth) == 255 or np.max(groundTruth) == 1)))  # CHASE_DB1数据集GT图像为单通道二值（0和1）图像
    if np.max(groundTruth) == 1:
        print("\033[0;31m Single channel binary image is multiplied by 255 \033[0m")
        groundTruth = groundTruth * 255

def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs

def data_preprocess(data_path_list):
    train_imgs_original, train_masks, train_FOVs = load_data(data_path_list)
    # save_img(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train.png')#.show()  #check original train imgs

    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks//255
    train_FOVs = train_FOVs//255
    return train_imgs, train_masks, train_FOVs

def get_dataloaderV2(args):
    """
    该函数加载数据集所有图像到内存，并创建训练样本提取位置的索引，所以占用内存量较少，
    测试结果表明，相比于上述原始的get_dataloader方法并不会降低训练效率
    """
    # imgs_train, masks_train, fovs_train = data_preprocess(data_path_list = args.train_data_path_list)
    imgs_train, masks_train, fovs_train = data_preprocess(data_path_list = './unet/prepare_dataset/STARE/train.txt')
    print(imgs_train,masks_train,fovs_train)


    # patches_idx = create_patch_idx(fovs_train, args)
    #
    # train_idx,val_idx = np.vsplit(patches_idx, (int(np.floor((1-args.val_ratio)*patches_idx.shape[0])),))
    #
    # train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,train_idx,mode="train",args=args)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size,
    #                           shuffle=True, num_workers=0)
    #
    # val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size,
    #                         shuffle=False, num_workers=0)
    #
    # # Save some samples of feeding to the neural network
    # if args.sample_visualization:
    #     visual_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
    #     visual_loader = DataLoader(visual_set, batch_size=1,shuffle=True, num_workers=0)
    #     N_sample = 50
    #     visual_imgs = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
    #     visual_masks = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
    #
    #     for i, (img, mask) in tqdm(enumerate(visual_loader)):
    #         visual_imgs[i] = np.squeeze(img.numpy(),axis=0)
    #         visual_masks[i,0] = np.squeeze(mask.numpy(),axis=0)
    #         if i>=N_sample-1:
    #             break
    #     save_img(group_images((visual_imgs[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
    #             join(args.outf, args.save, "sample_input_imgs.png"))
    #     save_img(group_images((visual_masks[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
    #             join(args.outf, args.save,"sample_input_masks.png"))
    # return train_loader,val_loader

if __name__ == '__main__':
    get_dataloaderV2(args=1)
