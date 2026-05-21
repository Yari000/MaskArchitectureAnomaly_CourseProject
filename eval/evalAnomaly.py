# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from matplotlib import pyplot as plt
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        #Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    anomaly_score_list = []
    anomaly_score_list_logit = []
    anomaly_score_list_entropy = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    # Import the model architecture 
    # NUM_CLASSES is set to 20 cuz the model was trained on Cityscapes
    model = ERFNet(NUM_CLASSES)

    # Load the model weights and set the model to evaluation mode.
    # The weights are loaded using a custom function that handles cases where not all elements of the state dictionary are present in the model.
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    # Upload the images, apply unsqueeze to add a batch dimension, and permute the dimensions to match 
    # the expected input format of the model (batch_size, channels, height, width).

    # CONTROLLO
    pattern = os.path.expanduser(str(args.input[0]))
    print("Pattern espanso:", pattern)
    paths = glob.glob(pattern)
    print(f"Trovati {len(paths)} file")
    if paths:
       print("Esempio:", paths[0])
    else:
    # Prova a listare la directory padre per capire cosa c'è
       parent = os.path.dirname(pattern)
    
    


    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        # images = images.permute(0,3,1,2)  POSSIBILE BUG
        with torch.no_grad():
            result = model(images)

        # The anomaly score is calculated as 1 minus the maximum value of the model's output for each pixel
        # Anomaly score here is computed via MSP (Maximum Softmax Probability) method
        probs= torch.nn.Softmax(dim=1)(result)
        anomaly_result = 1.0 - torch.max(probs, dim=1)[0] 
        print('prediction shape', anomaly_result.shape)
        #print('GT shape', ood_gts.shape)
        print('MSP min:', anomaly_result.min().item())
        print('MSP max:', anomaly_result.max().item())
        plt.imshow(anomaly_result[0].cpu().numpy())
        plt.show()
        
        
        # MaxLogits version
        anomaly_result_logit = -torch.max(result, dim=1)[0]

        # Entropy version
        anomaly_result_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # The path to the ground truth mask is constructed by replacing the "images" directory in the input path with "labels_masks" 
        # and changing the file extension to match the format of the ground truth masks for each dataset.
        # Ground truth maske are used to evaluate the performance of the anomaly detection by comparing the predicted anomaly scores with the actual anomalies present in the images.
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  
        print(os.path.exists(pathGT))
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)
        print('unique GT values',np.unique(ood_gts))

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

         # CONTROLLO 
        print("GT path:", pathGT)
        print("Unique mask values:", np.unique(ood_gts))
        print("Images processed:", len(anomaly_score_list))
        if 1 not in np.unique(ood_gts):
            continue              
        else:
              ood_gts_list.append(ood_gts)
              anomaly_score_list.append(anomaly_result.cpu().numpy())
              anomaly_score_list_logit.append(anomaly_result_logit.cpu().numpy())
              anomaly_score_list_entropy.append(anomaly_result_entropy.cpu().numpy())
        del result, anomaly_result, anomaly_result_logit, anomaly_result_entropy, ood_gts, mask
        torch.cuda.empty_cache()

    # After processing all the images, the collected ground truth masks and anomaly scores are converted into NumPy arrays for further analysis.
    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list).squeeze(1)
    anomaly_scores_logit = np.array(anomaly_score_list_logit).squeeze(1)
    anomaly_scores_entropy = np.array(anomaly_score_list_entropy).squeeze(1)

    
    



    # The ground truth masks are used to create binary masks for in-distribution (ID) and out-of-distribution (OOD) samples.
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    # The anomaly scores for OOD and ID samples are extracted using the respective masks, and corresponding labels are created (1 for OOD and 0 for ID).
    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    ood_out_logit = anomaly_scores_logit[ood_mask]
    ind_out_logit = anomaly_scores_logit[ind_mask]
    ood_out_entropy = anomaly_scores_entropy[ood_mask]
    ind_out_entropy = anomaly_scores_entropy[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    ood_label_logit = np.ones(len(ood_out_logit))
    ind_label_logit = np.zeros(len(ind_out_logit))
    ood_label_entropy = np.ones(len(ood_out_entropy))
    ind_label_entropy = np.zeros(len(ind_out_entropy))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    val_out_logit = np.concatenate((ind_out_logit, ood_out_logit))
    val_label_logit = np.concatenate((ind_label_logit, ood_label_logit))
    val_out_entropy = np.concatenate((ind_out_entropy, ood_out_entropy))
    val_label_entropy = np.concatenate((ind_label_entropy, ood_label_entropy))

    # The performance of the anomaly detection is evaluated using two metrics: 
    # the area under the precision-recall curve (AUPRC) and the false positive rate at 95% true positive rate (FPR@TPR95).
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    prc_auc_logit = average_precision_score(val_label_logit, val_out_logit)
    fpr_logit = fpr_at_95_tpr(val_out_logit, val_label_logit)
    prc_auc_entropy = average_precision_score(val_label_entropy, val_out_entropy)
    fpr_entropy = fpr_at_95_tpr(val_out_entropy, val_label_entropy)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')
    print(f'AUPRC score (logit): {prc_auc_logit*100.0}')
    print(f'FPR@TPR95 (logit): {fpr_logit*100.0}')
    print(f'AUPRC score (entropy): {prc_auc_entropy*100.0}')
    print(f'FPR@TPR95 (entropy): {fpr_entropy*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) +  '    AUPRC (logit) score:' + str(prc_auc_logit*100.0) + '   FPR@TPR95 (logit):' + str(fpr_logit*100.0) + '    AUPRC (entropy) score:' + str(prc_auc_entropy*100.0) + '   FPR@TPR95 (entropy):' + str(fpr_entropy*100.0)))
    file.close()

if __name__ == '__main__':
    main()

# The code is designed to evaluate the performance of an anomaly detection model (ERFNet) on a dataset of images. It processes each image, computes anomaly scores, 
# and compares them against ground truth masks to calculate performance metrics such as AUPRC and FPR@TPR95. The results are printed and saved to a text file for further analysis.
