import argparse
import torch
import torch.nn as nn

import model
import numpy as np
import matplotlib.pyplot as plt

from model import nll_across_batch
from model import channel_softmax, nll_across_channel, two_d_softmax, mse_across_channel, two_d_normalize
from landmark_dataset import LandmarkDataset
from utils import prepare_config_output_and_logger
from torchsummary.torchsummary import summary_string

from sklearn import svm

'''
Code design based on Bin Xiao's Deep High Resolution Network Repository:
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--training_images',
                        help='The path to the training images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    args = parser.parse_args()

    return args


def main():
    # get arguments and the experiment file
    args = parse_args()

    cfg, logger, _, save_model_path, _ = prepare_config_output_and_logger(args.cfg, 'train')

    # print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    # load the train dataset and put it into a loader
    training_dataset = LandmarkDataset(args.training_images, args.annotations, cfg.DATASET, perform_augmentation=True)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    '''
    for batch, (image, channels, meta) in enumerate(train_loader):
        s = 0
        plt.imshow(image[s, 0].detach().numpy(), cmap='gray')
        squashed_channels = np.max(channels[s].detach().numpy(), axis=0)
        plt.imshow(squashed_channels, cmap='inferno', alpha=0.5)

        landmarks_per_annotator = meta['landmarks_per_annotator'].detach().numpy()[s]
        averaged_landmarks = np.mean(landmarks_per_annotator, axis=0)
        for i, position in enumerate(averaged_landmarks):
            plt.text(position[0], position[1], "{}".format(i + 1), color="yellow", fontsize="small")
        plt.show()
    '''

    model = eval("model." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS).cuda()
    criterion_mix = nn.L1Loss()
    criterion = nn.MSELoss()
    logger.info("-----------Model Summary-----------")
    model_summary, _ = summary_string(model, (1, *cfg.DATASET.CACHED_IMAGE_SIZE))
    logger.info(model_summary)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 6, 8], gamma=0.1)

    for epoch in range(cfg.TRAIN.EPOCHS):

        logger.info('-----------Epoch {} Training-----------'.format(epoch))

        model.train()
        losses_total_per_epoch = []
        losses_per_epoch = []
        losses_texture_per_epoch = []
        losses_strcture_per_epoch = []

        for batch, (image, channels, meta) in enumerate(training_loader):
            # Put image and channels onto gpu
            image = image.cuda()
            channels = channels.cuda()
            texture_channels = meta["texture_channels"].cuda()
            strcture_channels = meta["strcture_channels"].cuda()

            output, texture_output, strcture_output = model(image.float())
            output = two_d_softmax(output)
            strcture_output = two_d_normalize(strcture_output)
            texture_output = two_d_normalize(texture_output)

            optimizer.zero_grad()
            loss = nll_across_batch(output, channels)

            if cfg.TRAIN.MULTIHEAD:
                loss_texture = criterion(texture_output.float(), texture_channels.float())
                loss_strcture = criterion(strcture_output.float(), strcture_channels.float())
                # deno = loss + loss_texture + loss_strcture
                # w1 = (loss_texture + loss_strcture) / deno
                # w2 = (loss + loss_strcture) / deno
                # w3 = (loss + loss_texture) / deno
                # loss_total = (w1*loss + w2*loss_texture + w3*loss_strcture)*100
                loss_total = loss + loss_texture*100 + loss_strcture*100
            else:
                loss_total = loss

            loss_total.backward()

            optimizer.step()

            losses_total_per_epoch.append(loss_total.item())

            if (batch + 1) % 5 == 0:
                logger.info("[{}/{}]\tLoss: {:.3f}".format(batch + 1, len(training_loader), np.mean(losses_total_per_epoch)))

        scheduler.step()
        val(model,cfg,logger)

    logger.info("Saving Model's State Dict to {}".format(save_model_path))
    torch.save(model.state_dict(), save_model_path)
    logger.info("-----------Training Complete-----------")

from evaluate import evaluate, produce_sdr_statistics
from tqdm import tqdm
best_MRE = 10000
best_SDR = [0,0,0,0]
def val(model: nn.Module, cfg, logger):
    model.train(False)

    global best_MRE
    global best_SDR

    test_dataset = LandmarkDataset(cfg.DATASET.TEST_SET, "/path/to/ISBI2015/CHHeatmaps_process/data/AnnotationsByMD/", cfg.DATASET, perform_augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    all_radial_errors = []
    all_expected_radial_errors = []
    all_mode_probabilities = []
    logger.info("+++++++++++++testing+++++++++++++++++")
    pbar = tqdm(total = test_loader.__len__())
    
    for idx, (image, channels, meta) in enumerate(test_loader):
        image = image.cuda()
        output, texture_output, strcture_output = model(image.float())
        output = model.scale(output)
        output = two_d_softmax(output)
        radial_errors, expected_radial_errors, mode_probabilities  = evaluate(output.detach().cpu().numpy(), meta['landmarks_per_annotator'].detach().cpu().numpy(),meta['pixel_size'].detach().cpu().numpy())
        all_radial_errors.append(radial_errors)
        all_expected_radial_errors.append(expected_radial_errors)
        all_mode_probabilities.append(mode_probabilities)
        msg = "Image: {}\t".format(meta['file_name'][0])
        for idx, radial_error in enumerate(radial_errors[0]):
            msg += "\t{:.3f}mm".format(radial_error)
        msg += "\taverage: {:.3f}mm".format(np.mean(radial_errors))
        logger.info(msg)
        pbar.update(1)
    pbar.close()
    all_radial_errors = np.array(all_radial_errors)
    mre_per_landmark = np.mean(all_radial_errors, axis=(0, 1))
    mre = np.mean(all_radial_errors)
    flattened_radial_errors = all_radial_errors.flatten()
    sdr_statistics = produce_sdr_statistics(flattened_radial_errors, [2.0, 2.5, 3.0, 4.0])
    if  sum(best_SDR) / len(best_SDR) < sum(sdr_statistics) / len(sdr_statistics):
        
        best_SDR = sdr_statistics
        best_MRE = mre
    logger.info(f"Test val MRE: {mre}, SDR([2mm, 2.5mm, 3mm, 4mm]): {sdr_statistics}")
    logger.info(f"Best val MRE: {best_MRE}, SDR([2mm, 2.5mm, 3mm, 4mm]): {best_SDR}")
    model.train(True)

if __name__ == '__main__':
    main()