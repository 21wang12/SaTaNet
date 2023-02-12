from collections import defaultdict
import os
import argparse
import torch

import model
import numpy as np

from model import two_d_softmax
from model import nll_across_batch
from evaluate import evaluate, visualise_list_point
from evaluate import visualise, save_visualize
from evaluate import produce_sdr_statistics
from plots import radial_error_vs_ere_graph
from plots import roc_outlier_graph
from plots import reliability_diagram
from landmark_dataset import LandmarkDataset
from utils import prepare_config_output_and_logger
from torchsummary.torchsummary import summary_string

from utils import ExtractNLargestBlobs,save_pred_result



LMS_ELEM = ['S','N','Or','Por','Subspin','Subpra','Pogo','Me','Gn','Go','Lii','Uii','Ul','Ll','Subna','Stp','Pns','Ans','Ar']
OUTLIER = defaultdict(lambda: [])

def parse_args():
    parser = argparse.ArgumentParser(description='Test a network trained to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--testing_images',
                        help='The path to the testing images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--pretrained_model',
                        help='the path to a pretrained model',
                        type=str,
                        required=True)

    parser.add_argument('--outlier_threshold',
                        help='Classify landmarks with an ERE score over this value as outliers',
                        type=float,
                        default=1.5)
    
    parser.add_argument('--visual',
                        help='visulize the outlier point',
                        type=bool,
                        default=False)

    parser.add_argument('--pred_result',
                        help='visulize the outlier point',
                        type=bool,
                        default=False)

    parser.add_argument('--save_visual',
                        help='visulize the outlier point',
                        type=bool,
                        default=False)

    args = parser.parse_args()

    return args


def main():

    # Get arguments and the experiment file
    args = parse_args()

    cfg, logger, output_path, _, _ = prepare_config_output_and_logger(args.cfg, 'test')

    # Outputfile
    if args.pred_result:
        output_file_name = "/path/to/ISBI2015/metric/pred.csv"
        output_file = open(output_file_name, "w+")
        FIELDNAMES = ['S_x','S_y','N_x','N_y','Or_x','Or_y','Por_x','Por_y','Subspin_x','Subspin_y','Subpra_x','Subpra_y','Pogo_x','Pogo_y','Me_x','Me_y','Gn_x','Gn_y','Go_x','Go_y','Lii_x','Lii_y','Uii_x','Uii_y','Ul_x','Ul_y','Ll_x','Ll_y','Subna_x','Subna_y','Stp_x','Stp_y','Pns_x','Pns_y','Ans_x','Ans_y','Ar_x','Ar_y']
        line = "imagename"
        for FN in FIELDNAMES:
            line += "," + FN
        line += "\n"
        output_file.write(line)
    # Print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # Print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    # Load the testing dataset and put it into a loader
    test_dataset = LandmarkDataset(args.testing_images, args.annotations, cfg.DATASET, perform_augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model and state dict from file
    model = eval("model." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS)
    loaded_state_dict = torch.load(args.pretrained_model, map_location=torch.device('cpu'))
    model.load_state_dict(loaded_state_dict, strict=True)

    logger.info("-----------Model Summary-----------")
    model_summary, _ = summary_string(model, (1, *cfg.DATASET.CACHED_IMAGE_SIZE), device=torch.device('cpu'))
    logger.info(model_summary)

    logger.info("-----------Start Testing-----------")
    model.eval()
    all_losses = []
    all_radial_errors = []
    all_expected_radial_errors = []
    all_mode_probabilities = []

    MRE_RANK = []

    with torch.no_grad():
        for idx, (image, channels, meta) in enumerate(test_loader):

            output, texture_output, strcture_output = model(image.float())
            # cls_output = two_d_softmax(cls_output)
            # cls_output /= cls_output.max()
            # mix_output = two_d_softmax(mix_output)
            output = model.scale(output)
            output = two_d_softmax(output)
            
            # output = cls_output
            # 后处理模块，清除多余连通图
            # output = output.cpu().detach().numpy()
            # cls_output = cls_output.cpu().detach().numpy()
            # for i in range(cls_output.shape[0]): # for each batch
            #     normalized_cls_output = cls_output[i] / np.max(cls_output[i], axis=(1, 2), keepdims=True)
            #     for j in range(normalized_cls_output.shape[0]): # for each channel
            #         normalized_cls_output[j][normalized_cls_output[j] > 0.3] = 1
            #         normalized_cls_output[j][normalized_cls_output[j] <= 0.9] = 0
            #         binaryImage = normalized_cls_output[j]
            #         largestBlob = ExtractNLargestBlobs(binaryImage, 1)
            #         output[i][j] *= largestBlob
            # output = torch.from_numpy(output).float()

            # Get the radial/localisation error and expected radial error values for each heatmap
            radial_errors, expected_radial_errors, mode_probabilities\
                = evaluate(output.detach().numpy(),
                           meta['landmarks_per_annotator'].detach().numpy(),
                           meta['pixel_size'].detach().numpy())
            all_radial_errors.append(radial_errors)
            all_expected_radial_errors.append(expected_radial_errors)
            all_mode_probabilities.append(mode_probabilities)

            # Print loss, radial error for each landmark and MRE for the image
            # Assumes that the batch size is 1 here
            msg = "Image: {}\t".format(meta['file_name'][0])
            vis_list = []
            for idx, radial_error in enumerate(radial_errors[0]):
                msg += "\t{:.3f}mm".format(radial_error)
                if(radial_error > 10):
                    OUTLIER[meta['file_name'][0]].append([LMS_ELEM[idx],radial_error])
                    vis_list.append(idx)
            msg += "\taverage: {:.3f}mm".format(np.mean(radial_errors))
            logger.info(msg)
            MRE_RANK.append([np.mean(radial_errors), meta['file_name'][0],image.cpu().detach().numpy(), output.cpu().detach().numpy(), texture_output.cpu().detach().numpy(), strcture_output.cpu().detach().numpy(),meta['landmarks_per_annotator'].cpu().detach().numpy()])
            if args.visual:
                visualise(meta['file_name'][0],image.cpu().detach().numpy(), output.cpu().detach().numpy(), texture_output.cpu().detach().numpy(), strcture_output.cpu().detach().numpy(),meta['landmarks_per_annotator'].cpu().detach().numpy())
            if len(vis_list) >= 1 and args.visual:
                visualise_list_point(image.cpu().detach().numpy(), output.cpu().detach().numpy(), meta['landmarks_per_annotator'].cpu().detach().numpy(), vis_list )
            if args.pred_result:
                save_pred_result(output_file,meta['file_name'][0], output.cpu().detach().numpy())

    # save visulize result.
    if args.save_visual:
        MRE_RANK = sorted(MRE_RANK, key = lambda x:x[0])
        for rank in range(len(MRE_RANK)):
            save_visualize(name=MRE_RANK[rank][1], rank=rank, heatmap_stack=MRE_RANK[rank][3], landmarks_per_annotator=MRE_RANK[rank][6])
    
    # Print out the statistics and graphs shown in the pape    logger.info("\n-----------Final Statistics-----------")
    # MRE per landmark
    all_radial_errors = np.array(all_radial_errors)
    mre_per_landmark = np.mean(all_radial_errors, axis=(0, 1))
    msg = "Average radial error per landmark: "
    for mre in mre_per_landmark:
        msg += "\t{:.3f}mm".format(mre)
    logger.info(msg)

    # Total MRE
    mre = np.mean(all_radial_errors)
    logger.info("Average radial error (MRE): {:.3f}mm".format(mre))

    # Detection rates
    flattened_radial_errors = all_radial_errors.flatten()
    sdr_statistics = produce_sdr_statistics(flattened_radial_errors, [2.0, 2.5, 3.0, 4.0])
    logger.info("Successful Detection Rate (SDR) for 2mm, 2.5mm, 3mm and 4mm respectively: "
                "{:.3f}% {:.3f}% {:.3f}% {:.3f}%".format(*sdr_statistics))

    # Generate graphs
    logger.info("\n-----------Save Graphs-----------")
    flattened_expected_radial_errors = np.array(all_expected_radial_errors).flatten()
    all_mode_probabilities = np.array(all_mode_probabilities).flatten()

    # Save the correlation between radial error and ere graph
    graph_save_path = os.path.join(output_path, "re_vs_ere_correlation_graph")
    logger.info("Saving radial error vs expected radial error (ERE) graph to => {}".format(graph_save_path))
    radial_error_vs_ere_graph(flattened_radial_errors, flattened_expected_radial_errors, graph_save_path)

    # Save the roc outlier graph
    graph_save_path = os.path.join(output_path, "roc_outlier_graph")
    logger.info("Saving roc outlier graph to => {}".format(graph_save_path))
    proposed_threshold = roc_outlier_graph(flattened_radial_errors, flattened_expected_radial_errors, graph_save_path)

    # Save the roc outlier graph
    graph_save_path = os.path.join(output_path, "reliability_diagram")
    logger.info("Saving reliability diagram to => {}".format(graph_save_path))
    reliability_diagram(flattened_radial_errors, all_mode_probabilities, graph_save_path)

    logger.info("\n-----------Outlier Prediction Experiment-----------")

    # Outlier threshold proposal
    logger.info("Classifying heatmaps with an ERE > {:.3f} produces "
                "a true positive rate of 0.5 for detecting outliers".format(proposed_threshold))

    logger.info("Using {:.3f} as a threshold for ERE we split the overall "
                "set into a 'good' and 'erroneous' set with the following statistics:".format(args.outlier_threshold))
    good_set_radial_errors = flattened_radial_errors[flattened_radial_errors <= args.outlier_threshold]
    good_set_sdr_statistics = produce_sdr_statistics(good_set_radial_errors, [2.0, 2.5, 3.0, 4.0])
    logger.info("A good set with {} landmarks for which the MRE is {:.3f}mm and the (SDR) "
                "for 2mm, 2.5mm, 3mm and 4mm respectively is {:.3f}% {:.3f}% {:.3f}% {:.3f}%"
                .format(len(good_set_radial_errors), np.mean(good_set_radial_errors), *good_set_sdr_statistics))

    erroneous_set_radial_errors = flattened_radial_errors[flattened_radial_errors > args.outlier_threshold]
    erroneous_set_sdr_statistics = produce_sdr_statistics(erroneous_set_radial_errors, [2.0, 2.5, 3.0, 4.0])
    logger.info("An erroneous set with {} landmarks for which the MRE is {:.3f}mm and the (SDR) "
                "for 2mm, 2.5mm, 3mm and 4mm respectively is {:.3f}% {:.3f}% {:.3f}% {:.3f}%"
                .format(len(erroneous_set_radial_errors), np.mean(erroneous_set_radial_errors),
                        *erroneous_set_sdr_statistics))

if __name__ == '__main__':
    main()