from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt
from torch import maximum
from utils import get_maxima
from joblib import load
from PIL import Image, ImageOps, ImageDraw

# Get the predicted landmark point from the coordinate of the hottest point
def get_hottest_point(heatmap):
    w, h = heatmap.shape
    flattened_heatmap = np.ndarray.flatten(heatmap)
    hottest_idx = np.argmax(flattened_heatmap)
    return np.flip(np.array(np.unravel_index(hottest_idx, [w, h])))


def get_mode_probability(heatmap):
    return np.max(heatmap)


def calculate_ere(heatmap, predicted_point_scaled, pixel_size, significant_pixel_cutoff=0.05):
    normalized_heatmap = heatmap / np.max(heatmap)
    normalized_heatmap = np.where(normalized_heatmap > significant_pixel_cutoff, normalized_heatmap, 0)
    normalized_heatmap /= np.sum(normalized_heatmap)
    indices = np.argwhere(normalized_heatmap)
    ere = 0
    for twod_idx in indices:
        scaled_idx = np.flip(twod_idx) * pixel_size
        dist = np.linalg.norm(predicted_point_scaled - scaled_idx)
        ere += dist * normalized_heatmap[twod_idx[0], twod_idx[1]]
    return ere


def evaluate_with_KNN(heatmap_stack, landmarks_per_annotator, pixels_sizes):
    # pixel_sizes = 0.30234375
    batch_size, no_of_key_points, w, h = heatmap_stack.shape
    radial_error_per_landmark = np.zeros((batch_size, no_of_key_points))
    expected_error_per_landmark = np.zeros((batch_size, no_of_key_points))
    mode_probability_per_landmark = np.zeros((batch_size, no_of_key_points))
    KNNModel = load('/path/to/test/KNN.joblib')

    psuedo_predicted_point = []
    uncertain_point_idxs = []
    for i in range(batch_size):
        for j in range(no_of_key_points):
            psuedo_predicted_point.append(get_hottest_point(heatmap_stack[i, j]))
    for i in range(no_of_key_points):
        for j in range(i+1, no_of_key_points):
            # calculate distance of psuedo_predicted_point[i] psuedo_predicted_point[j]
            psuedo_dist = np.linalg.norm(psuedo_predicted_point[i]-psuedo_predicted_point[j])
            # if distance < 4mm
            if psuedo_dist < 13:
                # add i and j to uncertain_point_idxs
                uncertain_point_idxs.append(i)
                uncertain_point_idxs.append(j)

    for i in range(batch_size):
        pixel_size_for_sample = pixels_sizes[i]

        heatmap_stack[i] = heatmap_stack[i] / np.max(heatmap_stack[i], axis=(1, 2), keepdims=True)
        for j in range(no_of_key_points):

            predicted_point = get_hottest_point(heatmap_stack[i, j])
            # Get predicted point, 使用KNN来选择对应节点
            if j in uncertain_point_idxs:
                proposal_points = []
                maxima_threshold = 0.01
                while len(proposal_points) == 0:
                    maxima_cordinates = get_maxima(heatmap_stack[i, j], maxima_threshold)
                    for cordinate in maxima_cordinates:
                        if KNNModel.predict([cordinate])[0,j] == 1:
                            proposal_points.append(cordinate)
                    maxima_threshold /= 10
                proposal_points = sorted(proposal_points, key=lambda point: heatmap_stack[i, j, int(point[1]),int(point[0])], reverse=True)
                predicted_point = proposal_points[0].astype(int)
            
            predicted_point_scaled = predicted_point * pixel_size_for_sample
            # Average the annotators to get target point
            target_point = np.mean(landmarks_per_annotator[i, :, j], axis=0)
            target_point_scaled = target_point * pixel_size_for_sample

            localisation_error = np.linalg.norm(predicted_point_scaled - target_point_scaled)
            radial_error_per_landmark[i, j] = localisation_error

            expected_error_per_landmark[i, j] = calculate_ere(heatmap_stack[i, j], predicted_point_scaled,
                                                              pixel_size_for_sample)

            mode_probability_per_landmark[i, j] = get_mode_probability(heatmap_stack[i, j])

    return radial_error_per_landmark, expected_error_per_landmark, mode_probability_per_landmark

def evaluate(heatmap_stack, landmarks_per_annotator, pixels_sizes):
    batch_size, no_of_key_points, w, h = heatmap_stack.shape
    radial_error_per_landmark = np.zeros((batch_size, no_of_key_points))
    expected_error_per_landmark = np.zeros((batch_size, no_of_key_points))
    mode_probability_per_landmark = np.zeros((batch_size, no_of_key_points))

    for i in range(batch_size):

        pixel_size_for_sample = pixels_sizes[i]

        for j in range(no_of_key_points):

            # Get predicted point
            predicted_point = get_hottest_point(heatmap_stack[i, j])
            predicted_point_scaled = predicted_point * pixel_size_for_sample

            # Average the annotators to get target point
            target_point = np.mean(landmarks_per_annotator[i, :, j], axis=0)
            target_point_scaled = target_point * pixel_size_for_sample

            localisation_error = np.linalg.norm(predicted_point_scaled - target_point_scaled)
            radial_error_per_landmark[i, j] = localisation_error

            expected_error_per_landmark[i, j] = calculate_ere(heatmap_stack[i, j], predicted_point_scaled,
                                                              pixel_size_for_sample)

            mode_probability_per_landmark[i, j] = get_mode_probability(heatmap_stack[i, j])

    return radial_error_per_landmark, expected_error_per_landmark, mode_probability_per_landmark

def save_visualize(name, rank, heatmap_stack, landmarks_per_annotator):
    COLOR = ['thistle', 'pink', 'mediumvioletred', 'gold', 'mediumorchid', 'turquoise', 'royalblue', 'navajowhite', 'plum', 'tomato','lightseagreen','thistle', 'pink', 'mediumvioletred', 'gold', 'mediumorchid', 'turquoise', 'royalblue', 'navajowhite']
    origin_image_folder = "/path/to/rawimg/"
    origin_image_path = origin_image_folder + name + ".bmp"
    # image = plt.imread(origin_image_path)
    # plt.imshow(image)
    # plt.axis('off')
    s = 0
    ground_truth_landmark_position = np.mean(landmarks_per_annotator[s], axis=0)

    COLOR = [(228, 87, 46),( 23, 190, 187),(255, 201, 20), (118, 176, 65),(152, 206, 0),(22, 224, 189),(120, 195, 251),(137, 166, 251),(152, 131, 143),(121, 35, 89),(215, 36, 131),(253, 62, 129),(45, 130, 183),(66, 226, 184),(243, 223, 191),(235, 138, 144),(160, 113, 120),(230, 204, 190), (200, 204, 146)]
    BASE_IMG = Image.open(origin_image_path).convert('RGB')
    DRAW = ImageDraw.Draw(BASE_IMG)
    for i,gt in enumerate(ground_truth_landmark_position):
        x = gt[0]
        y = gt[1]
        sz = 17
        DRAW.ellipse(((x*3-sz,y*3-sz,x*3+sz,y*3+sz)),fill=COLOR[i])
    BASE_IMG.save(f"/path/to/tmp/gt_r{rank}_n{name}.png")

    normalized_heatmaps = heatmap_stack[s] / np.max(heatmap_stack[s], axis=(1, 2), keepdims=True)
    predicted_landmark_positions = np.array([get_hottest_point(heatmap) for heatmap in normalized_heatmaps])
    BASE_IMG = Image.open(origin_image_path).convert('RGB')
    DRAW = ImageDraw.Draw(BASE_IMG)
    for i,pred in enumerate(predicted_landmark_positions):
        x = pred[0]
        y = pred[1]
        sz = 17
        DRAW.ellipse(((x*3-sz,y*3-sz,x*3+sz,y*3+sz)),fill=COLOR[i])
    BASE_IMG.save(f"/path/to/tmp/pred_r{rank}_n{name}.png")

    BASE_IMG = Image.open(origin_image_path).convert('RGB')
    DRAW = ImageDraw.Draw(BASE_IMG)
    for i in range(len(ground_truth_landmark_position)):
        sz = 17
        gt = ground_truth_landmark_position[i]
        pred = predicted_landmark_positions[i]
        gt_x = gt[0]
        gt_y = gt[1]
        pred_x = pred[0]
        pred_y = pred[1]
        DRAW.ellipse(((gt_x*3-sz,gt_y*3-sz,gt_x*3+sz,gt_y*3+sz)),fill="green")
        DRAW.ellipse(((pred_x*3-sz,pred_y*3-sz,pred_x*3+sz,pred_y*3+sz)),fill="red")
        DRAW.line(xy=[(gt_x*3,gt_y*3),(pred_x*3,pred_y*3)],fill=(115,250,253),width=5)
    BASE_IMG.save(f"/path/to/tmp/pg_r{rank}_n{name}.png")
    # Display image

    # Display heatmaps
    # normalized_heatmaps = heatmap_stack[s] / np.max(heatmap_stack[s], axis=(1, 2), keepdims=True)
    # squashed_heatmaps = np.max(normalized_heatmaps, axis=0)
    # plt.imshow(squashed_heatmaps, cmap='inferno', alpha=0.4)

    # Display ground truth points
    # ground_truth_landmark_position = np.mean(landmarks_per_annotator[s], axis=0)
    # plt.scatter(ground_truth_landmark_position[:, 0], ground_truth_landmark_position[:, 1],color='green', s=70, alpha = 0.15)
    # plt.scatter(ground_truth_landmark_position[:, 0], ground_truth_landmark_position[:, 1], edgecolors='none',color='green', s=5, alpha = 1)

    # Display predicted points
    # predicted_landmark_positions = np.array([get_hottest_point(heatmap) for heatmap in normalized_heatmaps])
    # plt.scatter(predicted_landmark_positions[:, 0], predicted_landmark_positions[:, 1],color='red', s=70, alpha = 0.15)
    # plt.scatter(predicted_landmark_positions[:, 0], predicted_landmark_positions[:, 1], edgecolors='none',color='red', s=5, alpha = 1)

    # plt.show()
    # plt.savefig(f"/path/to/tmp/{rank}_{name}.png", bbox_inches='tight',dpi=600,pad_inches = 0)
    # plt.close()

# This function will visualise the output of the first image in the batch
def visualise(name, images, heatmap_stack, texture_stack, structure_stack,landmarks_per_annotator):
    s = 0

    # Display image
    image = images[s, 0]
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Display heatmaps
    normalized_heatmaps = heatmap_stack[s] / np.max(heatmap_stack[s], axis=(1, 2), keepdims=True)
    squashed_heatmaps = np.max(normalized_heatmaps, axis=0)
    plt.imshow(squashed_heatmaps, cmap='inferno', alpha=0.4)

    # Display ground truth points
    ground_truth_landmark_position = np.mean(landmarks_per_annotator[s], axis=0)
    plt.scatter(ground_truth_landmark_position[:, 0], ground_truth_landmark_position[:, 1],color='green', s=70, alpha = 0.15)
    plt.scatter(ground_truth_landmark_position[:, 0], ground_truth_landmark_position[:, 1], edgecolors='none',color='green', s=5, alpha = 1)

    # Display predicted points
    predicted_landmark_positions = np.array([get_hottest_point(heatmap) for heatmap in normalized_heatmaps])
    plt.scatter(predicted_landmark_positions[:, 0], predicted_landmark_positions[:, 1],color='red', s=70, alpha = 0.15)
    plt.scatter(predicted_landmark_positions[:, 0], predicted_landmark_positions[:, 1], edgecolors='none',color='red', s=5, alpha = 1)

    # plt.show()
    # plt.savefig(f"/path/to/tmp/{name}.png", bbox_inches='tight',dpi=600,pad_inches = 0)
    # plt.close()

# This function will visualise output of the first image in the batch, where the visulized point is sotre in the vis_list
def visualise_list_point(images, heatmap_stack, landmarks_per_annotator, vis_list ):
    s = 0

    # Display image
    image = images[s, 0]
    plt.imshow(image, cmap='gray')

    # Display heatmaps
    normalized_heatmaps = heatmap_stack[s] / np.max(heatmap_stack[s], axis=(1, 2), keepdims=True)
    squashed_heatmaps = np.max(normalized_heatmaps, axis=0)
    
    # plt.imshow(squashed_heatmaps, cmap='inferno', alpha=0.4)
    plt.imshow(normalized_heatmaps[vis_list[0]], cmap='inferno', alpha=0.4)
    # plt.imshow(normalized_cls_channels[vis_list[0]], cmap='Blues')
    # Display predicted points
    predicted_landmark_positions = np.array([get_hottest_point(heatmap) for heatmap in normalized_heatmaps])
    plt.scatter(predicted_landmark_positions[vis_list, 0], predicted_landmark_positions[vis_list, 1], color='red', s=1)

    # Display ground truth points
    ground_truth_landmark_position = np.mean(landmarks_per_annotator[s], axis=0)
    plt.scatter(ground_truth_landmark_position[vis_list, 0], ground_truth_landmark_position[vis_list, 1], color='green', s=1)
    maxima_cord = get_maxima(normalized_heatmaps[vis_list[0]])
    plt.scatter(maxima_cord[:, 0], maxima_cord[:, 1], color='blue', s=1)
    plt.show()

def produce_sdr_statistics(radial_errors, thresholds):
    successful_detection_rates = []
    for threshold in thresholds:
        sdr = 100 * np.sum(radial_errors < threshold) / len(radial_errors)
        successful_detection_rates.append(sdr)
    return successful_detection_rates


