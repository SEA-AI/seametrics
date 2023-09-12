import numpy as np
import matplotlib.pyplot as plt 
import json
# from Utils.image_utils import DetectionDrawer, DetectionObject, AnnotationObiect



class SaveMetricFile():
    """The class saves the detection (bboxes and labels) and ground truth data to a json file
       to be evaluated with e.g. MetricAP class  
    """
    def __init__(self):
        self.metic_dict = {}
        self.label_list = []

    def add_sample(self, img_path, predictions, ground_truths):
        """ Add a sample of predictions and ground truths for a single image to the metric dict

        Args:
            img_path (str): path to the image will be used as a key to find the sample
            predictions (list): predictions in the format [[label, confidence, x1 y1 x2 y2], ...]
            ground_truths (list): ground truths in the format [[label, x1 y1 x2 y2], ...]
        """
        write_predictions = [[item[0], float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])] for item in predictions]
        write_ground_truths = [[item[0], float(item[1]), float(item[2]), float(item[3]), float(item[4])] for item in ground_truths]
        self.metic_dict[img_path] = {"predictions": write_predictions, "ground_truths": write_ground_truths}

    def save_file(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.metic_dict, f)


class MetricAP():
    def __init__(self, label_list, IoU_threshold=0.5, interpolation_method="monotone_decreasing", ignore_double_detections=False):
        """Initializes the MetricAP class: This call calculates the AP for all classes in label_list and returns the mean AP

        Args:
            label_list (_type_): A list of all labels in the format [label1, label2, ...]
            IoU_threshold (float, optional): The IoU threshold to regard a detection a math. Defaults to 0.5.
            interpolation_method (str, optional): For the AP we need to integrate over the precision recall curve
                                                  and here we select the integration method. Defaults to "monotone_decreasing".
            ignore_double_detections (bool, optional): to select if double annotations are valid or not. Defaults to False.
        """
        self.match_list = []
        self.label_list = label_list
        self.IoU_threshold = IoU_threshold
        self.interpolation_method = interpolation_method
        self.all_detections = {label: [] for label in self.label_list}
        self.all_detections_sorted = {label: [] for label in self.label_list}
        self.annotation_count = {label: 0 for label in self.label_list}
        self.detection_count = {label: 0 for label in self.label_list}
        self.ignore_double_detections = ignore_double_detections

    def add_sample_to_metric(self, y_preds,  y_annos):
        """add sample to metric

        Args:
            y_preds (list): predictions in the format [[label, confidence, x1 y1 x2 y2], ...]
            y_annos (list): annotations in the format [[label, x1 y1 x2 y2], ...]
        """
        # calculate TP and FP for each label
        for label in self.label_list:

            # get the detections for the corresponding label
            label_detections = [y_pred for y_pred in y_preds if y_pred[0] == label]
            # sort detections by confidence
            label_detections = sorted(label_detections, key = lambda x: x[1], reverse = True)
            # get the annotation for the corresponding label
            label_annotations = [y_anno for y_anno in y_annos if y_anno[0] == label]
            # count the annotations per label
            self.annotation_count[label] += len(label_annotations)
            # count the detections per label
            self.detection_count[label] += len(label_detections)
            

            # ignore multiple detections of the same object
            if self.ignore_double_detections:
                for detection in label_detections:
                    TP = 0
                    for annotation in label_annotations:
                        iou = MetricAP.IoU(detection[2:], annotation[1:])
                        if iou > self.IoU_threshold:
                            TP = 1
                    self.all_detections[label].append([*detection, TP])

            # just count one object per annotation and label the others as false positives
            else:
                label_match_list = [0]*len(label_annotations)
                for detection in label_detections:
                    TP = 0
                    best_iou = 0
                    for idx, annotation in enumerate(label_annotations):
                        iou = MetricAP.IoU(detection[2:], annotation[1:])
                        # find best match
                        if iou >= best_iou:
                            best_iou = iou
                            best_match_idx = idx

                    if best_iou > self.IoU_threshold:   
                        if label_match_list[best_match_idx] == 0:
                            label_match_list[best_match_idx] = 1
                            TP = 1

                    self.all_detections[label].append([*detection, TP])   


    def calc_AP(self):
        """calculate the average precision for each label

        Returns:
            dict: Average Precision for each label
        """
        self.precision = {label: [] for label in self.label_list}
        self.recall = {label: [] for label in self.label_list}
        self.interpolation = {label: None for label in self.label_list}
        self.AP = {}
        for label in self.label_list:
            self.all_detections_sorted[label] = sorted(self.all_detections[label], key = lambda x: x[1], reverse = True)
            cumulative = 0
            for i, elem in enumerate(self.all_detections_sorted[label]):
                cumulative += elem[-1]
                self.all_detections_sorted[label][i].append(cumulative)
        
            detections = self.all_detections_sorted[label]
            annotation_count = self.annotation_count[label]
            precision = [elem[-1]/(idx+1) for idx, elem in enumerate(detections)]
            recall = [elem[-1]/(annotation_count+1e-8) for idx, elem in enumerate(detections)]
            self.precision[label] = precision
            self.recall[label] = recall

            if self.interpolation_method == "monotone_decreasing":
                self.AP[label], mpred, mrec = MetricAP.calculate_AP_monotonically_decreasing(precision, recall)
                self.interpolation[label] = {"precision": mpred, "recall": mrec}
            elif self.interpolation_method == "trapz_integration":
                self.AP[label] = np.trapz(precision, recall)
            else:
                print("interpolation_method not defined")
        return self.AP


    def calc_mAP(self):
        """calculate the mean average precision

        Returns:
            float: mean average precision
        """
        self.mAP = np.mean(list(self.AP.values()))
        return self.mAP
        

    @staticmethod
    def IoU(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        union = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - intersection
        iou = intersection / union
        return iou


    @staticmethod
    def calculate_AP_monotonically_decreasing(prec, rec):
        """    
            Calculate the AP given the recall and precision array
            1st) We compute a version of the measured precision/recall curve with
                precision monotonically decreasing
            2nd) We compute the AP as the area under this curve by numerical integration.

        Args:
            prec (list): precision list
            rec (list): recall list

        Returns:
            AP: Average Precision
            mpre: precision list with monotonically decreasing values
            mrec: recall list with increasing values
        """

        mpre = prec.copy()
        mpre.insert(0, 0)
        mpre.append(0)

        mrec = rec.copy()
        mrec.insert(0, 0)
        mrec.append(1)

        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1

        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mpre, mrec
    


    def plot_precision_recall(self, label, figure_size=(20,20)):
        """plot the precision recall curve for a given label

        Args:
            label (str): label
        """
        plt.figure(figsize=figure_size)
        if self.interpolation[label] != None:
            plt.plot(self.interpolation[label]["recall"], self.interpolation[label]["precision"], label = label + " interpolated", marker = "o")
        plt.plot(self.recall[label], self.precision[label], "x", label = label)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend()
        plt.show()



class MetricAP_old():
    """
    This class is used to calculate the average precision (AP) of an object detection model.
    It takes the predicted boxes and the annotation boxes as parameters and calculates their 
    intersection over union (IoU). If the IoU is above a certain threshold, it will be considered 
    a match. The predicted boxes and the annotation boxes are sorted based on the confidence 
    score and the false positive (FP) and true positive (TP) results are accumulated. Then, 
    the eleven-point interpolation is used to calculate the AP. The AP is the mean of all the 
    precision values from the eleven-point interpolation.
    """

    # Add parameters to control the IoU threshold and the number of interpolation points
    def __init__(self, IoU_threshold=0.01, interpolation_points=11, datagen=None):
        """Initializes the MetricAP class"""
        self.match_list = []
        self.annotation_sum = 0
        self.IoU_threshold = IoU_threshold
        self.interpolation_points = interpolation_points
        self.datagen = datagen



    def add_sample_to_metric(self, y_preds,  y_annos):
        """
        Adds a single sample to the metric

        Parameters
        ----------
        y_preds: list of predictions
        y_annos: list of annotations
        """
        self.annotation_sum += len(y_annos)

        for y_pred in y_preds:
            IoU_max_l = [0, y_pred, None]
            for y_anno in y_annos:
                IoU_val = MetricAP.IoU(y_pred.bbox.bbox,y_anno.bbox.bbox)
                IoU_max_l[2] = y_anno
                if IoU_max_l[0] < IoU_val:
                    IoU_max_l[0] = IoU_val
            if IoU_max_l[0] > self.IoU_threshold:
                self.match_list.append({"IoU":IoU_max_l[0], "Confidence": IoU_max_l[1].max_label_score, "TP": 1, "FP":0})
            else:
                self.match_list.append({"IoU":IoU_max_l[0], "Confidence": IoU_max_l[1].max_label_score, "TP": 0, "FP":1})


    def sort_match_list(self):
        """
        Sorts the match list by confidence, in descending order
        """
        self.match_list_sorted = sorted(self.match_list, key = lambda x: x["Confidence"], reverse = True)


    def accumulate_FP_TP(self):
        """
        Accumulates the true positives and false positives of the sorted match list and calculate precision and recall
        """
        ACC_TP = 0
        ACC_FP = 0
        for elem in self.match_list_sorted:
            ACC_TP += elem["TP"]
            ACC_FP += elem["FP"]
            elem["ACC_TP"] = ACC_TP 
            elem["ACC_FP"] = ACC_FP 
            Precision = ACC_TP/(ACC_TP+ACC_FP)
            Recall = ACC_TP/self.annotation_sum
            elem["Precision"] = Precision
            elem["Recall"] = Recall


    def curve_interpolation(self):
        """
        Calculates the x-Point interpolation of the precision-recall curve
        """

        def find_max_in_list(list, interpolation_recall):
            max_precision = 0
            for idx, elem in enumerate(list):
                if interpolation_recall <= elem["Recall"]:
                    if elem["Precision"] > max_precision:
                        max_precision = elem["Precision"]
            return max_precision
        
        self.AP_interpolation_list = []
        for i in range(self.interpolation_points):
            interpolation_recall = i/(self.interpolation_points-1)
            max_precision = find_max_in_list(self.match_list_sorted, interpolation_recall)
            self.AP_interpolation_list.append([interpolation_recall, max_precision])
        self.AP_interpolation_list = np.array(self.AP_interpolation_list).T

    def calc_AP(self):
        """
        Calculates and prints the AP of the given predictions and annotations
        """
        self.sort_match_list()
        self.accumulate_FP_TP()       
        self.curve_interpolation()
        self.AP = np.mean(self.AP_interpolation_list[1])
        return self.AP

    @staticmethod
    def IoU(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        union = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - intersection
        iou = intersection / union
        return iou

    def print(self):
        """
        Prints out the metric results
        """
        print("annotation_sum",self.annotation_sum)
        print("match_list",self.match_list)
        print("match_list_sorted",self.match_list_sorted)

    def draw_metric(self, figsize=(20,20)):
        """
        Draws the precision-recall curve of the given prediction and annotation
        """
        Precision = [elem["Precision"] for elem in self.match_list_sorted]
        Recall = [elem["Recall"] for elem in self.match_list_sorted]
        plt.figure(figsize=figsize)
        plt.plot(Recall, Precision)
        plt.plot(self.AP_interpolation_list[0], self.AP_interpolation_list[1])
        plt.xlabel("Recall")
        plt.ylabel("Precision")



## Helper functions
