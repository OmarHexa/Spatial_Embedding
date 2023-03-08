from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import numpy as np






class Metrics():
    def __init__(self,num_class) -> None:
        self.num_class = num_class+1 # The background class is added by the last 1
        self.CM = np.zeros((self.num_class, self.num_class))
        self.AP = np.zeros((num_class)) # without background label
        self.F1 =  None
        self.mAP = None
        self.TP = None
        self.FP = None
        self.FN = None
        self.P = None
        self.R = None
        
        self.count = 0
        
    def _add2CM(self,gt, pred):
        cf = confusion_matrix(gt,pred,labels=self.num_class)
        self.cm+=cf
        self.update()
        
    def _add2AP(self,gt, pred_score):
        for i in range(self.num_class - 1): # calculate AP for each class except the background class
            gt_mask = gt == i+1 # i+1 because the background class has label 0
            cl_score = pred_score[i].reshape(-1)
            ap = average_precision_score(gt_mask, cl_score)
            self.AP[i] += ap
        self.AP = self.AP/self.count
    
    
    def _update(self):
        self.TP = np.diag(self.CM)
        self.FN = self.CM.sum(axis=0) - self.TP
        self.FP = self.CM.sum(axis=1) - self.TP
        self.P = self.TP/(self.TP+self.FP)
        self.R = self.TP/(self.TP+self.FN)
        self.F1 = 2* (self.P*self.R)/(self.P+self.R)
        
        
        
    def add(self,gt_label,pred_label,pred_score):
        if not isinstance(gt_label,np.ndarray):
            print("The given array to Metric class has to be a numpy array \
                     but a different was given. Convert the array to numpy.")
        gt = gt.reshape(-1)
        pred_label = pred_label.reshape(-1)
        
        self._add2CM(gt, pred_label)
        self.count+=1
        self._add2AP(gt, pred_score)
        
        # Calculate mAP
        self.mAP = np.mean(self.AP)
        
        self.count+=1
        
    def log(self):
        with open(self.filename, 'w') as f:
            f.write(f"Mean Average Precision (mAP): {self.mAP:.4f}\n")
            f.write("\n")
            f.write("Class-wise Average Precision (AP):\n")
            for i in range(1,self.num_class):
                f.write(f"Class {i}: {self.AP[i-1]:.4f}\n")
            f.write("\n")
            f.write("Class-wise F1 score (F1):\n")
            for i in range(1,self.num_class):
                f.write(f"Class {i}: {self.F1[i]:.4f}\n")
            f.write("Confusion Matrix (CM):\n")
            for i in range(self.num_class):
                f.write("\t".join([str(int(x)) for x in self.metrics.CM[i]]) + "\n")
            f.write("\n")
            f.write(f"True Positives (TP): {self.metrics.TP}\n")
            f.write(f"False Positives (FP): {self.metrics.FP}\n")
            f.write(f"False Negatives (FN): {self.metrics.FN}\n")
            f.write(f"Precision (P): {self.metrics.P}\n")
            f.write(f"Recall (R): {self.metrics.R}\n")