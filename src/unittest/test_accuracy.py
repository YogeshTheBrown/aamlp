import sys
sys.path.append("../")
import metrics
import unittest


class Testmetrics(unittest.TestCase):

    def test_accuracy(self):
        l1=[0,1,1,1,0,0,0,1]
        l2=[0,1,0,1,0,1,0,0]
        self.assertEqual(metrics.accuracy(l1, l2), 0.625)

    def test_true_positive(self):
        l1=[0,1,1,1,0,0,0,1]
        l2=[0,1,0,1,0,1,0,0]
        self.assertEqual(metrics.true_positive(l1, l2), 2)

    def test_true_negative(self):
        l1=[0,1,1,1,0,0,0,1]
        l2=[0,1,0,1,0,1,0,0]
        self.assertEqual(metrics.true_negative(l1, l2), 3)

    def test_false_positive(self):
        l1=[0,1,1,1,0,0,0,1]
        l2=[0,1,0,1,0,1,0,0]
        self.assertEqual(metrics.false_positive(l1, l2), 1)

    def test_false_negative(self):
        l1=[0,1,1,1,0,0,0,1]
        l2=[0,1,0,1,0,1,0,0]
        self.assertEqual(metrics.false_negative(l1, l2), 2)

    def test_accuracy_v2(self):
        l1=[0,1,1,1,0,0,0,1]
        l2=[0,1,0,1,0,1,0,0]
        self.assertEqual(metrics.accuracy_v2(l1, l2), 0.625)        
    
    def test_precision(self):
        l1=[0,1,1,1,0,0,0,1]
        l2=[0,1,0,1,0,1,0,0]
        self.assertEqual(round(metrics.precision(l1, l2), 3), 0.667)

    def test_recall(self):
        l1 = [0,1,1,1,0,0,0,1]
        l2 = [0,1,0,1,0,1,0,0]
        self.assertEqual(round(metrics.recall(l1, l2), 3), 0.5) 

    def test_f1(self):
        y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                  1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        self.assertEqual(round(metrics.f1(y_true, y_pred), 3), 0.571)     

    def test_log_loss(self):
        y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
        y_proba = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]
        self.assertEqual(round(metrics.log_loss(y_true, y_proba), 3),0.499)



    
if __name__ == "__main__":
    unittest.main()
    
