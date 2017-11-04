#Recursive-Feature-Elimination-using-Support-Vector-Machines

Support Vector Machines (SVM) usually generate a classification score rather than other techniques such as Principal Component Analysis (PCA) or Hidden markov Models (HMM).
This allows us to apply SVM's directly to alreaady existing scores from other techniques. Here, I have classification scores of 3 techniques, namely:
1. HMM (Hidden Markov Model)
2. OGS (Opcode Graph Similarity)
3. SSD (Simple Substitution Distance)
These techniques were used to determine whether the given sample is malware or benign. The file (mbscore.csv) consists of these scores for both malware and benign files (first 40 samples are malware, the next 40 samples are benign). Also since this code is written in python, I have added an additional classification score:
1. +1 denotes a malware sample
2. -1 denotes a benign sample
This is done so I can test the model accuracy based on this explicit score. 
If the number of features are large (in this case there are only 3), in order to improve efficieny we use fewer features without substantially hampering the effectiveness of the classification technique employed by SVM's. There are a number of ways to to this, here I have used a technique called Recursive Feature Elimination (RFE).
The objective is to reduce the number of features in consideration without impacting the accuracy of the model.

The concept is simple: 
1. We train an SVM based on all the features. 
2. After training, we obtain the weights associated with each feature.
3. Discard the feature with the lowest weight associated with it.
4. Re-train the model with the remaining features.

So in this way we are removing those features which if not taken into consideration do not alter the accuracy of the model significantly.
Now we can use other feature reduction techniques such as Ranked Feature Elimination, which discards low weight features after training an SVM only once, since the recursive technique is computationally very expensive. 
But, there is always some dependency between features, which show up in the recursive technique as compared to the ranked technique and the scores change every time you re-train the model with the reduced features. We definitely do not want to miss out on that.

Thus, we get a higher quality model, at the risk of being computationally expensive. 

I have used the Python ML library scikit-learn in order to implement SVM.
On running this program, we rank the feature weights, eliminate the feature with the lowest weight and continue the process recursively.
Finally we calculate the accuracy of the model at every step and see that by reducing the features, the accuracy of the model is not affected significantly.  
