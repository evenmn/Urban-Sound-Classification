# Urban-Sound-Classification
The [Urban Sound Challenge](https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/) is a classification contest provided by [Vidhya Analytics](https://datahack.analyticsvidhya.com/?utm_source=main-logo) with the purpose of introducing curious people to a real-world classification problem. After registered, one is provided with a dataset containing sounds from ten classes. For the training data set, the classes (targets) are given, but there is also a test dataset where the targets are unknown. Our task is to classify the test dataset correctly, and by uploading our results to Analytics Vidhya's webpage, they will return the accuracy score. Participants are expected to submit their answers by December 31st 2018, and there will be a leaderboard. We are allowed to use all possible tools, including open-source libraries like TensorFlow/Keras and Scikit-Learn.

## Scientific abstract
The aim of this project is to divide various sounds into ten categories, inspired by the [Urban Sound Challenge](https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/). For that we have investigated the performance of logistic regression and various neural networks, like Feed-forward Neural Networks (FNNs), Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). 

When it comes to error estimation, accuracy score is a natural choice. Various activation functions were investigated, and regularization was added for the logistic case. Since we mostly used ADAM as optimization tool, we did not bother about the learning rate.
	
To extract features from our dataset (dimensionality reduction), we used a spectrogram in the CNN case, and Mel Frequency Ceptral Coefficients (MFCCs) for FNN and RNN. We also tried two RNN networks, mainly the Long Short-Term Memory (LSTM) network and the Gated Recurrent Unit (GRU) network. 

All the neural networks were more or less able to recognize the training set, but it was a significantly difference in validation accuracy. FNN provided the highest validation accuracy of 94%, followed by LSTM (88%), GRU (86%) and CNN (82%). Our linear model, logistic regression, was only able to classify 60% of the test set correctly. 

## Our approach
For detailed about what we have done any why we do so, please read [report.pdf](report.pdf). All the scripts are found in [this](https://github.com/evenmn/Urban-Sound-Classification/tree/master/doc/Python) folder.

## Contest results
You should note carefully that the training set and validation set described in the report have known targets. In addition, Vidhya Analytics have provided us with a test set without known targets as described above. The real challenge is to classify this data set, which we did in the appendix in [report.pdf](report.pdf). 

> Using 4 hidden layers of 1024 nodes each, ADAM optimizer with a batch size of 32 and 1000 epochs, ReLU activation function on hidden layers and softmax on the output, a dropout of 50\% in all layers and the MFCCs as inputs with 40 frames, we obtain a training accuracy of 98%. According to Vidhya Analytics, the test accuracy is just 60%, which is poor compared to the validation accuracy. Even though the test data set is way larger than the validation set (5434 against 1746), there is no reason why the accuracy should be lower if the average prediction difficulties are the same, which they should be. Maybe TensorFlow calculates the validation accuracy in another way than Vidhya Analytics? Or maybe there are
issues with the CSV-file that we uploaded?
