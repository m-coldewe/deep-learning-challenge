# Alphabet Soup Charity

## Overview (explain the purpose of this analysis)
The purpose of this analysis is to use machine learning and neural networks knowledge to build a tool that can help the non-profit foundation Alphabet Soup select applicants for funding with the best chance of success in their ventures. The data provided includes:
- EIN or Electronic Identification Number
- name
- application type
- affiliation
- classification
- use case
- organization
- status
- income amount
- special considerations
- ask amount, and
- whether the venture is successful (Boolean)

## Results (bulleted lists and images to support your answers)

### Data Preprocessing
During initial data processing, I chose to:
- drop EIN and Names as column that did not contribute to the data
- consolidate low-volume instances in the Application and Classification columns to lower the impact of outlier variables
- transform all remaining non-numeric columns using get_dummies
- set is_successful column as target(y) value
- assign the remaining columns as features(X)
- use StandardScaler to scale the data so inherently larger values wouldn't skew the results without cause

### Compiling, Training and Evaluating the model
With the scaler fit, and the data transformed, I compiled trained and evaluated the model by:
- creating 2 hidden layers
- (approximately) halving the input neurons from the pervious layer for each hidden layer
- using 'relu' for first all activation functions except the last, where 'sigmoid' is used
The idea was to keep the process simple and move steadily toward the final result.

![model_initial](https://github.com/m-coldewe/deep-learning-challenge/assets/152045367/a70d4d0c-94d4-495a-8c7c-522108c695b7)


Unfortunately, the initial model did not achieve the target model performance.

![model_initial_accuracy](https://github.com/m-coldewe/deep-learning-challenge/assets/152045367/e34705c2-f951-4757-979a-071f3de89f87)


For my first attempt to optimize the mode to achieve 75% accuracy, I added two additional hidden layers and used 'relu' for the activation function, thinking the additional complexity might discern more information within the possible relationships between values.

![model_1](https://github.com/m-coldewe/deep-learning-challenge/assets/152045367/e5f47fd0-86ae-45b5-8492-3dc0f72b631f)

![model_1_accuracy](https://github.com/m-coldewe/deep-learning-challenge/assets/152045367/f600f6a6-0157-44ad-8c6a-b46dd2fd227f)

As the accuracy for this first attempt at optimization returned a slightly lower accuracy than the initial attempt, I used the initial models' structure for the second attempt where, in addition to EIN and Name, I also dropped Affiliation and Special Considerations. However, the accuracy of the model dropped significantly.

![model_2_accuracy](https://github.com/m-coldewe/deep-learning-challenge/assets/152045367/a9475e81-ad0c-4d7b-9311-07bce7b6dbac)

For my third attempt, I returned the extra dropped columns (Affiliation and Special Circumstances) and instead tried changing the activation functions for the two hidden layers to 'tanh'.

![model_3](https://github.com/m-coldewe/deep-learning-challenge/assets/152045367/1c91d56d-97a7-4ab4-889e-f23aef640618)

While this model performed better than the previous model, it still lagged behind the initial and first optimization attempt models.

![model_3_accuracy](https://github.com/m-coldewe/deep-learning-challenge/assets/152045367/8d01f3f2-3080-42f2-8ff8-93e49e362ff3)

My classmate, Harsh, suggested I try adding a column, so for the fourth optimization model I added Name back into the dataset as a feature and consolidated the low value count instances to try to mitigate the impact of many outlier values, and then applied get_dummies. As this added a lot of columns to the dataset, I also added an additional hidden layer to this model, and adjusted the number of nodes in each layer to maintain a funnel.

![model_4](https://github.com/m-coldewe/deep-learning-challenge/assets/152045367/7c6e0376-fd6a-4799-b17a-c5a4e010b958)

While this fourth attempt also did not achieve 75% accuracy, it did perform better than the initial model, as well as the first optimization model, over 100 epochs. 

![model_4_accuracy](https://github.com/m-coldewe/deep-learning-challenge/assets/152045367/77e335a3-d31b-4112-b72b-53fa1103db32)




## Summary 
While none of the models performed as well as I had hoped, ending with ranked accuracy scores as follows:
Model 4: 0.741
Initial model: 0.7275
Model 1: 0.7271
Model 3: 0.7266
Model 2: 0.658 
I do believe the model is moving in the right direction. I'm not sure what sort of different model might achieve better results (or I would have done that) but for further optimization, I would recommend consolidating the Names at a lower value. The consolidation of 'Other' values for Names far exceeded any of the real value counts, which may have skewed the results. It also could be worth either binning the Ask Amount or dropping it from consideration. The most common ask amount within the dataset is 5000, effectively rendering every other Ask Amount an outlier, so consolidating the other values into ranger might clarify the relationship between Ask Amount and the rest of the features, or it might not be relevant. 
