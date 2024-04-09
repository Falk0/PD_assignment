
import nibabel as nib 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, mean_squared_error
from sklearn.preprocessing import StandardScaler


# Paths to the NIfTI files
voi_template_path = 'data/voi_template.nii'
measurement_nii_path = 'data/training_images_rcbf.nii'  # Placeholder path
measurement_nii_path_sbr = 'data/training_images_sbr.nii' 


# Load the VOI template and the measurement data
voi_img = nib.load(voi_template_path)
measurement_img = nib.load(measurement_nii_path)
measurement_img_sbr = nib.load(measurement_nii_path_sbr)

voi_data = voi_img.get_fdata()
measurement_data = measurement_img.get_fdata()
measurement_data_sbr = measurement_img_sbr.get_fdata()

labels = pd.read_csv('data/labels.csv')

def get_patient_diagnosis(df, patientID):
    filtered_df = df[df['patient_number'] == patientID]
    return int(filtered_df['diagnose'].iloc[0])

# Define a function to calculate the mean value in a region specified by a label
def calculate_mean_measurement_for_label(measurement_data, voi_data, label):
    mask = voi_data == label
    masked_data = measurement_data[mask]
    if masked_data.size > 0:
        return np.mean(masked_data)
    else:
        return np.nan  # Return NaN if the label is not found in the voi_data


# Labels and structures
labels_structures = {
    #10: "Right cerebellum",
    #11: "Left cerebellum",
    16: "Right cingulate anterior",
    17: "Left cingulate anterior",
    #18: "Right thalamus",
    #19: "Left thalamus",
    28: "Right parietal cortex",
    29: "Left parietal cortex",
    34: "Right occipital cortex",
    35: "Left occipital cortex",
    #39: "Right cingulate posterior", ---
    #40: "Left cingulate posterior", 
    #80: "Right caudate nucleus",
    #90: "Left caudate nucleus",
    105: "Right putamen",
    120: "Left putamen",
}

labels_structures_sbr = {
    #10: "Right cerebellum",
    #11: "Left cerebellum",
    #16: "Right cingulate anterior",
    #17: "Left cingulate anterior",
    #18: "Right thalamus",
    #19: "Left thalamus",
    #28: "Right parietal cortex",
    #29: "Left parietal cortex",
    #34: "Right occipital cortex",
    #35: "Left occipital cortex",
    39: "Right cingulate posterior_sbr", 
    #40: "Left cingulate posterior_sbr", 
    #80: "Right caudate nucleus",
    #90: "Left caudate nucleus",
    105: "Right putamen_sbr",
    120: "Left putamen_sbr",
}

results = [] 
results_sbr = [] 
for i in range(0,40,1):
    measurement_data_index = measurement_data[:,:,:,i]
    measurement_data_index_sbr = measurement_data_sbr[:,:,:,i]    
    # Calculate and print mean measurements for each structure
    diagnosis = get_patient_diagnosis(labels, i+1)
    mean_measurements = {'Diagnosis': diagnosis}
    mean_measurements_sbr = {'Diagnosis': diagnosis}
    for label, structure in labels_structures.items():
        mean_value = calculate_mean_measurement_for_label(measurement_data_index, voi_data, label)
        mean_measurements[structure] = mean_value     
    for label, structure in labels_structures_sbr.items():
        mean_value_sbr = calculate_mean_measurement_for_label(measurement_data_index_sbr, voi_data, label)  
        mean_measurements_sbr[structure] = mean_value_sbr

    # Add the completed dictionary for the current patient to our results list
    results.append(mean_measurements)
    results_sbr.append(mean_measurements_sbr)


patients_stats_df = pd.DataFrame(results)
patients_stats_df_sbr = pd.DataFrame(results_sbr)


Y = patients_stats_df['Diagnosis']

X1 = patients_stats_df.drop(columns=['Diagnosis'])
X2 = patients_stats_df_sbr.drop(columns=['Diagnosis'])


X = pd.concat([X1, X2], axis=1)


print(X.head())

# Standardizing the features
X = StandardScaler().fit_transform(X)


train_model = True

if train_model:
    accuracies = []
    true_labels = []
    predictions = []
    k = 5
    for seed in range(0,1000,1):   
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed) 
        model = LogisticRegression(solver='liblinear')
        #model = KNeighborsClassifier()
        # Perform k-fold cross validation
        for train_index, test_index in skf.split(X, Y):    # Split data into training and test sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Fit the model
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)

            # Accumulate true labels and predictions
            true_labels.extend(y_test)
            predictions.extend(y_pred)

            accuracies.append(accuracy_score(y_test, y_pred))

    plt.figure(figsize=(3, 3))
    plt.hist(accuracies, bins=7)
    plt.xlim((0,1))
    plt.ylabel('Number of occurrences')
    plt.xlabel('Accuracy')
    plt.show()


    print(f'acc mean: {np.mean(accuracies)}')
    print(f'acc std: {np.std(accuracies)}')

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    print('Confusion matrix:')
    print(cm)
    
    print('Model coefficients:')
    print(model.coef_) # returns a matrix of weights (coefficients)


