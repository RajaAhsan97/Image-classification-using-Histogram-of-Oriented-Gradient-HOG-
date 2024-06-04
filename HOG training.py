import HOG_v3
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# set path of directory to read image
#base_dir = "HOG archive\\data"
base_dir = "HOG logos\\new_train_data"

data_class = os.listdir(base_dir)

data_dir = base_dir.rsplit("\\")[0]

data_rslt_save_dir = os.path.join(data_dir, "HOG_results")
if not os.path.isdir(data_rslt_save_dir):
    os.mkdir(data_rslt_save_dir)

HOG_FD = {}
DS_FUNC_TIME = {}

# load vehicle and non-vehicle images snd assign them a suppoting index value
for d_class in data_class:
    data_pth = os.path.join(base_dir, d_class)
    data = os.listdir(data_pth)
    for d in data:
        data[data.index(d)] = os.path.join(data_pth, d)

    data_rslt_dir = os.path.join(data_dir, "HOG_results", d_class)
    if not os.path.isdir(data_rslt_dir):
        os.mkdir(data_rslt_dir)

    start_time_fd = time.time()
    fd, func_ex_tm = HOG_v3.HOG(data[:], data_rslt_dir)
    end_time_fd = time.time()
    print("-------------------------------")
    print("Computed Feature Vectors: ")
    print("Time: ", end_time_fd - start_time_fd, " sec")
    print("-------------------------------")

    DS_FUNC_TIME[d_class] = func_ex_tm
    HOG_FD[d_class] = fd

dict_keys = list(HOG_FD.keys())
img_data = []
support_index = []
for key in dict_keys:
    images_fd = HOG_FD[key].columns
    for c in images_fd:
        img_data.append(np.array(HOG_FD[key][c]).reshape(1,-1).flatten())
        support_index.append(dict_keys.index(key))
        
X_train, X_test, Y_train, Y_test = train_test_split(img_data, support_index, test_size=0.3, random_state=42)

svc = svm.SVC(kernel="linear")
svc.fit(X_train, Y_train)

Y_predict = svc.predict(X_test)

accuracy = accuracy_score(Y_predict, Y_test)
print("Accuracy of model: ", accuracy*100)


fd_file = os.path.join(data_rslt_save_dir, "HOG_Fds.csv")
# write data to csv file
dataset_class = list(HOG_FD.keys())
with open(fd_file, "w") as csvfile:
    svm_acc = "Accuracy of SVM: " + str(accuracy*100) + "\n\n"
    csvfile.write(svm_acc)
    for ds_class in dataset_class:
        csvfile.write(ds_class)
        SV = "Support Index for SVM: " + str(dataset_class.index(ds_class))
        csvfile.write(SV)
        csvfile.write("\n\n")
        ds_class_img_fd = list(HOG_FD[ds_class].keys())
        for fd in ds_class_img_fd:
            get_fd = list(HOG_FD[ds_class][fd].values)
            csvfile.write(",".join(str(fd_val) for fd_val in get_fd))
            csvfile.write("\n")

