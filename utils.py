from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from rff import RFF, RFF_sincos


def plot_comparison(dataset_name,
                     rbf_svm_score, rbf_svm_time,
                     linear_svm_score, linear_svm_time,
                     rff_scores, rff_times,
                     rff_sc_scores, rff_sc_times,
                     nystroem_scores, nystroem_times,
                     component_nums, task='classification'):

    fig, (fig_score, fig_time) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    fig_score.axhline(rbf_svm_score, color='r', label='RBF SVM')
    fig_time.axhline(rbf_svm_time, color='r', label='RBF SVM')

    fig_score.axhline(linear_svm_score, color='b', label='Linear SVM')
    fig_time.axhline(linear_svm_time, color='b', label='Linear SVM')

    fig_score.plot(component_nums, rff_scores, '--', label='RFF approx + Linear SVM')
    fig_time.plot(component_nums, rff_times, '--', label='RFF approx + Linear SVM')

    fig_score.plot(component_nums, rff_sc_scores, '--', label='RFF_sincos approx + Linear SVM')
    fig_time.plot(component_nums, rff_sc_times, '--', label='RFF_sincos approx + Linear SVM')

    fig_score.plot(component_nums, nystroem_scores, '--', label='Nystroem approx + Linear SVM')
    fig_time.plot(component_nums, nystroem_times, '--', label='Nystroem approx + Linear SVM')

    fig_score.legend(loc='best')
    fig_time.legend(loc='best')

    fig_score.set_xlabel('Transformed feature dimension', size=12)
    if task == 'classification':
        fig_score.set_ylabel('Mean accuracy', size=12)
    if task == 'regression':
        fig_score.set_ylabel('coefficient R^2', size=12)

    fig_time.set_xlabel('Transformed feature dimension', size=12)
    fig_time.set_ylabel('Training time (s)', size=12)

    plt.suptitle('Comparison on dataset ' + dataset_name, va='baseline', size=20, weight='bold')
    plt.tight_layout(3)
    plt.savefig('result/'+dataset_name+'.png')
    plt.show()


def data_preparation(dataset_name):
    # load dataset
    if dataset_name == 'digits':
        data_target = datasets.load_digits(n_class=10)
    elif dataset_name == 'california':
        data_target = datasets.fetch_california_housing()

    data = data_target.data
    target = data_target.target
    
    # split for training and test
    split = int(0.6 * len(data))
    data_train = data[:split]
    data_test = data[split:]
    targets_train = target[:split]
    targets_test = target[split:]

    # scaling
    if dataset_name == 'digits':
        scaler = MinMaxScaler()
    if dataset_name == 'california':
        scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)

    return data_train, data_test, targets_train, targets_test


def compare(dataset_name, gamma, task='classification'):
    data_train, data_test, targets_train, targets_test = data_preparation(dataset_name)
    
    if task == 'classification':
        rbf_svm = SVC(gamma=gamma, kernel='rbf')
        linear_svm = LinearSVC()
    if task == 'regression':
        rbf_svm = SVR(gamma=gamma, kernel='rbf')
        linear_svm = LinearSVR()

    # baseline1: SVM with RBF kernel
    starttime = time()
    rbf_svm.fit(data_train, targets_train)
    rbf_svm_time = time() - starttime
    rbf_svm_score = rbf_svm.score(data_test, targets_test)

    # baseline2: Linear SVM
    starttime = time()
    linear_svm.fit(data_train, targets_train)
    linear_svm_time = time() - starttime
    linear_svm_score = linear_svm.score(data_test, targets_test)

    # Approximation methods for random features and linear SVM
    rff = RFF(gamma=gamma)
    rff_sc = RFF_sincos(gamma=gamma)
    nystroem = Nystroem(gamma=gamma)

    if task == 'classification':
        rff_approx = Pipeline([('feature_map', rff), ('svm', LinearSVC())])
        rff_sc_approx = Pipeline([('feature_map', rff_sc), ('svm', LinearSVC())])
        nystroem_approx = Pipeline([('feature_map', nystroem), ('svm', LinearSVC())])
    if task == 'regression':
        rff_approx = Pipeline([('feature_map', rff), ('svm', LinearSVR())])
        rff_sc_approx = Pipeline([('feature_map', rff_sc), ('svm', LinearSVR())])
        nystroem_approx = Pipeline([('feature_map', nystroem), ('svm', LinearSVR())])
    
    rff_scores = []
    rff_times = []

    rff_sc_scores = []
    rff_sc_times = []

    nystroem_scores = []
    nystroem_times = []

    if data_test.shape[0] > 5000:
        component_nums = np.arange(50, 1000, 50)
    else:
        component_nums = int(data_test.shape[0]/30) * np.arange(1, 10)

    for n in component_nums:
        rff_approx.set_params(feature_map__n_components=n)
        starttime = time()
        rff_approx.fit(data_train, targets_train)
        rff_times.append(time() - starttime)
        rff_score = rff_approx.score(data_test, targets_test)
        rff_scores.append(rff_score)

        rff_sc_approx.set_params(feature_map__n_components=n)
        starttime = time()
        rff_sc_approx.fit(data_train, targets_train)
        rff_sc_times.append(time() - starttime)
        rff_sc_score = rff_sc_approx.score(data_test, targets_test)
        rff_sc_scores.append(rff_sc_score)

        nystroem_approx.set_params(feature_map__n_components=n)
        starttime = time()
        nystroem_approx.fit(data_train, targets_train)
        nystroem_times.append(time() - starttime)
        nystroem_score = nystroem_approx.score(data_test, targets_test)
        nystroem_scores.append(nystroem_score)

    plot_comparison(dataset_name,
                     rbf_svm_score, rbf_svm_time,
                     linear_svm_score, linear_svm_time,
                     rff_scores, rff_times,
                     rff_sc_scores, rff_sc_times,
                     nystroem_scores, nystroem_times,
                     component_nums, task)