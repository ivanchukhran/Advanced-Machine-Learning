# %%
import json
from enum import Enum
import numpy as np
import torch
from torch import nn
from torch.distributions.distribution import Distribution
from utils import *
# %%
with open('/home/ivan/nure/Advanced Machine Learning/labs/01/data/dataset_uniform.json', 'r') as f:
    ds_dict = json.load(f)
# %%
distributions, labels, distributions_params, lengths = ds_dict['distributions'], ds_dict['labels'], ds_dict['distributions_params'], ds_dict['lengths']
data_stream = np.array(distributions)
# %%
plot_distributions((data_stream, labels), distributions_params, distributions, lengths)

# %%
from river.datasets import synth  # we are going to use some synthetic datasets too
from river import naive_bayes, tree, linear_model, metrics, preprocessing, stream
from river.tree import HoeffdingTreeClassifier
from river.naive_bayes import GaussianNB
from river.linear_model import Perceptron
from models import IncrementalELM

def evaluate_incremental_learners(stream_data):
    """
    Evaluate different incremental learning algorithms on the data stream
    """
    # Initialize learners
    learners = {
        'Naive Bayes': GaussianNB(),
        'Hoeffding Tree': HoeffdingTreeClassifier(),
        'IELM': IncrementalELM(n_hidden=10),
        'SGD Perceptron': Perceptron()
    }

    # Initialize metrics for each learner
    accuracy_metrics = {name: metrics.Accuracy() for name in learners}

    # Store accuracy values for plotting
    accuracy_values = {name: [] for name in learners}

    n_iterations = len(stream_data)

    for i in range(n_iterations):
        x, y = stream_data[i]

        for name, learner in learners.items():
            y_pred = learner.predict_one({"x": x})
            accuracy_metrics[name].update(y, y_pred)
            accuracy_values[name].append(accuracy_metrics[name].get())
            learner.learn_one({"x": x}, y)

        if (i+1) % 1000 == 0:
            print(f'Processed {i+1}/{n_iterations} instances')
            for name, metric in accuracy_metrics.items():
                print(f'{name} accuracy: {metric.get():.4f}')
            print('-' * 40)

            with open('data/metrics_uniform.json', 'w') as f:
                json.dump(accuracy_values, f)

    plt.figure(figsize=(12, 6))
    for name, values in accuracy_values.items():
        plt.plot(values, label=name)

    plt.title('Incremental Learning Accuracy')
    plt.xlabel('Instances')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Return final accuracy values
    return {name: metric.get() for name, metric in accuracy_metrics.items()}

# %%
import time
stream = [(x, y) for x, y in zip(distributions, labels)]
start = time.time()
results = evaluate_incremental_learners(stream)
end = time.time()
print(f'Elapsed time: {end - start:.2f} seconds')
