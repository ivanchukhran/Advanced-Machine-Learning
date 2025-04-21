# %% imports
import matplotlib.pyplot as plt
import numpy as np
from modAL.disagreement import vote_entropy_sampling
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import entropy_sampling
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %% setup
np.random.seed(42)

# Load the DIGITS dataset
digits = load_digits()
X, y = digits.data, digits.target

# Print dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters for active learning
n_initial = 200  # Initial labeled pool size
n_queries = 15  # Number of queries/iterations
n_instances_per_query = 10  # Number of instances to label in each iteration


# %% Function to evaluate model performance
def evaluate_model(learner, X_test, y_test):
    y_pred = learner.predict(X_test)
    return accuracy_score(y_test, y_pred)


# %% Function to plot digits
def plot_digit(data):
    image = data.reshape(8, 8)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


# %% Initialize the unlabeled pool
# We'll use the remaining training data as our unlabeled pool
X_initial, X_pool, y_initial, y_pool = train_test_split(X_train, y_train, train_size=n_initial, random_state=42)

# %% 1. Maximum Uncertainty Sampling with Entropy (ActiveLearner)
# Initialize base model for entropy-based uncertainty sampling
base_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize the ActiveLearner with entropy sampling strategy
entropy_learner = ActiveLearner(estimator=base_model, query_strategy=entropy_sampling, X_training=X_initial, y_training=y_initial)

# %% 2. Committee-based Learning with Maximum Uncertainty Entropy
# Create different base models for the committee
n_committee_members = 3
committee_members = []

for i in range(n_committee_members):
    # Different random states for diversity
    committee_member = ActiveLearner(
        estimator=RandomForestClassifier(n_estimators=100, random_state=i * 10), X_training=X_initial, y_training=y_initial
    )
    committee_members.append(committee_member)

# Initialize the Committee with vote_entropy sampling strategy
committee = Committee(learner_list=committee_members, query_strategy=vote_entropy_sampling)

# %% Evaluate initial models
initial_entropy_acc = evaluate_model(entropy_learner, X_test, y_test)
initial_committee_acc = evaluate_model(committee, X_test, y_test)

print(f"Initial accuracy (Entropy-based ActiveLearner): {initial_entropy_acc:.4f}")
print(f"Initial accuracy (Committee): {initial_committee_acc:.4f}")

# Lists to store accuracy at each iteration
entropy_accuracies = [initial_entropy_acc]
committee_accuracies = [initial_committee_acc]

# %% Active learning loop for both models
X_pool_entropy = X_pool.copy()
y_pool_entropy = y_pool.copy()
X_pool_committee = X_pool.copy()
y_pool_committee = y_pool.copy()

for i in range(n_queries):
    print(f"\nIteration {i + 1}/{n_queries}")

    # Entropy-based uncertainty sampling query
    query_idx_entropy, _ = entropy_learner.query(X_pool_entropy, n_instances=n_instances_per_query)

    # Get the corresponding instances and labels
    X_query_entropy = X_pool_entropy[query_idx_entropy]
    y_query_entropy = y_pool_entropy[query_idx_entropy]

    # Teach the entropy learner with new labeled instances
    entropy_learner.teach(X_query_entropy, y_query_entropy)

    # Remove queried instances from the entropy model's pool
    X_pool_entropy = np.delete(X_pool_entropy, query_idx_entropy, axis=0)
    y_pool_entropy = np.delete(y_pool_entropy, query_idx_entropy)

    # Evaluate entropy model
    entropy_acc = evaluate_model(entropy_learner, X_test, y_test)
    entropy_accuracies.append(entropy_acc)
    print(f"Entropy-based ActiveLearner accuracy: {entropy_acc:.4f}")

    # Committee-based query
    query_idx_committee, _ = committee.query(X_pool_committee, n_instances=n_instances_per_query)

    # Get the corresponding instances and labels
    X_query_committee = X_pool_committee[query_idx_committee]
    y_query_committee = y_pool_committee[query_idx_committee]

    # Teach the committee with new labeled instances
    committee.teach(X_query_committee, y_query_committee)

    # Remove queried instances from the committee's pool
    X_pool_committee = np.delete(X_pool_committee, query_idx_committee, axis=0)
    y_pool_committee = np.delete(y_pool_committee, query_idx_committee)

    # Evaluate committee model
    committee_acc = evaluate_model(committee, X_test, y_test)
    committee_accuracies.append(committee_acc)
    print(f"Committee accuracy: {committee_acc:.4f}")

# %% Plot learning curves
plt.figure(figsize=(12, 8))
plt.plot(range(n_queries + 1), entropy_accuracies, "b-", marker="o", label="Entropy Sampling (ActiveLearner)")
plt.plot(range(n_queries + 1), committee_accuracies, "r-", marker="s", label="Committee (Vote Entropy)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel("Number of Queries")
plt.ylabel("Accuracy")
plt.title("Active Learning Accuracy Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("active_learning_comparison.png")
plt.show()


# %% Interactive visualization function for a few sample digits
def interactive_visualization():
    # Randomly select a few samples from the pool
    sample_indices = np.random.choice(len(X_pool), size=5, replace=False)
    samples = X_pool[sample_indices]

    plt.figure(figsize=(15, 3))
    for i, sample in enumerate(samples):
        plt.subplot(1, 5, i + 1)
        plot_digit(sample)

        # Predict with both models
        entropy_pred = entropy_learner.predict(np.array([sample]))[0]
        committee_pred = committee.predict(np.array([sample]))[0]

        plt.title(f"E:{entropy_pred} C:{committee_pred}")

    plt.tight_layout()
    plt.savefig("sample_digits_prediction.png")
    plt.show()


# %% Call the interactive visualization function
interactive_visualization()


# %% Create a function to simulate user labeling
def simulate_labeling(X_sample, model):
    """Simulate a user labeling process by asking the model for predictions and confidence."""
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)

    # Get the confidence for each prediction
    confidences = np.max(probabilities, axis=1)

    return predictions, confidences


# %% Demonstrate the labeling process with a few examples
X_demo = X_pool[:5]  # Take 5 samples for demonstration
entropy_preds, entropy_conf = simulate_labeling(X_demo, entropy_learner)
committee_preds, committee_conf = simulate_labeling(X_demo, committee)

print("\nInteractive Labeling Demonstration:")
for i in range(len(X_demo)):
    plt.figure(figsize=(3, 3))
    plot_digit(X_demo[i])
    plt.savefig(f"digit_sample_{i}.png")
    plt.close()

    print(f"Sample {i + 1}:")
    print(f"  Entropy Model Prediction: {entropy_preds[i]} (Confidence: {entropy_conf[i]:.4f})")
    print(f"  Committee Prediction: {committee_preds[i]} (Confidence: {committee_conf[i]:.4f})")

    # In a real application, here you would ask the user for the true label
    # For simulation, we can use the actual label from y_pool
    print(f"  True Label: {y_pool[i]}")
    print()

# %% Print summary of findings
print("\nSummary of Active Learning Performance:")
print(f"Initial accuracy (Entropy-based ActiveLearner): {initial_entropy_acc:.4f}")
print(f"Final accuracy (Entropy-based ActiveLearner): {entropy_accuracies[-1]:.4f}")
print(f"Improvement: {entropy_accuracies[-1] - initial_entropy_acc:.4f}")
print()
print(f"Initial accuracy (Committee): {initial_committee_acc:.4f}")
print(f"Final accuracy (Committee): {committee_accuracies[-1]:.4f}")
print(f"Improvement: {committee_accuracies[-1] - initial_committee_acc:.4f}")
