

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# -------------------------------------------------
# 1. Feature Simulation (Blink-based Integrity)

def extract_features(num_samples=100):
    """
    Feature Vector:
    [eye_blink_rate, alignment_error]

    Label:
    0 = Real
    1 = Deepfake
    """
    X = []
    y = []

    # Real videos
    for _ in range(num_samples // 2):
        blink_rate = np.random.randint(4, 7)          # natural blinking
        ae = np.random.uniform(0.0, 0.02)              # low alignment error
        X.append([blink_rate, ae])
        y.append(0)

    # Deepfake videos
    for _ in range(num_samples // 2):
        blink_rate = np.random.randint(0, 3)           # abnormal blinking
        ae = np.random.uniform(0.02, 0.08)             # higher alignment error
        X.append([blink_rate, ae])
        y.append(1)

    return np.array(X), np.array(y)


# 2. Rule-Based Decision
def rule_based_decision(prob, blink_rate, ae):
    if prob >= 0.5 and blink_rate >= 4 and ae < 0.02:
        return "Real"
    else:
        return "Deepfake"


# 3. Main Execution

if __name__ == "__main__":

    # Generate dataset
    X, y = extract_features(num_samples=120)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train SVM classifier
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    # Evaluate performance
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

 
    # Single Video Demonstration

    blink_rate, ae = X_test[0]
    prob_real = model.predict_proba([[blink_rate, ae]])[0][0]
    final_label = rule_based_decision(prob_real, blink_rate, ae)

    print("\n--- Sample Video Classification ---")
    print(f"Eye Blink Rate     : {blink_rate}")
    print(f"Alignment Error AE : {ae:.4f}")
    print(f"Probability (Real) : {prob_real:.2f}")
    print(f"Final Decision     : {final_label}")
