# streamlit_app.py
import streamlit as st
import joblib
import os
from core import (
    load_and_process, split_train_test, run_grid_search,
    evaluate_model, plot_confusion_matrix, plot_decision_boundary
)

MODEL_PATH = "saved_model.joblib"


def main():
    st.set_page_config(page_title="Fish Disease Prediction", layout="wide")
    st.title("IoT Tilapia Fish Disease Prediction")

    # Sidebar controls
    file = st.sidebar.file_uploader("Upload Dataset (Excel)", type=["xlsx"])
    window_size = st.sidebar.number_input(
        "Test Window Size", min_value=50, max_value=1000, value=200, step=50)
    step = st.sidebar.number_input(
        "Step Size", min_value=10, max_value=500, value=50, step=10)
    param_C = st.sidebar.multiselect("C values", [0.1, 1, 10], default=[0.1, 1, 10])
    param_gamma = st.sidebar.multiselect("Gamma options", ['scale', 'auto'], default=['scale', 'auto'])
    param_kernel = st.sidebar.multiselect("Kernel types", ['rbf', 'sigmoid'], default=['rbf', 'sigmoid'])
    run = st.sidebar.button("Train & Save Model")
    load = st.sidebar.button("Load Saved Model")

    if not file:
        st.warning("Upload the dataset to proceed.")
        return

    # Load and process data
    df, X, y = load_and_process(file)
    st.subheader("Dataset Summary")
    st.write(f"Total samples: {len(df)}")

    try:
        X_train, X_test, y_train, y_test = split_train_test(X, y, window_size, step)
    except ValueError as e:
        st.error(str(e))
        return

    st.write(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    st.write("Class distribution (train):")
    st.write(y_train.value_counts(normalize=True))
    st.write("Class distribution (test):")
    st.write(y_test.value_counts(normalize=True))

    if run:
        param_grid = {
            'svm__C': param_C,
            'svm__gamma': param_gamma,
            'svm__kernel': param_kernel
        }
        st.info("Running grid search and saving model...")
        grid = run_grid_search(X_train, y_train, param_grid)
        joblib.dump(grid.best_estimator_, MODEL_PATH)
        st.success("Model trained and saved!")

    if load:
        if not os.path.exists(MODEL_PATH):
            st.error("Saved model not found. Please train and save a model first.")
            return
        best = joblib.load(MODEL_PATH)
        st.success("Loaded saved model!")

        metrics = evaluate_model(best, X_test, y_test)

        st.subheader("Performance Metrics")
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.metric("Recall", f"{metrics['recall']:.2%}")
        st.metric("F1 Score", f"{metrics['f1']:.2%}")

        st.subheader("Classification Report")
        st.text(metrics['classification_report'])

        st.subheader("Confusion Matrix")
        st.pyplot(plot_confusion_matrix(metrics['confusion_matrix']))

        st.subheader("Decision Boundary")
        st.pyplot(plot_decision_boundary(best, X_train, y_train, X_test, y_test))


if __name__ == "__main__":
    main()
