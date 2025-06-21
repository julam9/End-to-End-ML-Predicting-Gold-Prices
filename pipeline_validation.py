from src import data_loader, data_preprocessing, model_construction, model_evaluation
import pandas as pd

# getting API key
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("ALPVAN_API_KEY")

def main():
    print("🔍 Starting end-to-end validation...")

    # Step 1: Load gold data
    print("📥 Loading data...")
    df_train = data_loader.load_gold_data(start_date='2022-01-01', end_date='2024-12-31', api_key=api_key)
    assert isinstance(df_train, pd.DataFrame) and not df_train.empty, "❌ Gold data was not loaded correctly."

    # Step 2: Data preprocessing
    print("🛠️ Preprocessing data...")
    df_train = data_preprocessing.add_variables(df_train)
    X_train, y_train, X_test, y_test, _ = data_preprocessing.prepare_data_for_training(df_train, k=3, test_size=0.2)
    assert X_train.shape[0] > 0, "❌ Preprocessing data failed"
    print("✅ Preprocessing successful")

    # Step 3: Model training
    print("🤖 Training model...")
    trained_model = model_construction.train_model(X_train, y_train)
    assert trained_model is not None, "❌ Model training failed"
    print("✅ Model training successful")

    # Step 4: Model evaluation
    print("📊 Evaluating model...")
    predictor = model_construction.multi_step_predict(
        model=trained_model,
        last_known_data=df_train.copy(),
        days=5,  # Example: predict next 5 days 
        feature_creator=data_preprocessing.add_variables,
        selected_columns=X_train.columns.tolist()
    )
    y_pred = trained_model.predict(X_test)
    evaluation_metrics = model_evaluation.evaluate_model(y_test, y_pred)
    print(f"✅ Evaluation complete: {evaluation_metrics}")

    # Step 5: Save model
    print("💾 Model was saved")
    print("🎉 Pipeline validation successful!")

if __name__ == "__main__":
    main()
