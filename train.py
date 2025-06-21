from src.data_loader import load_gold_data
from src.data_preprocessing import add_variables, prepare_data_for_training
from src.model_construction import train_model, save_model 
from src.model_evaluation import evaluate_model
from src.data_schema import GoldInput
from src.data_validation import validate_data

# getting API key
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("ALPVAN_API_KEY")

# Input data validation
gold_input = GoldInput(start_date="2022-01-01", end_date="2024-12-31")

# Load and preprocess the data
df_train = load_gold_data(start_date=gold_input.start_date, end_date=gold_input.end_date, api_key=api_key)
df_train = add_variables(df_train)

# Validate the data
validate_data(df_train)

# Prepare data for training
X_train, y_train, X_test, y_test, _ = prepare_data_for_training(df_train, k=3, test_size=0.2)

# Train the model
model = train_model(X_train, y_train)

# Save the model
save_model(model)

# Make predictions and evaluate
preds = model.predict(X_test)
print(evaluate_model(y_test, preds))
