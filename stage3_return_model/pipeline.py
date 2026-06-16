from processing_steps.step0 import load_all_needed_data_from_postgres_to_given_date
from processing_steps.step1 import drop_rare_users_items_and_engingeer_features
from processing_steps.step2 import data_quality
from processing_steps.train_and_gridsearch_and_validate import train_different_models_and_gridsearch_best_and_validate

# load_all_needed_data_from_postgres_to_given_date(
#     processing_date="2015-05-01 00:00:00", 
#     history_window="365 days",
#     path_to_output_data="stage3_return_model/processing_data/step0.parquet"
# )

# drop_rare_users_items_and_engingeer_features(
#     path_to_input_data="stage3_return_model/processing_data/step0.parquet",
#     minimum_user_count=10,
#     minimum_item_count=10,
#     path_to_output_data="stage3_return_model/processing_data/step1.parquet"
# )

# data_quality(
#     path_to_input_data="stage3_return_model/processing_data/step1.parquet"
# )

train_different_models_and_gridsearch_best_and_validate(
    path_to_input_data="stage3_return_model/processing_data/step1.parquet",
    path_to_output_metrics="stage3_return_model/processing_data/metrics.parquet"
)
