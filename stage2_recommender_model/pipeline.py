from processing_steps.step0 import load_all_needed_data_from_postgres_to_given_date
from processing_steps.step1 import drop_rare_users_items_and_engingeer_features
from processing_steps.step2 import infer_external_embeddings_and_append_to_items
from processing_steps.step3 import data_quality
from processing_steps.train_and_predict import train_model_and_make_recommendations
from processing_steps.validate import validate_on_future_window


load_all_needed_data_from_postgres_to_given_date(
    processing_date="2015-05-01 00:00:00", 
    history_window="365 days",
    path_to_output_data="stage2_recommender_model/processing_data/step0.parquet"
)

drop_rare_users_items_and_engingeer_features(
    path_to_input_data="stage2_recommender_model/processing_data/step0.parquet",
    minimum_user_count=10,
    minimum_item_count=10,
    path_to_output_data="stage2_recommender_model/processing_data/step1.parquet"
)

infer_external_embeddings_and_append_to_items(
    path_to_input_data="stage2_recommender_model/processing_data/step1.parquet",
    path_to_placeholder_embeddings="stage2_recommender_model/processing_data/external_item_embeddings.parquet",
    pca_dim=25,
    path_to_output_data="stage2_recommender_model/processing_data/step2.parquet"
)

data_quality(
    path_to_input_data="stage2_recommender_model/processing_data/step2.parquet",
    pca_dim=25
)


# делаем разные реки разными подходами по очереди:

# models: ["random", "popular", "logreg", "DSSM"]
# use_features: [True, False]
chosen_model = "DSSM"
use_features = True
train_model_and_make_recommendations(
    path_to_input_data="stage2_recommender_model/processing_data/step2.parquet",
    model=chosen_model,
    use_features=use_features,
    path_to_output_data=f"stage2_recommender_model/processing_data/recommendations_{chosen_model}{'_with_features' if use_features else ''}.parquet"
)

# валидируем разные комбинации которые нас интересуют:

validate_on_future_window(
    processing_date="2015-05-01 00:00:00",
    future_window="30 days",
    list_of_paths_to_different_recommendations=[
        # "stage2_recommender_model/processing_data/recommendations_popular.parquet",
        "stage2_recommender_model/processing_data/recommendations_DSSM.parquet",
        "stage2_recommender_model/processing_data/recommendations_logreg.parquet",
        ],
    output_metrics_file_name="stage2_recommender_model/result_metrics/union_DSSMLogreg_metrics.json"
)