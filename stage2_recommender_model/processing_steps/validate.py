import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from processing_steps.sql import step0_query
import json

# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])




def validate_on_future_window(processing_date, future_window, list_of_paths_to_different_recommendations, output_metrics_file_name):
    print("\n--- step5: validate_on_future_window ---\n")

    # в реальности делали бы load dotenv и креды из Vault
    PG_USER = "aaa"
    PG_PASSWORD = "111"
    POSTGRES_AUTH_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@localhost:5433/ml_course_project"

    engine = create_engine(POSTGRES_AUTH_STRING)

    today_date = pd.to_datetime(processing_date)
    end_date = today_date + pd.Timedelta(future_window)

    query = step0_query.format(start_date=today_date, end_date=end_date)

    df = pd.read_sql(query, engine)

    recommendations = []
    for path in list_of_paths_to_different_recommendations:
        recs = pd.read_parquet(path)

        recommendations.append(
            recs.set_index("customerID")["recommendations"].to_dict()
        )

    users = set()
    for recs in recommendations:
        users.update(recs.keys())
    print(f"рекомендации подготовлены для {len(users)} юзеров")

    # если пришёл только один источник рекомендаций, берём все его 200 рекомендаций. Если 2 - то 100 первых от 1го и 100 первых от 2го и т.д.
    num_of_sources = len(recommendations)

    truncated_recommendations = []
    for recs in recommendations:
        truncated_recommendations.append(
            {k: [x for x in v[:(200//num_of_sources)]] for k, v in recs.items()} # закастовал v[:(200//num_of_sources)] из ndarray в список
        )

    union_recommendations = dict()
    for user in users:
        for recs in truncated_recommendations:
            if user in union_recommendations:
                union_recommendations[user].extend(recs[user])
            else:
                union_recommendations[user] = recs[user]
    
    user_count = df.groupby("customerID")["articleID"].count()
    good_users = set(user_count[user_count >= 5].index)
    users_to_validate = sorted(users & good_users)

    print(f"сделано рекомендаций для {len(users)} юзеров")
    print(f"будем валидироваться по {len(users_to_validate)} юзерам")

    df = df[df.customerID.isin(users_to_validate)]

    actual = []
    predicted = []
    user_metrics = []
    all_recommended_items = []

    future_items = df.groupby("customerID")["articleID"].agg(lambda x: list(set(x))).to_dict()
    future_spend = df.groupby("customerID")["price"].sum().fillna(0).to_dict()
    item_price = df.groupby("articleID")["price"].median().fillna(0).to_dict()

    for user in users_to_validate:
        actual_items = future_items[user]
        predicted_items = union_recommendations[user]
        hits = [item for item in predicted_items if item in actual_items]
        hit_revenue = sum(item_price[item] for item in hits)

        actual.append(actual_items)
        predicted.append(predicted_items)
        all_recommended_items.extend(predicted_items)

        user_metrics.append({
            "customerID": user,
            "actual_items_count": len(actual_items),
            "hits_count": len(hits),
            "hit_revenue": hit_revenue,
            "future_spend": future_spend[user],
            "spend_coverage": hit_revenue / future_spend[user] if future_spend[user] > 0 else 0,
        })

    item_counts = pd.Series(all_recommended_items).value_counts()
    recommendation_slots = len(all_recommended_items)
    unique_items = item_counts.shape[0]
    hit_revenues = [x["hit_revenue"] for x in user_metrics]
    spend_coverages = [x["spend_coverage"] for x in user_metrics]

    metrics = {
        "k": 200,
        "users_to_validate": len(users_to_validate),
        "mapk": mapk(actual, predicted, 200),
        "hit_revenue_per_user": np.mean(hit_revenues) if hit_revenues else 0,
        "total_hit_revenue": np.sum(hit_revenues) if hit_revenues else 0,
        "spend_coverage": np.mean(spend_coverages) if spend_coverages else 0,
        "unique_items": unique_items,
        "item_coverage_per_slot": unique_items / recommendation_slots if recommendation_slots > 0 else 0,
        "top_item_share": item_counts.iloc[0] / recommendation_slots if recommendation_slots > 0 else 0,
    }

    for k, v in metrics.items():
        print(k, ": ", v) 

    with open(output_metrics_file_name, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
