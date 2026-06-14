import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from processing_steps.sql import step0_query


def load_all_needed_data_from_postgres_to_given_date(processing_date, history_window, path_to_output_data):
    """ 
    вытаскиваем данные о покупках из postgres 
    """
    print("\n--- step0: load_all_needed_data_from_postgres_to_given_date ---\n")
    
    # в реальности делали бы load dotenv и креды из Vault
    PG_USER = "aaa"
    PG_PASSWORD = "111"
    POSTGRES_AUTH_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@localhost:5433/ml_course_project"

    engine = create_engine(POSTGRES_AUTH_STRING)

    today_date = pd.to_datetime(processing_date)
    start_date = today_date - pd.Timedelta(history_window)

    query = step0_query.format(start_date=start_date, end_date=today_date)

    df = pd.read_sql(query, engine)
    print(f"строк выгружено за период {start_date} - {today_date}:", df.shape[0])
    print(f"айтемов: {df.articleID.nunique()}, юзеров: {df.customerID.nunique()}")

    print("доля юзеров, у которых меньше 10 заказов:", (df.groupby("customerID")["articleID"].count() < 10).mean())
    print("доля айтемов, которые заказали меньше 10 раз:", (df.groupby("articleID")["customerID"].count() < 10).mean())

    df.to_parquet(path_to_output_data, index=False)
