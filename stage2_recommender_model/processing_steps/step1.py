import pandas as pd


def drop_rare_users_items_and_engingeer_features(
        path_to_input_data, 
        minimum_user_count, 
        minimum_item_count, 
        path_to_output_data
        ):
    """
    оставляем в датасете только взаимодействия с айтемами, которые заказали не менее minimum_item_count раз (аналогично юзеры)
    """
    print("\n--- step1: normalize_fields_and_drop_rare_users_items ---\n")

    df = pd.read_parquet(path_to_input_data)

    itemcount = df.articleID.value_counts()
    usercount = df.customerID.value_counts()

    print("было строк:", df.shape[0])
    print("было юзеров и айтемов:", df.customerID.nunique(), df.articleID.nunique())

    df = df[df.articleID.isin(itemcount[itemcount >= minimum_item_count].index)].reset_index(drop=True) # удаляем редкие айтемы, так что статистику для юзеров почти не испортит
    df = df[df.customerID.isin(usercount[usercount >= minimum_user_count].index)].reset_index(drop=True)

    print("стало строк:", df.shape[0])
    print("стало юзеров и айтемов:", df.customerID.nunique(), df.articleID.nunique())

    # feature engineering

    df["orderDate"] = pd.to_datetime(df["orderDate"])
    df["was_returned"] = df["returnQuantity"].map(lambda x: 1 if x > 0 else 0)

    df["colorCode"] = df["colorCode"].astype(str) # знаем из EDA что нет пропусков (но вообще при выгрузке на другие даты могут быть пропуски, опасный момент)
    # у sizeCode и так str, всё ок 
    
    df["productGroup"] = df["productGroup"].fillna("missing").astype(str)
    df["voucherID"] = df["voucherID"].fillna("missing").astype(str)

    df["discount"] = df["rrp"] - df["price"]

    df["price_bin"] = pd.qcut(df["price"], q=10, duplicates="drop").astype("object")
    df["price_bin"] = df["price_bin"].where(df["price_bin"].notna(), "missing").astype(str)

    df["discount_bin"] = pd.qcut(df["discount"], q=10, duplicates="drop").astype("object")
    df["discount_bin"] = df["discount_bin"].where(df["discount_bin"].notna(), "missing").astype(str)

    df.to_parquet(path_to_output_data, index=False)
