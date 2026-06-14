import pandas as pd
import great_expectations as gx

# по гайду https://www.datacamp.com/tutorial/great-expectations-tutorial

def data_quality(path_to_input_data, pca_dim):
    print("\n--- step3: data_quality ---\n")
    
    # подготовка данных 

    df = pd.read_parquet(path_to_input_data)
    embedding_columns = [f"external_emb_pca_{i}" for i in range(pca_dim)]
    df["emb_abs_sum"] = df[embedding_columns].abs().sum(axis=1)

    expected_columns = [
        "orderID",
        "orderDate",
        "articleID",
        "colorCode",
        "sizeCode",
        "productGroup",
        "quantity",
        "price",
        "rrp",
        "voucherID",
        "voucherAmount",
        "customerID",
        "deviceID",
        "paymentMethod",
        "returnQuantity",
        "was_returned",
        "discount",
        "price_bin",
        "discount_bin"
    ] + embedding_columns + ["emb_abs_sum"]

    # фреймворк great_expectations

    context = gx.get_context()

    data_source = context.data_sources.add_pandas(name="recommender_model_data")
    data_asset = data_source.add_dataframe_asset(name="recommender_model_data_asset")

    batch_definition_name = "recommender_model_data_batch"
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)

    batch_parameters = {"dataframe": df}
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    # проверки

    results = []

    results.append(batch.validate(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=expected_columns,
            exact_match=True,
        )
    ))

    # проверки типов

    str_columns = [
        "orderID",
        "articleID",
        "colorCode",
        "sizeCode",
        "voucherID",
        "customerID",
        "paymentMethod",
        "productGroup",
        "price_bin",
        "discount_bin"
    ]
    for column in str_columns:
        results.append(batch.validate(
            gx.expectations.ExpectColumnValuesToBeOfType(
                column=column,
                type_="str",
            )
        ))

    int_columns = [
        "quantity",
        "deviceID",
        "returnQuantity",
        "was_returned",
    ]
    for column in int_columns:
        results.append(batch.validate(
            gx.expectations.ExpectColumnValuesToBeInTypeList(
                column=column,
                type_list=["int64"],
            )
        ))

    float_columns = [
        "price",
        "rrp",
        "discount",
        "voucherAmount",
        "emb_abs_sum",
    ] + embedding_columns
    for column in float_columns:
        results.append(batch.validate(
            gx.expectations.ExpectColumnValuesToBeInTypeList(
                column=column,
                type_list=["float64"],
            )
        ))

    # проверки значений

    for column in ["orderID", "orderDate", "customerID", "articleID", "colorCode", "sizeCode", "productGroup", "price_bin", "discount_bin"]:
        results.append(batch.validate(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        ))

    results.append(batch.validate(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="quantity",
            min_value=0,
        )
    ))

    results.append(batch.validate(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="price",
            min_value=0,
        )
    ))

    results.append(batch.validate(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="was_returned",
            min_value=0,
            max_value=1,
        )
    ))

    results.append(batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="emb_abs_sum"
        )
    ))

    results.append(batch.validate(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="emb_abs_sum",
            min_value=0,
            strict_min=True,
        )
    ))

    if not all(result.success for result in results):
        raise ValueError("data quality failed")
    
    print("data quality passed")
