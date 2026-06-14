import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, PCA


def placeholder_embeddings_creator_if_not_exist_yet(path_to_input_data, tsvd_dim, path_to_placeholder_embeddings):

    print("внешних эмбеддингов не существует, создадим и затем будем имитировать, будто получили их от внешней triton модели")

    df = pd.read_parquet(path_to_input_data)

    # превращаем айдишники в номера для разреженной матрицы
    users = sorted(df["customerID"].unique())
    items = sorted(df["articleID"].unique())
    user2id = {user: i for i, user in enumerate(users)}
    item2id = {item: i for i, item in enumerate(items)}

    df["customerID"] = df["customerID"].map(user2id)
    df["articleID"] = df["articleID"].map(item2id)

    users_list = df["customerID"].tolist()
    items_list = df["articleID"].tolist()
    data = [1] * len(df)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    matrix = csr_matrix((data, (items_list, users_list)), shape=(len(items), len(users)))

    # айтемы ставили первыми, чтобы fit_transform дал эмбеды именно для айтемов

    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    tsvd = TruncatedSVD(n_components=tsvd_dim, random_state=0)
    item_embeddings = tsvd.fit_transform(matrix)
    print("матрица эмбеддингов айтемов: ", item_embeddings.shape)

    result = pd.DataFrame(
        item_embeddings,
        columns=[f"emb{i}" for i in range(tsvd_dim)]
    )

    result["articleID"] = items

    # print("сколько дисперсии объясняет это число компонент: ", tsvd.explained_variance_ratio_.sum())

    result.to_parquet(path_to_placeholder_embeddings, index=False)

    
def imitate_triton_inference(path_to_input_data, path_to_placeholder_embeddings):

    print("имитируем, будто получили контентные эмбеддинги айтемов от внешней triton модели, чтобы обогатить айтемы новыми фичами")

    # get item_ids -> get images -> embeds = triton.infer(images)

    df = pd.read_parquet(path_to_input_data) # как будто взяли айдишники из df и прогоняем их через модель

    embeds = pd.read_parquet(path_to_placeholder_embeddings)

    item2emb = embeds.set_index("articleID").to_dict(orient="index")
    item2emb = {k: v.values() for k, v in item2emb.items()}

    return item2emb


def infer_external_embeddings_and_append_to_items(
        path_to_input_data, 
        path_to_placeholder_embeddings, 
        pca_dim, 
        path_to_output_data
        ):

    print("\n--- step2: infer_external_embeddings_and_append_to_items ---\n")

    placeholder_embeddings_for_items_exist = os.path.exists(path_to_placeholder_embeddings)

    if not placeholder_embeddings_for_items_exist:
        placeholder_embeddings_creator_if_not_exist_yet(path_to_input_data, pca_dim*8, path_to_placeholder_embeddings)

    item2emb = imitate_triton_inference(path_to_input_data, path_to_placeholder_embeddings)
    print("получили эмбеддинги от внешней triton модели, которая (типа) делает контентные эмбеддинги для товаров по их изображениям")

    embeddings_matrix = np.array([list(x) for x in item2emb.values()])

    pca = PCA(n_components=pca_dim, random_state=0)
    pca_embeddings = pca.fit_transform(embeddings_matrix)

    print(f"""сколько дисперсии оригинальных эмбеддингов (считаем их полезными контентными эмбедами, хотя на самом деле это 
          просто слабенький SVD по матрице взаимодействий) объясняет {pca_dim} компонент в PCA: """, pca.explained_variance_ratio_.sum())

    pca_df = pd.DataFrame(
        pca_embeddings,
        columns=[f"external_emb_pca_{i}" for i in range(pca_dim)]
    )
    pca_df["articleID"] = list(item2emb.keys())

    df = pd.read_parquet(path_to_input_data)
    df = df.merge(pca_df, on="articleID", how="left")
    df.to_parquet(path_to_output_data, index=False)
