import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def train_model_and_make_recommendations(path_to_input_data, model, use_features, path_to_output_data):
    print("\n--- step4: train_model_and_make_recommendations ---\n")

    possible_models = ["random", "popular", "logreg", "DSSM"]
    assert model in possible_models
    if model == "DSSM":
        train_dssm_and_make_recommendations(path_to_input_data, use_features, path_to_output_data)
    elif model == "random":
        make_random_recommendations(path_to_input_data, path_to_output_data)
    elif model == "popular":
        make_popular_recommendations(path_to_input_data, path_to_output_data)
    elif model == "logreg":
        train_logreg_and_make_recommendations(path_to_input_data, path_to_output_data)
    else:
        assert False

def make_random_recommendations(path_to_input_data, path_to_output_data):
    print(f"\n--- recommend random (bottom baseline) ---\n")

    train = pd.read_parquet(path_to_input_data)
    users = sorted(train["customerID"].unique())
    items = sorted(train["articleID"].unique())

    recommendations = dict()
    for user in users:
        recommendations[user] = random.choices(items, k=200)

    result = pd.DataFrame(recommendations.items(), columns=["customerID", "recommendations"])
    result.to_parquet(path_to_output_data, index=False)

def make_popular_recommendations(path_to_input_data, path_to_output_data):
    print(f"\n--- recomend popular (better baseline) ---\n")

    train = pd.read_parquet(path_to_input_data)
    users = sorted(train["customerID"].unique())
    items = train["articleID"].value_counts().head(200).index.tolist()

    recommendations = dict()
    for user in users:
        recommendations[user] = items

    result = pd.DataFrame(recommendations.items(), columns=["customerID", "recommendations"])
    result.to_parquet(path_to_output_data, index=False)

def train_logreg_and_make_recommendations(path_to_input_data, path_to_output_data):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression

    print(f"\n--- logreg (текущее продакшн решение, с которым соревнуемся) ---\n")

    train = pd.read_parquet(path_to_input_data)
    users = sorted(train["customerID"].unique())
    items = sorted(train["articleID"].unique())

    # готовим (исходники) фичей
    user2pop = train["customerID"].value_counts().to_dict()
    item2pop = train["articleID"].value_counts().to_dict()
    item2price = train.groupby("articleID")["price"].mean().fillna(0).to_dict()

    # готовим таргеты (негатив семплинг) - позитивы это реальные заказы, негативы это юзер+рандомный айтем 
    positives = train[["customerID", "articleID"]]
    positives["target"] = 1

    negatives = train[["customerID", "articleID"]]
    negatives["articleID"] = negatives["articleID"].values[::-1] # переворачиваем колонку айтемов, полностью ломая пары юзер-айтем. Получается почти гарантированный рандом
    negatives["target"] = 0

    # готовим датасет для обучения и уже делаем нормальные фичи
    train = pd.concat([positives, negatives], ignore_index=True)
    train["user_popularity"] = train["customerID"].map(user2pop)
    train["item_popularity"] = train["articleID"].map(item2pop)
    train["item_price"] = train["articleID"].map(item2price)

    # знаем, что популярность (число встречаний в датасете) и цена распределены не равномерно. поэтому перед минмаксом логарифмнем их
    train["user_popularity"] = np.log(train["user_popularity"] + 0.1)
    train["item_popularity"] = np.log(train["item_popularity"] + 0.1)
    train["item_price"] = np.log(train["item_price"] + 0.1)

    up_scaler = MinMaxScaler()
    ip_scaler = MinMaxScaler()
    price_scaler = MinMaxScaler()

    train["user_popularity"] = up_scaler.fit_transform(train[["user_popularity"]])
    train["item_popularity"] = ip_scaler.fit_transform(train[["item_popularity"]])
    train["item_price"] = price_scaler.fit_transform(train[["item_price"]])

    model = LogisticRegression(max_iter=10)
    model.fit(train[["user_popularity", "item_popularity", "item_price"]], train["target"])

    candidates = {}
    for user in tqdm(users):
        test = pd.DataFrame({
            "customerID": [user] * len(items),
            "articleID": items,
        })

        test["user_popularity"] = test["customerID"].map(user2pop)
        test["item_popularity"] = test["articleID"].map(item2pop)
        test["item_price"] = test["articleID"].map(item2price)

        test["user_popularity"] = np.log(test["user_popularity"] + 0.1)
        test["item_popularity"] = np.log(test["item_popularity"] + 0.1)
        test["item_price"] = np.log(test["item_price"] + 0.1)

        test["user_popularity"] = up_scaler.transform(test[["user_popularity"]])
        test["item_popularity"] = ip_scaler.transform(test[["item_popularity"]])
        test["item_price"] = price_scaler.transform(test[["item_price"]])

        test["score"] = model.predict_proba(test[["user_popularity", "item_popularity", "item_price"]])[:, 1]
        candidates[user] = test.sort_values("score", ascending=False)["articleID"].head(200).to_list()

    result = pd.DataFrame(candidates.items(), columns=["customerID", "recommendations"])

    result.to_parquet(path_to_output_data)

def train_dssm_and_make_recommendations(path_to_input_data, use_features, path_to_output_data):
    from info_nce import InfoNCE
    import faiss
    from sklearn.preprocessing import normalize
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    print(f"\n--- DSSM ---\n")

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    if device == "cuda":
        print("torch.cuda.get_device_name(): ", torch.cuda.get_device_name())
    else:
        print("torch.cuda.get_device_name(): CUDA is not available")

    train = pd.read_parquet(path_to_input_data)

    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    train["user_id"] = train["customerID"]
    train["item_id"] = train["articleID"]

    # фичи энкодим сразу, чтобы не раздуть код TwoTowerDataset
    categorical_feature_columns = [
        "colorCode",
        "sizeCode",
        "productGroup",
        "price_bin",
        "discount_bin",
    ]
    external_embedding_columns = [
        column for column in train.columns if column.startswith("external_emb_pca_")
    ]
    categorical_feature_encoders = {}
    categorical_feature_sizes = {}

    if use_features:
        for column in categorical_feature_columns:
            # train[column] = train[column].astype(str)  уже конвертировали в str на этапе подготовки фичей
            categorical_feature_encoders[column] = LabelEncoder().fit(train[column])
            train[f"encoded_{column}"] = categorical_feature_encoders[column].transform(train[column])
            categorical_feature_sizes[column] = len(categorical_feature_encoders[column].classes_)

        train[external_embedding_columns] = train[external_embedding_columns].astype("float32")

    print("1. готовим торчовые датасеты, енкодеры, даталоадеры")

    user_encoder = LabelEncoder().fit(train["user_id"])
    item_encoder = LabelEncoder().fit(train["item_id"])

    class TwoTowerDataset(Dataset):

        def __init__(self,
                    positive_interactions_dataframe,
                    user_encoder, 
                    item_encoder
                    ):
            self.positive_interactions_dataframe = positive_interactions_dataframe.copy()
            
            self.positive_interactions_dataframe["encoded_user_id"] = user_encoder.transform(self.positive_interactions_dataframe.user_id)
            self.positive_interactions_dataframe["encoded_item_id"] = item_encoder.transform(self.positive_interactions_dataframe.item_id)
            
            self.user2features = {}
            self.item2features = {}
            self.interactions_features_list = []

            for row in tqdm(self.positive_interactions_dataframe.itertuples()):
                self.interactions_features_list.append(
                    (
                        (row.encoded_user_id,),
                        (row.encoded_item_id,)
                    )
                )
                if row.user_id not in self.user2features:
                    self.user2features[row.user_id] = (row.encoded_user_id,)

                if row.item_id not in self.item2features:
                    self.item2features[row.item_id] = (row.encoded_item_id,)

            self.start = 0
            self.end = len(self.interactions_features_list)
        
        def __len__(self):
            return len(self.interactions_features_list)
        
        def __getitem__(self, i):
            return self.interactions_features_list[i]

    class TwoTowerDatasetWithFeatures(Dataset):

        def __init__(self,
                    positive_interactions_dataframe,
                    user_encoder, 
                    item_encoder
                    ):
            self.positive_interactions_dataframe = positive_interactions_dataframe.copy()
            
            self.positive_interactions_dataframe["encoded_user_id"] = user_encoder.transform(self.positive_interactions_dataframe.user_id)
            self.positive_interactions_dataframe["encoded_item_id"] = item_encoder.transform(self.positive_interactions_dataframe.item_id)
            
            self.user2features = {}
            self.item2features = {}
            self.interactions_features_list = []

            for row in tqdm(self.positive_interactions_dataframe.itertuples()):
                item_features = (
                    row.encoded_item_id,
                    row.encoded_colorCode,
                    row.encoded_sizeCode,
                    row.encoded_productGroup,
                    row.encoded_price_bin,
                    row.encoded_discount_bin,
                )
                item_external_features = tuple(
                    getattr(row, column) for column in external_embedding_columns
                )
                self.interactions_features_list.append(
                    (
                        (row.encoded_user_id,),
                        item_features,
                        item_external_features, # отдельно, потому что его придётся держать во float в отличие от предыдущих
                    )
                )
                if row.user_id not in self.user2features:
                    self.user2features[row.user_id] = (row.encoded_user_id,)

                if row.item_id not in self.item2features:
                    self.item2features[row.item_id] = (
                        item_features,
                        item_external_features,
                    )

            self.start = 0
            self.end = len(self.interactions_features_list)
        
        def __len__(self):
            return len(self.interactions_features_list)
        
        def __getitem__(self, i):
            return self.interactions_features_list[i]

    train_dataset = TwoTowerDatasetWithFeatures(train, user_encoder, item_encoder) if use_features else TwoTowerDataset(train, user_encoder, item_encoder)

    shuffle = True
    num_workers = 0
    batch_size = 4096

    # как Dataset должен возвращать результаты 
    def collate_fn(batch):
        if use_features:
            users_features, items_features, item_external_features = zip(*batch)

            return (
                torch.IntTensor(np.array(users_features)),
                torch.IntTensor(np.array(items_features)),
                torch.FloatTensor(np.array(item_external_features)),
            )
        else:
            users_features, items_features = zip(*batch)

            return (
                torch.IntTensor(np.array(users_features)),
                torch.IntTensor(np.array(items_features)),
            )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        generator=g
    )

    print("2. описываем модель DSSM")

    class TwoTowerModel(nn.Module):
        def __init__(self, user_embedding_sizes, item_embedding_sizes, device):
            super(TwoTowerModel, self).__init__()
            self.device = device

            self.user_embeds = nn.Embedding(user_embedding_sizes[0], user_embedding_sizes[1]).double()
            self.item_embeds = nn.Embedding(item_embedding_sizes[0], item_embedding_sizes[1]).double()

        def get_user_embeddings(self, user_features):
            user_embeddings = self.user_embeds(user_features[:, 0])
            return user_embeddings

        def get_item_embeddings(self, item_features):
            item_embeddings = self.item_embeds(item_features[:, 0])
            return item_embeddings

        def forward(self, user_features, item_features):
            user_embs = self.get_user_embeddings(user_features.to(self.device))
            item_embs = self.get_item_embeddings(item_features.to(self.device))

            return user_embs, item_embs
        
    class TwoTowerModelWithFeatures(nn.Module):
        def __init__(
                self,
                user_embedding_sizes,
                item_embedding_sizes,
                device,
                categorical_feature_sizes,
                external_embedding_dim
                ):
            super(TwoTowerModelWithFeatures, self).__init__()
            self.device = device

            self.user_embeds = nn.Embedding(user_embedding_sizes[0], user_embedding_sizes[1]).double()
            self.item_embeds = nn.Embedding(item_embedding_sizes[0], item_embedding_sizes[1]).double()

            self.color_embeds = nn.Embedding(categorical_feature_sizes["colorCode"], item_embedding_sizes[1]).double()
            self.size_embeds = nn.Embedding(categorical_feature_sizes["sizeCode"], item_embedding_sizes[1]).double()
            self.product_group_embeds = nn.Embedding(categorical_feature_sizes["productGroup"], item_embedding_sizes[1]).double()
            self.price_bin_embeds = nn.Embedding(categorical_feature_sizes["price_bin"], item_embedding_sizes[1]).double()
            self.discount_bin_embeds = nn.Embedding(categorical_feature_sizes["discount_bin"], item_embedding_sizes[1]).double()
            self.external_embeddings_projection = nn.Linear(external_embedding_dim, item_embedding_sizes[1]).double()

        def get_user_embeddings(self, user_features):
            user_embeddings = self.user_embeds(user_features[:, 0])
            return user_embeddings

        def get_item_embeddings(self, item_features, item_external_features):
            item_embeddings = self.item_embeds(item_features[:, 0])

            color_embeddings = self.color_embeds(item_features[:, 1])
            size_embeddings = self.size_embeds(item_features[:, 2])
            product_group_embeddings = self.product_group_embeds(item_features[:, 3])
            price_bin_embeddings = self.price_bin_embeds(item_features[:, 4])
            discount_bin_embeddings = self.discount_bin_embeds(item_features[:, 5])
            external_embeddings = self.external_embeddings_projection(item_external_features.double())
            item_embeddings = (
                item_embeddings
                + color_embeddings
                + size_embeddings
                + product_group_embeddings
                + price_bin_embeddings
                + discount_bin_embeddings
                + external_embeddings
            )

            return item_embeddings

        def forward(self, user_features, item_features, item_external_features):
            user_embs = self.get_user_embeddings(user_features.to(self.device))
            item_external_features = item_external_features.to(self.device)
            item_embs = self.get_item_embeddings(
                item_features.to(self.device),
                item_external_features,
            )

            return user_embs, item_embs

    if device == "cuda":
        torch.cuda.empty_cache()

    user_embedding_sizes = [len(user_encoder.classes_), 50]
    item_embedding_sizes = [len(item_encoder.classes_), 50]

    if use_features:
        model = TwoTowerModelWithFeatures(
            user_embedding_sizes,
            item_embedding_sizes,
            device,
            categorical_feature_sizes,
            len(external_embedding_columns),
        ).to(device)
    else:
        model = TwoTowerModel(user_embedding_sizes, item_embedding_sizes, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-7)

    infonceloss = InfoNCE(temperature=0.1)

    print("3. учим модель DSSM")

    def train_loop(model, optimizer, train_loader, n_epochs=10):
        for epoch in range(n_epochs):
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
                
                user_embs, item_embs = model(*batch)
                loss = infonceloss(user_embs, item_embs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return

    model.train()
    train_loop(model, optimizer, train_loader)

    print("4. делаем рекомендации для всех юзеров из трейна")

    # юзерные эмбеддинги
    users_ids, users_features = zip(*train_dataset.user2features.items())
    
    users_embeddings = model.get_user_embeddings(
        torch.IntTensor(np.array(users_features)).to(device)
    ).cpu().detach().numpy()
    
    user2idx = {user_id: idx for idx, user_id in enumerate(users_ids)}
    idx2user = {idx: user_id for user_id, idx in user2idx.items()}

    # айтемные эмбеддинги
    items_ids, items_all_features = zip(*train_dataset.item2features.items())

    if use_features:
        items_features, items_external_features = zip(*items_all_features)
        items_external_features = torch.FloatTensor(np.array(items_external_features)).to(device)
    else:
        items_features = items_all_features

    if use_features:
        items_embeddings = model.get_item_embeddings(
            torch.IntTensor(np.array(items_features)).to(device),
            items_external_features
        ).cpu().detach().numpy()
    else:
        items_embeddings = model.get_item_embeddings(
            torch.IntTensor(np.array(items_features)).to(device)
        ).cpu().detach().numpy()

    item2idx = {item_id: idx for idx, item_id in enumerate(items_ids)}
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}

    # делаем рекомендации

    def get_candidates(
        users2infer: np.array,
        user_factors: np.array,
        item_factors: np.array,
        idx2user: dict,
        idx2item: dict,
        n_candidates: int,
        l2_normalize=True
    ):
        user_factors = user_factors[users2infer]

        user_factors = user_factors.astype("float32")
        item_factors = item_factors.astype("float32")

        if l2_normalize:
            user_factors = normalize(user_factors, axis=1, norm='l2')
            item_factors = normalize(item_factors, axis=1, norm='l2')

        index = faiss.IndexFlatIP(item_factors.shape[1])
        
        index.add(item_factors)
        distances, preds = index.search(user_factors, n_candidates)
        
        candidates = {}
        for idx, recs in zip(users2infer, preds):
            user = idx2user[idx]
            recs = [idx2item[i] for i in recs]
            
            candidates[user] = recs
            
        return candidates
    
    all_users_idx = np.array(list(user2idx.values()))

    candidates = get_candidates(
        all_users_idx,
        users_embeddings,
        items_embeddings,
        idx2user,
        idx2item,
        200,
        l2_normalize=False
    )

    result = pd.DataFrame(candidates.items(), columns=["customerID", "recommendations"])

    result.to_parquet(path_to_output_data)
