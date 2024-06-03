import pandas as pd
import numpy as np

from sklearn.preprocessing import TargetEncoder


drop_columns =  [
                "client_id",
                "использование",
                "pack",
                "mrg_",
                "pack_freq",
                "зона_2",
                "зона_1",
                "сумма",
                "частота_пополнения",
                "продукт_2",
                ]

class Predproc:
    def __init__(self):
        self.region_encoder = TargetEncoder(random_state=42, target_type="binary")
        self.median_dict = {}
        self.region_dict = {}

    def fit(self, train_path:str):
        df = pd.read_csv(train_path)
        self.region_encoder.fit(df[["регион"]], df["binary_target"])
        self.median_dict = {
            "сумма": np.mean(df["сумма"]),
            "частота_пополнения": np.mean(df["частота_пополнения"]),
            "доход": np.mean(df["доход"]),
            "сегмент_arpu": np.mean(df["сегмент_arpu"]),
            "частота": np.mean(df["частота"]),
            "объем_данных": np.mean(df["объем_данных"]),
            "on_net": np.mean(df["on_net"]),
            "продукт_1": np.mean(df["продукт_1"]),
            "продукт_2": np.mean(df["продукт_2"]),
            "секретный_скор": np.mean(df["секретный_скор"]),
            "pack_freq": np.mean(df["pack_freq"]),
        }

        df[["регион_encode"]] = self.region_encoder.transform(df[["регион"]])
        self.region_dict = dict(df.groupby("регион")["регион_encode"].mean())
        df.drop(["регион_encode"], axis=1, inplace=True)

    def transform_data_frame(self, df:pd.DataFrame):
        df.drop(drop_columns,
            axis=1,
            inplace=True,
        )
        df["null_count"] = df.isna().sum(axis=1)
        df["регион"] = self.region_encoder.transform(df[["регион"]])
        df.fillna(-1, inplace=True)

    def transform_dict(self, dict):
        array = np.array(
            [
                self.region_dict.get(dict["регион"], -1.0),
                dict["доход"],
                dict["сегмент_arpu"],
                dict["частота"],
                dict["объем_данных"],
                dict["on_net"],
                dict["продукт_1"],
                dict["секретный_скор"],
                0.0,
            ]
        )

        array[-1] = np.isnan(array).sum()
        return array
