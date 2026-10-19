from pathlib import Path
import pandas as pd

DEFAULT_DATA_DIR = Path("data")


class DataLoader:

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)

    def load_customers(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "customers.csv")

    def load_orders(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / "orders.csv")
        df["order_datetime"] = pd.to_datetime(df["order_datetime"])
        return df

    def load_order_items(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / "orders_items.csv")
        time_cols = [
            "availability_start",
            "availability_end",
            "delivery_start",
            "delivery_end",
        ]
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df

    def load_producers(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "producers.csv")

    def load_locations(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "locations.csv")

    def load_distance_matrix(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "distance_matrix.csv")

    def load_localities(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "localities.csv")

    def load_products(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "products.csv")

    def load_vehicles(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "vehicles.csv")

    def load_delivery_locations(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "delivery_locations.csv")

    def load_collection_centers(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "collection_centers.csv")