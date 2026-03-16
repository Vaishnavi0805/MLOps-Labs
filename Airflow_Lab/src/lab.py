import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64


def load_data():
    """
    Loads Mall Customers data from CSV file, serializes it, and returns the serialized data.
    
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("Loading Mall Customers dataset...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/customers.csv"))
    print(f"Dataset loaded: {df.shape[0]} customers, {df.shape[1]} features")
    
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    
    Args:
        data_b64 (str): Base64-encoded serialized DataFrame
    
    Returns:
        str: Base64-encoded preprocessed data
    """
    print("Starting data preprocessing...")
    
    # Decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    
    # Handle missing values
    df = df.dropna()
    
    # Select features for clustering: Annual Income and Spending Score
    clustering_data = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    
    print(f"Selected features for clustering: {clustering_data.columns.tolist()}")
    
    # Normalize data using MinMaxScaler
    min_max_scaler = MinMaxScaler()
    clustering_data_normalized = min_max_scaler.fit_transform(clustering_data)
    
    print("Data preprocessing completed")
    
    # Serialize and encode
    clustering_serialized_data = pickle.dumps(clustering_data_normalized)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans clustering model, saves it, and returns SSE values for elbow method.
    
    Args:
        data_b64 (str): Base64-encoded preprocessed data
        filename (str): Filename to save the model
    
    Returns:
        list: SSE (Sum of Squared Errors) values for different k values
    """
    print("Building KMeans clustering model...")
    
    # Decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    clustering_data = pickle.loads(data_bytes)
    
    # KMeans parameters
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42
    }
    
    # Calculate SSE for different k values (elbow method)
    sse = []
    k_range = range(1, 11)  # Test k from 1 to 10
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(clustering_data)
        sse.append(kmeans.inertia_)
        print(f"  k={k}, SSE={kmeans.inertia_:.2f}")
    
    # Build final model with optimal k (will be determined by elbow method)
    # For now, use k=5 as default
    optimal_k = 5
    final_kmeans = KMeans(n_clusters=optimal_k, **kmeans_kwargs)
    final_kmeans.fit(clustering_data)
    
    # Save the model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(final_kmeans, f)
    
    print(f"Model saved to: {output_path}")
    
    return sse  # List is JSON-safe for XCom


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to determine optimal number of clusters.
    
    Args:
        filename (str): Path to saved model
        sse (list): SSE values from build_save_model
    
    Returns:
        dict: Results including optimal k and cluster centers
    """
    print("Loading model and determining optimal clusters...")
    
    # Load the saved model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))
    
    # Use elbow method to find optimal k
    kl = KneeLocator(
        range(1, 11), 
        sse, 
        curve="convex", 
        direction="decreasing"
    )
    
    optimal_k = kl.elbow if kl.elbow else 5
    
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Model has {loaded_model.n_clusters} clusters")
    print(f"Cluster centers shape: {loaded_model.cluster_centers_.shape}")
    
    result = {
        "optimal_k": int(optimal_k),
        "model_clusters": int(loaded_model.n_clusters),
        "inertia": float(loaded_model.inertia_)
    }
    
    print(f"Results: {result}")
    
    return result