import numpy as np
import pandas as pd
import arcpy
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import sys
import os

# Thiết lập environment cho ArcGIS Pro
arcpy.env.overwriteOutput = True

FEATURES = [
    'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
    'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
    'slope', 'twi', 'NDVI', 'rainfall'
]

FEATURE_MIN_MAX = {
    'lulc': (0.0, 12.0),
    'Density_River': (0.0, 0.000675744),
    'Density_Road': (0.0, 16.5452),
    'Distan2river_met': (0.0, 12407.1),
    'Distan2road_met': (0.0, 14716.4),
    'aspect': (-1.0, 360.0),
    'curvature': (-17.4153, 16.4418),
    'dem': (-21.0, 1756.0),
    'flowDir': (0.0, 255.0),
    'slope': (0.0, 68.5592),
    'twi': (-0.94, 21.0),
    'NDVI': (-0.186454, 0.599315),
    'rainfall': (196.525, 1292.31)
}

RF_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 50,
    'min_samples_split': 20,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': False,
    'max_leaf_nodes': 1000,
}


def map_features(columns):
    return {col: col for col in columns if col in FEATURES}


def train_model():
    try:
        csv_path = r"Z:\guest01\QuanVX\25-26_HKI_DATN_QuanVX\train\data\training_points.csv"
        arcpy.AddMessage(f"Loading training data from: {csv_path}")
        df = pd.read_csv(csv_path).dropna()
        feature_columns = [col for col in df.columns if col != 'flood']
        X, y = df[feature_columns].values, df['flood'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        arcpy.AddMessage("Training RandomForest model...")
        model = RandomForestRegressor(**RF_PARAMS, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        arcpy.AddMessage(f"Model trained. R2 Score: {score:.4f}")
        return model
    except Exception as e:
        arcpy.AddError(f"Error in train_model: {str(e)}")
        return None


def normalize_features(features, feature_names):
    mins = np.array([FEATURE_MIN_MAX[f][0] for f in feature_names])
    maxs = np.array([FEATURE_MIN_MAX[f][1] for f in feature_names])
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1.0, ranges)
    normalized = (features - mins) / ranges
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.astype(np.float32)


def load_random_sample(file_path, layer_name, sample_size=2000000):
    try:
        feature_class = str(file_path) + "\\" + layer_name
        arcpy.AddMessage(f"Reading from: {feature_class}")
        if not arcpy.Exists(feature_class):
            arcpy.AddError(f"Feature class not found: {feature_class}")
            return np.array([]), np.array([])
        field_names = [field.name for field in arcpy.ListFields(feature_class) if field.type not in ['OID', 'Geometry']]
        features_map = map_features(field_names)
        if not features_map:
            arcpy.AddError("No valid features found!")
            return np.array([]), np.array([])
        feature_cols = list(features_map.keys())
        arcpy.AddMessage(f"Found features: {feature_cols}")
        total_count = int(arcpy.GetCount_management(feature_class)[0])
        arcpy.AddMessage(f"Total records: {total_count:,}")
        coords_list, features_list, valid_count = [], [], 0
        arcpy.AddMessage(f"Using reservoir sampling for {sample_size:,} from {total_count:,} records...")
        with arcpy.da.SearchCursor(feature_class, feature_cols + ['SHAPE@XY']) as cursor:
            for idx, row in enumerate(cursor):
                if idx % 100000 == 0:
                    arcpy.AddMessage(f"  Processed {idx:,} records, sampled {len(features_list):,}")
                xy = row[-1]
                if xy is not None:
                    feature_values = row[:-1]
                    if None not in feature_values:
                        valid_count += 1
                        if len(features_list) < sample_size:
                            coords_list.append([xy[0], xy[1]])
                            features_list.append(feature_values)
                        else:
                            replace_idx = random.randint(1, valid_count)
                            if replace_idx <= sample_size:
                                coords_list[replace_idx - 1] = [xy[0], xy[1]]
                                features_list[replace_idx - 1] = feature_values
        if len(coords_list) == 0:
            arcpy.AddError("No valid data found!")
            return np.array([]), np.array([])
        coords, features = np.array(coords_list), np.array(features_list)
        features = normalize_features(features, feature_cols)
        arcpy.AddMessage(f"Successfully loaded {len(features):,} valid samples")
        return coords, features
    except Exception as e:
        arcpy.AddError(f"Error in load_random_sample: {str(e)}")
        return np.array([]), np.array([])


def main():
    try:
        arcpy.AddMessage("=== FLOOD PREDICTION - ArcGIS Pro VERSION ===")
        arcpy.AddMessage("\n1. Training model from CSV...")
        model = train_model()
        if model is None:
            return
        arcpy.AddMessage("\n2. Loading GDB data...")
        file_path = Path(r"Z:\guest01\QuanVX\QuanVX\QuanVX\Default.gdb")
        layer_name = "RasterT_Extract1"
        coords, features = load_random_sample(file_path, layer_name)
        if len(features) > 0:
            arcpy.AddMessage("\n3. Making predictions...")
            predictions = model.predict(features)
            arcpy.AddMessage(f"\nPrediction Results: Count: {len(predictions):,}, Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
            output_file = Path(__file__).parent.parent / "results" / "sample_predictions.csv"
            output_file.parent.mkdir(exist_ok=True)
            df_output = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'flood_probability': predictions})
            df_output.to_csv(output_file, index=False)
            arcpy.AddMessage(f"\n4. Results saved to: {output_file}")
            arcpy.AddMessage("\n=== COMPLETED SUCCESSFULLY ===")
        else:
            arcpy.AddError("No valid data loaded!")
    except Exception as e:
        arcpy.AddError(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()