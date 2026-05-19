import sys
import os
import warnings
warnings.filterwarnings("ignore")  # Supprimer les warnings numpy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

def compute_drift(reference_df, current_df, numerical_features=None, categorical_features=None):
    if numerical_features is None:
        numerical_features = reference_df.select_dtypes(include=['number']).columns.tolist()
    if categorical_features is None:
        categorical_features = reference_df.select_dtypes(include=['object']).columns.tolist()

    column_mapping = ColumnMapping(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target=None
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
    return report

def main():
    train_path = "data/processed/test_features_drifted.csv"
    test_path = "data/processed/test_features.csv"
    output_html = "reports/drift_report.html"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Fichiers de données manquants.")
        return

    ref = pd.read_csv(train_path)
    cur = pd.read_csv(test_path)

    report = compute_drift(ref, cur)
    report.save_html(output_html)
    print(f"Rapport de drift sauvegardé dans {output_html}")

if __name__ == "__main__":
    main()