"""
Data format converters and loaders for EquiMed_DSS library.

This module provides utilities to load data from various sources (MySQL, CSV, TSV, JSON)
and convert them to the standard EquiMed_DSS format for analysis.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class CorpusLoader:
    """
    Load and standardize data from multiple formats.

    Supports:
    - MySQL databases
    - CSV files
    - TSV files
    - JSON files
    """

    def __init__(self):
        self.standard_schema = {"corpus_type": str, "documents": list, "metadata": dict}

    def load_from_mysql(
        self,
        config: Dict[str, Any],
        query: str,
        text_column: str = "content",
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load corpus from MySQL database.

        Args:
            config: Database configuration dict with keys: host, user, password, database
            query: SQL query to retrieve documents
            text_column: Name of column containing text content
            id_column: Name of column containing document IDs (optional)

        Returns:
            DataFrame with standardized schema

        Example:
            >>> config = {'host': 'localhost', 'user': 'user', 'password': 'pass', 'database': 'db'}
            >>> query = "SELECT id, fullcontent FROM documents WHERE content IS NOT NULL"
            >>> df = loader.load_from_mysql(config, query, text_column='fullcontent')
        """
        try:
            import mysql.connector
        except ImportError:
            raise ImportError(
                "mysql-connector-python is required for MySQL support. "
                "Install with: pip install mysql-connector-python"
            )

        connection = mysql.connector.connect(**config)
        df = pd.read_sql(query, connection)
        connection.close()

        return self._standardize_dataframe(df, text_column, id_column)

    def load_from_csv(
        self,
        file_path: Union[str, Path],
        text_column: str = "content",
        id_column: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load corpus from CSV file.

        Args:
            file_path: Path to CSV file
            text_column: Name of column containing text content
            id_column: Name of column containing document IDs (optional)
            **kwargs: Additional arguments passed to pd.read_csv()

        Returns:
            DataFrame with standardized schema
        """
        df = pd.read_csv(file_path, **kwargs)
        return self._standardize_dataframe(df, text_column, id_column)

    def load_from_tsv(
        self,
        file_path: Union[str, Path],
        text_column: str = "content",
        id_column: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load corpus from TSV file.

        Args:
            file_path: Path to TSV file
            text_column: Name of column containing text content
            id_column: Name of column containing document IDs (optional)
            **kwargs: Additional arguments passed to pd.read_csv()

        Returns:
            DataFrame with standardized schema
        """
        df = pd.read_csv(file_path, sep="\t", **kwargs)
        return self._standardize_dataframe(df, text_column, id_column)

    def load_from_json(
        self, file_path: Union[str, Path], orient: str = "records"
    ) -> pd.DataFrame:
        """
        Load corpus from JSON file.

        Args:
            file_path: Path to JSON file
            orient: JSON orientation ('records', 'index', 'columns')

        Returns:
            DataFrame with standardized schema
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "documents" in data:
            df = pd.DataFrame(data["documents"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.read_json(file_path, orient=orient)

        return self._standardize_dataframe(df)

    def _standardize_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Standardize DataFrame to EquiMed_DSS schema."""
        result = pd.DataFrame()

        # Handle ID column
        if id_column and id_column in df.columns:
            result["id"] = df[id_column]
        elif "id" in df.columns:
            result["id"] = df["id"]
        else:
            result["id"] = [f"doc_{i}" for i in range(len(df))]

        # Handle content column
        if text_column in df.columns:
            result["content"] = df[text_column]
        elif "content" in df.columns:
            result["content"] = df["content"]
        elif "text" in df.columns:
            result["content"] = df["text"]
        else:
            raise ValueError(
                f"Could not find text column. Available columns: {list(df.columns)}"
            )

        # Add source/corpus type
        if "source" in df.columns:
            result["source"] = df["source"]
        else:
            result["source"] = "unknown"

        # Preserve demographic columns if present
        demographic_cols = [
            "race",
            "gender",
            "age_group",
            "socioeconomic_status",
            "ses",
            "ethnicity",
            "age",
            "income",
        ]
        for col in demographic_cols:
            if col in df.columns:
                result[col] = df[col]

        return result

    def validate_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame against EquiMed_DSS requirements.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with validation results and suggestions
        """
        results = {"valid": True, "errors": [], "warnings": [], "info": {}}

        # Check required columns
        required = ["id", "content"]
        for col in required:
            if col not in df.columns:
                results["valid"] = False
                results["errors"].append(f"Missing required column: {col}")

        # Check for empty content
        if "content" in df.columns:
            empty_count = df["content"].isna().sum()
            if empty_count > 0:
                results["warnings"].append(
                    f"{empty_count} documents have empty content"
                )

        # Check demographics
        demographic_cols = ["race", "gender", "age_group", "ses"]
        present_demographics = [col for col in demographic_cols if col in df.columns]
        results["info"]["demographics_present"] = present_demographics

        if not present_demographics:
            results["warnings"].append(
                "No demographic columns found. "
                "Fairness metrics require demographic information."
            )

        results["info"]["total_documents"] = len(df)
        results["info"]["columns"] = list(df.columns)

        return results


class DemographicProcessor:
    """Process and generate demographic combinations for intersectional analysis."""

    DEFAULT_CATEGORIES = {
        "race": ["White", "Black", "Hispanic", "Asian", "Native American"],
        "gender": ["Male", "Female", "Non-binary"],
        "ses": ["Low", "Middle", "High"],
        "age_group": ["Young Adult", "Middle Age", "Elderly"],
    }

    def __init__(self, custom_categories: Optional[Dict[str, List[str]]] = None):
        """
        Initialize demographic processor.

        Args:
            custom_categories: Optional dict of custom demographic categories
        """
        self.categories = custom_categories or self.DEFAULT_CATEGORIES.copy()

    def extract_demographics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Extract unique demographic categories from DataFrame.

        Args:
            df: DataFrame with demographic columns

        Returns:
            Dict mapping demographic dimension to unique values
        """
        demographics = {}

        for dim in ["race", "gender", "ses", "age_group", "socioeconomic_status"]:
            if dim in df.columns:
                unique_vals = df[dim].dropna().unique().tolist()
                if unique_vals:
                    key = "ses" if dim == "socioeconomic_status" else dim
                    demographics[key] = sorted(unique_vals)

        return demographics

    def create_intersections(
        self, dimensions: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate all intersectional combinations.

        Args:
            dimensions: List of demographic dimensions to combine.
                       If None, uses all available dimensions.

        Returns:
            List of dicts, each representing one intersectional group

        Example:
            >>> processor = DemographicProcessor()
            >>> combinations = processor.create_intersections(['race', 'gender'])
            >>> len(combinations)  # 5 races * 3 genders = 15
            15
        """
        from itertools import product

        if dimensions is None:
            dimensions = list(self.categories.keys())

        # Get categories for each dimension
        dimension_values = [self.categories[dim] for dim in dimensions]

        # Generate cartesian product
        combinations = []
        for combo in product(*dimension_values):
            combo_dict = dict(zip(dimensions, combo))
            combinations.append(combo_dict)

        return combinations

    def generate_query_combinations(
        self, template: str, dimensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate queries for all intersectional combinations.

        Args:
            template: Query template with placeholders like {race}, {gender}, {ses}
            dimensions: Demographic dimensions to include

        Returns:
            List of query dicts with filled templates

        Example:
            >>> template = "Chest pain in {race} {gender} patient"
            >>> queries = processor.generate_query_combinations(template, ['race', 'gender'])
        """
        intersections = self.create_intersections(dimensions)

        queries = []
        for i, demo_dict in enumerate(intersections):
            query_text = template.format(**demo_dict)
            query_id = "_".join(
                f"{k}_{v.lower().replace(' ', '_')}" for k, v in demo_dict.items()
            )

            queries.append(
                {
                    "id": query_id,
                    "query": query_text,
                    "demographics": demo_dict,
                    "index": i,
                }
            )

        return queries

    def validate_demographic_categories(
        self, df: pd.DataFrame, required_dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate demographic completeness in DataFrame.

        Args:
            df: DataFrame to validate
            required_dimensions: List of required demographic dimensions

        Returns:
            Validation results dict
        """
        if required_dimensions is None:
            required_dimensions = ["race", "gender"]

        results = {
            "valid": True,
            "missing_dimensions": [],
            "incomplete_dimensions": {},
            "coverage": {},
        }

        for dim in required_dimensions:
            if dim not in df.columns:
                results["valid"] = False
                results["missing_dimensions"].append(dim)
            else:
                missing_count = df[dim].isna().sum()
                total = len(df)
                coverage = (total - missing_count) / total if total > 0 else 0

                results["coverage"][dim] = coverage

                if coverage < 1.0:
                    results["incomplete_dimensions"][dim] = {
                        "missing_count": int(missing_count),
                        "coverage_percent": round(coverage * 100, 2),
                    }

        return results


def convert_to_standard_format(
    data: Union[pd.DataFrame, Dict, List],
    corpus_type: str = "unknown",
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convert any data format to standard EquiMed_DSS JSON format.

    Args:
        data: Input data (DataFrame, dict, or list)
        corpus_type: Type of corpus (peer_reviewed, community, clinical)
        metadata: Additional metadata dict

    Returns:
        Standard format dict
    """
    if isinstance(data, pd.DataFrame):
        documents = data.to_dict("records")
    elif isinstance(data, dict) and "documents" in data:
        documents = data["documents"]
    elif isinstance(data, list):
        documents = data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    return {
        "corpus_type": corpus_type,
        "documents": documents,
        "metadata": metadata or {},
        "total_documents": len(documents),
        "schema_version": "1.0",
    }
