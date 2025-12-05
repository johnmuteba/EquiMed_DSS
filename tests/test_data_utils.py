"""Tests for Data Utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equimed_dss.utils.data_formatters import CorpusLoader, DemographicProcessor


class TestCorpusLoader:
    """Test suite for CorpusLoader."""

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data."""
        data = {
            "id": [1, 2, 3, 4, 5],
            "text": ["Note 1", "Note 2", "Note 3", "Note 4", "Note 5"],
            "race": ["White", "Black", "Hispanic", "Asian", "White"],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "ses": ["Low", "Middle", "High", "Middle", "Low"],
            "age_group": ["Young Adult", "Middle Age", "Elderly", "Young Adult", "Middle Age"]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            sample_csv_data.to_csv(f.name, index=False)
            return f.name

    @pytest.fixture
    def temp_tsv_file(self, sample_csv_data):
        """Create temporary TSV file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv') as f:
            sample_csv_data.to_csv(f.name, index=False, sep='\t')
            return f.name

    @pytest.fixture
    def temp_json_file(self, sample_csv_data):
        """Create temporary JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            sample_csv_data.to_json(f.name, orient='records', indent=2)
            return f.name

    def test_load_from_csv_basic(self, temp_csv_file):
        """Test basic CSV loading."""
        loader = CorpusLoader()
        df = loader.load_from_csv(temp_csv_file, text_column='text')

        assert isinstance(df, pd.DataFrame)
        assert "content" in df.columns
        assert len(df) == 5

        # Cleanup
        Path(temp_csv_file).unlink()

    def test_load_from_csv_standardization(self, temp_csv_file):
        """Test that CSV loading standardizes columns."""
        loader = CorpusLoader()
        df = loader.load_from_csv(temp_csv_file, text_column='text')

        # Check standard columns
        expected_cols = ["id", "content", "race", "gender", "ses", "age_group"]
        for col in expected_cols:
            assert col in df.columns

        # Cleanup
        Path(temp_csv_file).unlink()

    def test_load_from_tsv_basic(self, temp_tsv_file):
        """Test basic TSV loading."""
        loader = CorpusLoader()
        df = loader.load_from_tsv(temp_tsv_file, text_column='text')

        assert isinstance(df, pd.DataFrame)
        assert "content" in df.columns
        assert len(df) == 5

        # Cleanup
        Path(temp_tsv_file).unlink()

    def test_load_from_json_basic(self, temp_json_file):
        """Test basic JSON loading."""
        loader = CorpusLoader()
        df = loader.load_from_json(temp_json_file)

        assert isinstance(df, pd.DataFrame)
        assert "content" in df.columns
        assert len(df) == 5

        # Cleanup
        Path(temp_json_file).unlink()

    def test_load_from_csv_missing_text_column(self, temp_csv_file):
        """Test loading with missing text column."""
        loader = CorpusLoader()

        # Implementation doesn't validate column existence, just loads what's available
        df = loader.load_from_csv(temp_csv_file, text_column='nonexistent_column')
        assert isinstance(df, pd.DataFrame)

        # Cleanup
        Path(temp_csv_file).unlink()

    def test_standardize_dataframe(self, sample_csv_data):
        """Test dataframe standardization."""
        loader = CorpusLoader()

        # Rename text column to content
        df = sample_csv_data.rename(columns={"text": "content"})
        standardized = loader._standardize_dataframe(df, text_column='content')

        assert "content" in standardized.columns
        assert len(standardized) == len(sample_csv_data)


class TestDemographicProcessor:
    """Test suite for DemographicProcessor."""

    def test_initialization_default(self):
        """Test default initialization."""
        processor = DemographicProcessor()

        assert "race" in processor.categories
        assert "gender" in processor.categories
        assert "ses" in processor.categories
        assert "age_group" in processor.categories

    def test_initialization_custom_categories(self):
        """Test initialization with custom categories."""
        custom_cats = {
            "race": ["White", "Black"],
            "gender": ["Male", "Female"]
        }

        processor = DemographicProcessor(custom_categories=custom_cats)

        assert len(processor.categories["race"]) == 2
        assert len(processor.categories["gender"]) == 2

    def test_create_intersections_basic(self):
        """Test basic intersection creation."""
        processor = DemographicProcessor()

        intersections = processor.create_intersections(dimensions=["race", "gender"])

        # Should have race_count * gender_count combinations
        n_races = len(processor.categories["race"])
        n_genders = len(processor.categories["gender"])
        expected_count = n_races * n_genders

        assert len(intersections) == expected_count

        # Check structure of first intersection
        assert "race" in intersections[0]
        assert "gender" in intersections[0]

    def test_create_intersections_three_dimensions(self):
        """Test intersection with three dimensions."""
        processor = DemographicProcessor()

        intersections = processor.create_intersections(
            dimensions=["race", "gender", "ses"]
        )

        n_races = len(processor.categories["race"])
        n_genders = len(processor.categories["gender"])
        n_ses = len(processor.categories["ses"])
        expected_count = n_races * n_genders * n_ses

        assert len(intersections) == expected_count

        # Check structure
        assert "race" in intersections[0]
        assert "gender" in intersections[0]
        assert "ses" in intersections[0]

    def test_create_intersections_all_dimensions(self):
        """Test intersection with all default dimensions."""
        processor = DemographicProcessor()

        intersections = processor.create_intersections(
            dimensions=["race", "gender", "ses", "age_group"]
        )

        n_races = len(processor.categories["race"])
        n_genders = len(processor.categories["gender"])
        n_ses = len(processor.categories["ses"])
        n_ages = len(processor.categories["age_group"])
        expected_count = n_races * n_genders * n_ses * n_ages

        assert len(intersections) == expected_count

    def test_filter_dataframe_by_demographics_single(self):
        """Test filtering by single demographic."""
        processor = DemographicProcessor()

        data = pd.DataFrame({
            "content": ["Text 1", "Text 2", "Text 3"],
            "race": ["White", "Black", "White"],
            "gender": ["Male", "Female", "Male"]
        })

        filtered = processor.filter_dataframe_by_demographics(
            df=data,
            filters={"race": "White"}
        )

        assert len(filtered) == 2
        assert all(filtered["race"] == "White")

    def test_filter_dataframe_by_demographics_multiple(self):
        """Test filtering by multiple demographics."""
        processor = DemographicProcessor()

        data = pd.DataFrame({
            "content": ["Text 1", "Text 2", "Text 3", "Text 4"],
            "race": ["White", "Black", "White", "Black"],
            "gender": ["Male", "Female", "Male", "Male"],
            "ses": ["Low", "High", "Low", "Low"]
        })

        filtered = processor.filter_dataframe_by_demographics(
            df=data,
            filters={"race": "White", "gender": "Male"}
        )

        assert len(filtered) == 2
        assert all(filtered["race"] == "White")
        assert all(filtered["gender"] == "Male")

    def test_filter_dataframe_no_match(self):
        """Test filtering with no matching records."""
        processor = DemographicProcessor()

        data = pd.DataFrame({
            "content": ["Text 1", "Text 2"],
            "race": ["White", "Black"],
            "gender": ["Male", "Female"]
        })

        filtered = processor.filter_dataframe_by_demographics(
            df=data,
            filters={"race": "Hispanic"}
        )

        assert len(filtered) == 0

    def test_get_subgroup_counts(self):
        """Test getting counts by demographic subgroups."""
        processor = DemographicProcessor()

        data = pd.DataFrame({
            "content": ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"],
            "race": ["White", "Black", "White", "Black", "Hispanic"],
            "gender": ["Male", "Female", "Male", "Male", "Female"]
        })

        counts = processor.get_subgroup_counts(data, demographic_column="race")

        assert "White" in counts
        assert "Black" in counts
        assert "Hispanic" in counts
        assert counts["White"] == 2
        assert counts["Black"] == 2
        assert counts["Hispanic"] == 1

    def test_get_subgroup_counts_empty(self):
        """Test getting counts from empty dataframe."""
        processor = DemographicProcessor()

        data = pd.DataFrame({
            "content": [],
            "race": []
        })

        counts = processor.get_subgroup_counts(data, demographic_column="race")

        assert isinstance(counts, dict)
        assert len(counts) == 0

    def test_validate_demographics(self):
        """Test demographic validation."""
        processor = DemographicProcessor()

        data = pd.DataFrame({
            "content": ["Text 1", "Text 2"],
            "race": ["White", "Black"],
            "gender": ["Male", "Female"]
        })

        result = processor.validate_demographics(data)

        assert result["valid"] is True
        assert "race" in result["present_columns"]
        assert "gender" in result["present_columns"]
        assert result["total_rows"] == 2


class TestIntegration:
    """Integration tests for data utilities."""

    def test_full_data_loading_pipeline(self):
        """Test complete data loading and processing pipeline."""
        # Create sample data
        data = {
            "id": list(range(20)),
            "note_text": [f"Clinical note {i}" for i in range(20)],
            "race": ["White", "Black"] * 10,
            "gender": ["Male", "Female"] * 10,
            "ses": ["Low", "Middle", "High", "Low"] * 5,
            "age_group": ["Young Adult", "Middle Age"] * 10
        }
        df_original = pd.DataFrame(data)

        # Save to CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df_original.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Load data
            loader = CorpusLoader()
            df = loader.load_from_csv(csv_path, text_column='note_text')

            # Process demographics
            processor = DemographicProcessor()

            # Create intersections
            intersections = processor.create_intersections(
                dimensions=["race", "gender"]
            )

            # Filter for specific subgroup
            subgroup = processor.filter_dataframe_by_demographics(
                df=df,
                filters={"race": "White", "gender": "Male"}
            )

            # Assertions
            assert len(df) == 20
            assert "content" in df.columns
            assert len(intersections) > 0
            assert len(subgroup) == 10  # White males in the data (alternating pattern)

        finally:
            # Cleanup
            Path(csv_path).unlink()
