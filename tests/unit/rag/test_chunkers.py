import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from backend.services.rag_service.src.infrastructure.chunkers.parent_child_chunker import ParentChildChunker
from backend.services.rag_service.src.infrastructure.chunkers.semantic_chunker import SemanticChunker
from backend.services.rag_service.src.infrastructure.chunkers.timeseries_chunker import TimeseriesChunker
from backend.services.rag_service.src.infrastructure.chunkers.recursive_chunker import RecursiveChunker


def test_parent_child_chunker():
    chunker = ParentChildChunker(
        parent_chunk_size=100,
        parent_overlap=10,
        child_chunk_size=20,
        child_overlap=5
    )
    
    # A realistic, short industrial manual snippet
    text = (
        "1.1 Introduction.\n"
        "This manual covers the XYZ-1000 Compressor. "
        "Maintenance should be performed every 500 hours.\n\n"
        "1.2 Filter Replacement.\n"
        "Turn off the compressor. "
        "Remove the main housing cover using a 10mm wrench. "
        "Extract the old filter. Insert the new HEPA-45 filter. "
        "Tighten bolts to 15 Nm torque."
    )
    
    metadata = {"doc_id": "manual_xyz1000"}
    pairs = chunker.chunk(text, metadata)
    
    assert len(pairs) > 0
    # Child ID should be deterministically generated from Parent ID
    assert pairs[0].child_id is not None
    assert pairs[0].parent_id is not None
    
    # Assert metadata propagation
    assert pairs[0].metadata["chunk_strategy"] == "parent_child"
    assert pairs[0].metadata["doc_id"] == "manual_xyz1000"


@patch("backend.services.rag_service.src.infrastructure.chunkers.semantic_chunker.SentenceTransformer")
def test_semantic_chunker(mock_sentence_transformer):
    # Mock the embedding model so it doesn't download weights
    mock_model = MagicMock()
    # Return random vectors
    mock_model.encode.side_effect = lambda sentences: [np.random.rand(384) for _ in sentences]
    mock_sentence_transformer.return_value = mock_model
    
    chunker = SemanticChunker(threshold=0.3)
    
    text = (
        "3.2.1 MAINTENANCE STEPS\n"
        "Ensure power is disconnected before opening the panel. "
        "The red wire connects to the positive terminal. "
        "WARNING:\n"
        "High voltage area. Do not touch capacitors."
    )
    
    metadata = {"doc_id": "test_semantic"}
    chunks = chunker.chunk(text, metadata)
    
    assert len(chunks) > 0
    assert chunks[0].metadata["chunk_strategy"] == "semantic"


def test_timeseries_chunker():
    chunker = TimeseriesChunker(window_size=5, step=2)
    
    # Simulated sensor readings (float values separated by commas)
    # e.g., 10 readings
    text = "12.0, 12.5, 13.0, 12.8, 12.2, 11.5, 10.0, 9.5, 10.2, 11.0"
    
    metadata = {"doc_id": "ts_001", "machine_id": "M1", "sensor_id": "S1"}
    chunks = chunker.chunk(text, metadata)
    
    # 10 readings, window=5, step=2 => Starts at 0, 2, 4, 6, 8 
    # Starts: 0(len 5), 2(len 5), 4(len 5), 6(len 4), 8(len 2 - skipped)
    assert len(chunks) == 3
    
    first_chunk = chunks[0]
    assert first_chunk.metadata["chunk_strategy"] == "timeseries"
    assert first_chunk.metadata["machine_id"] == "M1"
    
    # Check that statistical features were extracted
    features = first_chunk.metadata["features"]
    assert "mean" in features
    assert "std" in features
    assert "trend_slope" in features
    
    # The text should be a descriptive summary
    assert "Sensor S1 on Machine M1" in first_chunk.text


def test_recursive_chunker():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    
    text = (
        "# Main Component\n\n"
        "This is a paragraph about the main component.\n\n"
        "```python\n"
        "def start_engine():\n"
        "    pass\n"
        "```"
    )
    
    metadata = {"doc_id": "rec_001"}
    chunks = chunker.chunk(text, metadata)
    
    assert len(chunks) > 0
    assert chunks[0].metadata["chunk_strategy"] == "recursive"
    assert chunks[0].token_count > 0
