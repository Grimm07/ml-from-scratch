"""Example test module demonstrating test scaffolding."""


def test_example_passes():
    """A simple test that always passes."""
    assert True


def test_addition():
    """Test basic arithmetic."""
    assert 1 + 1 == 2


class TestExampleClass:
    """Example test class for grouping related tests."""

    def test_string_concatenation(self):
        """Test string operations."""
        result = "hello" + " " + "world"
        assert result == "hello world"

    def test_list_operations(self):
        """Test list operations."""
        items = [1, 2, 3]
        items.append(4)
        assert len(items) == 4
        assert items[-1] == 4
