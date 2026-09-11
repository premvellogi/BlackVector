"""
SatQuery AI - B2 Image Ontology Exceptions
"""

class OntologyError(Exception):
    """Base exception for all ontology-related errors."""
    pass


class InvalidMetadataError(OntologyError):
    """Raised when ontology image metadata fails scientific or schema constraints."""
    def __init__(self, field: str, message: str, invalid_value: any = None):
        self.field = field
        self.invalid_value = invalid_value
        super().__init__(f"Invalid metadata for '{field}': {message} (received: {invalid_value})")


class ModalityMismatchError(OntologyError):
    """Raised when an operation encounters incompatible image modalities."""
    pass


class SchemaSerializationError(OntologyError):
    """Raised when serializing or deserializing an ontology object fails."""
    pass
