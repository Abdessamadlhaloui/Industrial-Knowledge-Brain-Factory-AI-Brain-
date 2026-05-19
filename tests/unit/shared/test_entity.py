import pytest
from pydantic import ValidationError

from backend.shared.base.entity import BaseEntity


class DummyEntity(BaseEntity):
    name: str


def test_entity_equality():
    entity1 = DummyEntity(name="test")
    entity2 = DummyEntity(id=entity1.id, name="test2", created_at=entity1.created_at, updated_at=entity1.updated_at)
    
    assert entity1 == entity2
    assert hash(entity1) == hash(entity2)

def test_entity_immutability():
    entity = DummyEntity(name="test")
    
    with pytest.raises(ValidationError):
        entity.name = "new_test"

def test_entity_inequality():
    entity1 = DummyEntity(name="test")
    entity2 = DummyEntity(name="test")
    
    assert entity1 != entity2
    assert hash(entity1) != hash(entity2)
