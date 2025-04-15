import pytest

from src.services.storages.base_storage import BaseStorage


class DummyStorage(BaseStorage):
    def __init__(self):
        super().__init__()
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_base_storage_context_manager_calls_close():
    storage = DummyStorage()
    async with storage as s:
        assert s is storage
        assert not storage.closed
    assert storage.closed


@pytest.mark.asyncio
async def test_base_storage_close_is_abstract():
    with pytest.raises(TypeError):

        class IncompleteStorage(BaseStorage):
            pass

        IncompleteStorage()
