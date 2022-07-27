class StoreNotFoundException(ValueError):
    def __init__(self, store: int):
        self.store = store
        super().__init__(f"Store: {store} not found")
