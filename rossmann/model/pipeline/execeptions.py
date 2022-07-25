class StoreNotFoundException(ValueError):
    def __init__(self, store: int):
        self.store = store
        super(f"Store: {store} not found")
