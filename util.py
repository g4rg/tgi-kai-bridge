import kai


class BridgeException(Exception):
    def __init__(self, model: kai.BasicError):
        self.model = model
