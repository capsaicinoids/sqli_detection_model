import json

class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)

class Config:
    def __init__(self, data, train, model):
        self.data = data
        self.train = train
        self.model = model
    
    @classmethod
    def from_json(cls, json_str):
        """
        Construct config from json string.

        Args:
            json_str (str): The JSON string representing the config.

        Returns:
            Config: The constructed Config object.
        """

        params = json.loads(json.dumps(json_str), object_hook=HelperObject)

        # Create and return a Config object using the data, train, and model parameters from the HelperObject
        return cls(params.data, params.train, params.model)
        
    @classmethod 
    def from_yaml(cls, yaml_str):
        pass
