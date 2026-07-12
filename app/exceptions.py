class GeneratorError(Exception): pass

class ModelLoadingError(GeneratorError): pass
class EmptyGeneratorError(GeneratorError): pass