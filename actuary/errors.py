
class ArrayDifferentSizesError(ValueError):
    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
class InvalidPeriodCombinationError(ValueError):
    def __init__(self, origin_per, dev_per, *args, **kwargs):
        self.origin_per = origin_per
        self.dev_per = dev_per
        self.message = f'The origin period {self.origin_per} should be divisible by the development period {self.dev_per}.'
        super().__init__(self.message, *args, **kwargs)
